"""`python -m harness plans/x.yaml (--dry-run | --live | --mock)`."""

from __future__ import annotations

import argparse
import hashlib
import signal
import socket
import sys
from pathlib import Path

from . import __version__
from .frames import FrameSource
from .log import RunLog
from .plan import load_plan
from .runner import Runner
from .station_client import StationClient, StationError
from .verifiers import ManualVerifier, Verifier, make_verifier

SAFETY = """\
LIVE RUN. This drives the real robot through the station agent.
  - Keep your hand on the e-stop for the whole run.
  - Make sure the workspace is clear of people and loose objects.
  - Ctrl+C (or typing s + Enter) stops the run; the arms hold where they are."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m harness", description="Hill Climb harness for Dream-1")
    p.add_argument("plan", help="path to a plan YAML file")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="read status/frames only; send no POSTs")
    mode.add_argument("--live", action="store_true", help="real run (asks you to type yes)")
    mode.add_argument("--mock", action="store_true", help="run against an in-process fake station")
    p.add_argument("--station", default="http://127.0.0.1:8091", help="station API URL")
    p.add_argument("--view", default="http://127.0.0.1:8093", help="debug view URL")
    p.add_argument("--runs-dir", default="runs", help="where run logs go")
    p.add_argument("--scenario", default="happy", help="fake station scenario (--mock only)")
    p.add_argument("--detect", default=None, metavar="COLORS.yaml",
                   help="local ball/block detector (e.g. plans/table_colors.yaml): fast black-ball guard + boxes on "
                        "the monitor's frames")
    p.add_argument("--verifier", choices=["manual", "claude", "time", "claude+manual", "monitor", "monitor+manual"], default=None,
                   help="override every stage's verifier")
    p.add_argument("--var", action="append", default=[], metavar="NAME=VALUE",
                   help="fill a {NAME} placeholder in the plan, e.g. --var color=orange (repeatable)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    variables: dict[str, str] = {}
    for item in args.var:
        name, sep, value = item.partition("=")
        if not sep or not name.strip() or not value.strip():
            print(f"bad --var {item!r}: use NAME=VALUE", file=sys.stderr)
            return 2
        variables[name.strip()] = value.strip()
    try:
        plan = load_plan(args.plan, variables)
    except (ValueError, OSError) as exc:
        print(f"plan error: {exc}", file=sys.stderr)
        return 2
    if args.verifier:
        for stage in plan.stages:
            stage.verifier = args.verifier
    if plan.guard is not None and any(st.verifier not in ("monitor", "monitor+manual")
                                      for st in plan.stages if plan.guard.covers(st.name)):
        print("WARNING: plan.guard is only answered by the monitor; guarded stages without it are unguarded",
              file=sys.stderr)

    fake = None
    station_url, view_url = args.station, args.view
    if args.mock:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import fake_station  # noqa: PLC0415 (path set just above)

        fake = fake_station.FakeStation(scenario=args.scenario)
        fake.start()
        station_url, view_url = fake.api_url, fake.view_url
        print(f"mock station ({args.scenario}) at {station_url}")

    try:
        if args.live:
            print(SAFETY)
            if input('Type "yes" to start: ').strip() != "yes":
                print("not started.")
                return 1

        log = RunLog(args.runs_dir)
        print(f"logging to {log.path}")
        plan_path = Path(args.plan).resolve()
        log.update_manifest(
            plan_file=str(plan_path), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            harness_version=__version__, args={k: v for k, v in vars(args).items()},
            host=socket.gethostname())
        dry = args.dry_run
        station = StationClient(station_url, dry_run=dry, log=lambda m: (print(m), log.event("dry_run", message=m))[0])
        frames = FrameSource(view_url)

        holder: dict[str, Runner] = {}
        manual = None
        verifiers: dict[str, Verifier] = {}
        try:
            # One stdin reader serves y/n answers and the s + Enter stop.
            manual = ManualVerifier(on_stop=lambda: holder["r"].request_stop())
            for name in {s.verifier for s in plan.stages} | {"manual"}:
                verifiers[name] = make_verifier(name, plan, manual=manual)
        except ValueError as exc:
            print(f"verifier error: {exc}", file=sys.stderr)
            return 2

        runner = Runner(plan, station, frames, verifiers, log)
        if args.detect:
            from .detect import TableDetector  # noqa: PLC0415 (numpy + Pillow only when asked for)
            det = TableDetector.from_file(args.detect)
            runner.fast_guard = det.guard
            for v in verifiers.values():
                for m in (v, getattr(v, "second", None)):
                    if hasattr(m, "annotator"):
                        m.annotator, m.legend = det.annotate_frames, det.legend()
            log.event("detector", config=str(args.detect))
        holder["r"] = runner
        # Turn SIGTERM into the same clean stop as Ctrl+C.
        try:
            signal.signal(signal.SIGTERM, lambda *_: runner.request_stop())
        except ValueError:
            pass
        mode = "DRY RUN" if dry else ("MOCK" if args.mock else "LIVE")
        print(f"{mode}: {plan.task or '(no task)'}  [{len(plan.stages)} stages]  type s + Enter to stop")
        try:
            result = runner.run()
        except StationError as exc:
            print(f"station error: {exc.message}", file=sys.stderr)
            return 1
        for r in result.stages:
            print(f"  {'ok  ' if r.ok else 'FAIL'} {r.name:<20} attempts={r.attempts} {r.seconds:6.1f}s  {r.reason}")
        print(f"{'SUCCESS' if result.ok else 'FAILED'}: {result.reason}  ({result.seconds:.1f}s)  log: {log.path}")
        if result.guard_tripped and plan.guard is not None:
            bar = "!" * (len(plan.guard.message) + 8)
            print(f"\a\a\n{bar}\n!   {plan.guard.message}   !\n{bar}\n")
        if result.ok and plan.done_message:
            bar = "#" * (len(plan.done_message) + 8)
            print(f"\a\n{bar}\n#   {plan.done_message}   #\n{bar}\n")
        if not dry and not args.mock:
            print("The arms hold where they are. Home them with: dreamcli robot down")
        return 0 if result.ok else 1
    finally:
        if fake is not None:
            fake.close()
