"""Run full Hill Climb matches against the simulated relay table and score each one with the referee.

The harness runs unchanged (Runner + a match plan) against sim_station.py --scene hill_climb. Its verifier is
an oracle that answers each stage's question from the sim's ground truth (/objects), so the matches test the
harness's decisions and the policy's behaviour, not camera judgement. Each match ends with the referee's
rubric v4 score, DQ and 8-ball status next to what the harness believed.

    python sim_station.py --policy dream --scene hill_climb            # terminal 1
    python experiments/sim_match.py --matches 3 --out ../runs_sim_match  # terminal 2

Sim only: it calls /reset before each match.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.frames import FrameSource  # noqa: E402
from harness.log import RunLog  # noqa: E402
from harness.plan import Stage, load_plan  # noqa: E402
from harness.runner import Runner  # noqa: E402
from harness.station_client import StationClient  # noqa: E402
from harness.verifiers import MonitorVerdict  # noqa: E402
from sim_scenes import hill_climb_table as h  # noqa: E402

CLEAR_M = 0.08          # a TCP this far from an object counts as clear of it
OPEN = 0.6              # gripper opening (0-1) that counts as open


def get(base: str, route: str) -> dict:
    with urllib.request.urlopen(base + route, timeout=5) as resp:
        return json.loads(resp.read())


class OracleVerifier:
    """Answers plans/hill_climb_v4.yaml's stage questions, and its black-ball guard, from sim ground truth.
    "Not met yet" is done=None with status ok, as the real monitor reports it. Unknown stages: can't tell."""

    def __init__(self, station_url: str) -> None:
        self.url = station_url
        self.start: dict[str, dict] = {}   # (referee report, snapshot) when the stage was first checked

    def _near(self, snap: dict, xyz: list[float]) -> float:
        return min(math.dist(snap["tcp"][side], xyz) for side in ("left", "right"))

    def _open(self, snap: dict) -> bool:
        return all(snap["joints"][side][6] >= OPEN for side in ("left", "right"))

    def _blocks(self, bodies: dict) -> list[str]:
        return [b for b in bodies if b.startswith(("cube_", "cylinder_"))]

    def _tower_clear(self, snap: dict) -> bool:
        """Grippers open and away from the highest block: the tower stands on its own."""
        bodies = snap["bodies"]
        top = max(self._blocks(bodies), key=lambda b: bodies[b][2])
        return self._open(snap) and self._near(snap, bodies[top]) > 0.05

    def _holding_eight(self, snap: dict) -> bool:
        """Guard: an 8-ball lifted off the table right at a gripper."""
        return any(b.startswith("eight_") and pos[2] > h.TABLE_Z + h.EIGHT_R + 0.01 and self._near(snap, pos) < 0.05
                   for b, pos in snap["bodies"].items())

    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> MonitorVerdict:
        snap = get(self.url, "/objects")
        ref, bodies = snap["referee"], snap["bodies"]
        base = self.start.setdefault(stage.name, ref)
        name = stage.name
        tower = ref["tower_in"] * h.INCH
        held_out = [b for b in bodies if b.startswith("plugged_") and self._near(snap, bodies[b]) <= 0.05]
        if name in ("unplug_cable", "unplug_more"):
            done = ref["unplugged"] > base["unplugged"]
        elif name in ("put_cable_down", "put_more_down"):
            done = self._open(snap) and not held_out
        elif name in ("plug_1", "plug_2"):
            done = ref["seated"] > base["seated"] and self._open(snap)
        elif name == "arms_clear":
            done = (self._open(snap) and self._near(snap, bodies["hub"]) > CLEAR_M
                    and all(snap["tcp"][s][2] > h.TABLE_Z + 0.10 for s in ("left", "right")))
        elif name == "stack_2":
            done = tower >= 3 * h.CUBE_HALF and self._tower_clear(snap)
        elif name == "stack_3":
            done = tower >= 5 * h.CUBE_HALF and self._tower_clear(snap)
        elif name == "tower_pass":
            done = tower > h.TOWER_BAR_M and self._tower_clear(snap)
        elif name.startswith("higher_"):
            done = tower >= base["tower_in"] * h.INCH + 0.04 and self._tower_clear(snap)
        elif name.startswith("ball_"):
            need = int(name.split("_")[1])
            done = ref["golf_in_bin"] >= need and not ref["eight_in_bin"]
        else:
            return MonitorVerdict(None, 0.0, f"oracle: no rule for {name}")
        verdict = MonitorVerdict(True if done else None, 1.0, f"oracle: {name} {'met' if done else 'not met'}",
                                 status="ok", scene_ok=self._open(snap))
        guard = getattr(self, "_plan_guard", None)
        if guard is not None and guard.covers(name):
            verdict.guard = self._holding_eight(snap)
        return verdict


def run_match(args: argparse.Namespace, n: int, out: Path) -> dict[str, Any]:
    station = StationClient(args.station, timeout_s=10.0)
    if station.status().get("state") not in ("idle", "error"):
        station.stop()
        station.wait_for_state({"idle", "error"}, 60)
    urllib.request.urlopen(urllib.request.Request(args.station + "/reset", data=b"{}", method="POST",
                                                  headers={"Content-Type": "application/json"}), timeout=60).read()
    station.wait_for_state({"idle"}, 60)
    plan = load_plan(args.plan, dict(v.split("=", 1) for v in args.var))
    log = RunLog(out)
    t0 = time.time()
    oracle = OracleVerifier(args.station)
    oracle._plan_guard = plan.guard
    result = Runner(plan, station, FrameSource(args.view), {"monitor": oracle}, log).run()
    referee = get(args.station, "/objects")["referee"]
    summary = {"match": n, "wall_s": round(time.time() - t0, 1), "log": str(log.path),
               "harness": {k: v for k, v in asdict(result).items() if k != "stages"},
               "stages": [asdict(s) for s in result.stages], "referee": referee}
    (log.path / "referee.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="http://127.0.0.1:8091")
    ap.add_argument("--view", default="http://127.0.0.1:8093")
    ap.add_argument("--plan", default=str(Path(__file__).resolve().parents[1] / "plans" / "hill_climb_v4.yaml"))
    ap.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--out", default="runs_sim_match")
    args = ap.parse_args()
    status = get(args.station, "/status")
    if not status.get("sim") or "referee" not in get(args.station, "/objects"):
        raise SystemExit("refusing to run: needs sim_station.py --scene hill_climb (this script resets the scene)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in range(1, args.matches + 1):
        print(f"match {n} ...", flush=True)
        s = run_match(args, n, out)
        with (out / "matches.jsonl").open("a") as f:
            f.write(json.dumps(s) + "\n")
        r, hr = s["referee"], s["harness"]
        print(f"  harness ok={hr['ok']} guard={hr.get('guard_tripped')} ({hr['reason']}); referee points={r['points']} "
              f"dq={r['dq']} 8ball_in_bin={r['eight_in_bin']} unplugged={r['unplugged']} tower={r['tower_in']} in "
              f"golf={r['golf_in_bin']}", flush=True)


if __name__ == "__main__":
    main()
