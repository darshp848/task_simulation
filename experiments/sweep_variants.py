"""A/B harness and plan variants on the offline stress sim: same scenarios, same seeds, same monitor latency draws.

A variant is a set of harness constants (harness.runner module globals) and plan edits applied to
plans/hill_climb_v4.yaml. Every variant plays every (scenario, seed), so the differences are the variant's.

    python experiments/sweep_variants.py base inflight --seeds 4 -j 12 --latency 6.5-13
    python experiments/sweep_variants.py --list
    python experiments/sweep_variants.py base inflight -s happy -s slow_unplug --seeds 8 --out runs_tuning/ab.jsonl

Latency: monitor seconds per call, or a range drawn per call. 6.5-13 is what Haiku 5.5 measured on 7 Nov
(tools/check_monitor.py: median 8.4 s, max 12.7 s); the stress sim's own default is 4 s.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import stress_match as sm  # noqa: E402

OUT = HERE.parent / "runs_tuning"

# name -> {"runner": {CONSTANT: value}, "top": {plan key: value}, "stages": {stage or "*skip"/"*": {field: value}},
#          "drop": [stage names], "note": "..."}
CHEAP_ENTRY = ("release_cable", "put_more_down", "arms_clear")   # stages whose entry check only saves <= 12 s
VARIANTS: dict[str, dict[str, Any]] = {
    "base": {"note": "plans/hill_climb_v4.yaml as is"},
    "no_entry": {"stages": {n: {"skip_if_done": False} for n in CHEAP_ENTRY},
                 "note": "no skip_if_done entry check (a synchronous monitor call) on the 10-12 s stage-1 tail stages"},
    "no_more": {"drop": ["unplug_more", "put_more_down"],
                "note": "no second unplug (+5): its ~30 s go to the tower and balls"},
    "late_check": {"stages": {"put_cable_down": {"min_run_s": 6}, "put_more_down": {"min_run_s": 6}},
                   "note": "first put-down check after the put-down should be over (teleop median 4.8 s)"},
    "lean": {"drop": ["unplug_more", "put_more_down"],
             "stages": {n: {"skip_if_done": False} for n in CHEAP_ENTRY},
             "note": "no_more + no_entry"},
}


def edit_plan(spec: dict[str, Any]) -> dict[str, Any]:
    data = yaml.safe_load(sm.PLAN.read_text())
    data.update(copy.deepcopy(spec.get("top", {})))
    drop = set(spec.get("drop", []))
    data["stages"] = [s for s in data["stages"] if s["name"] not in drop]
    for st in data["stages"]:
        for key, fields in spec.get("stages", {}).items():
            hit = (key == st["name"] or key == "*" or (key == "*skip" and st.get("on_fail") == "skip")
                   or (key == "*retry" and st.get("on_fail") != "skip"))
            if hit:
                for k, v in fields.items():
                    st[k] = v(st.get(k)) if callable(v) else v
    return data


def _one(job: tuple[str, str, int, Any, str | None]) -> dict[str, Any]:
    variant, name, seed, latency, plan_path = job
    import harness.runner as runner
    for k, v in VARIANTS[variant].get("runner", {}).items():
        setattr(runner, k, v)
    sc = dataclasses.replace(sm.SCENARIOS[name], latency_s=latency)
    if plan_path and sc.plan == sm.PLAN:     # scenarios with their own plan keep it
        sc = dataclasses.replace(sc, plan=Path(plan_path))
    r = sm.run_match(sc, seed)
    return {"variant": variant, "scenario": name, "seed": seed, "points": r["points"], "lost": r["lost"],
            "dq": bool(r["dq"]), "black": r["black_in_bin"], "replugged": r["replugged"], "t_end": r["t_end"],
            "out": r["out"], "tower": r["tower"], "balls": r["balls"], "harness_ok": r["harness_ok"],
            "reason": r["reason"][:120], "monitor_calls": r["monitor_calls"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="*", default=["base", "inflight"])
    ap.add_argument("-s", "--scenario", action="append", help="scenario (repeatable); default all")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--latency", type=sm.parse_latency, default=(6.5, 13.0), metavar="S|LO-HI")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, help="append one JSON line per match")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n, v in VARIANTS.items():
            print(f"{n:16s} {v.get('note', '')}")
        return 0
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants {unknown}; see --list")
    OUT.mkdir(exist_ok=True)
    plans: dict[str, str | None] = {}
    for v in args.variants:
        spec = VARIANTS[v]
        if any(k in spec for k in ("top", "stages", "drop")):
            path = OUT / f"plan_{v}.yaml"
            path.write_text(yaml.safe_dump(edit_plan(spec), sort_keys=False))
            plans[v] = str(path)
        else:
            plans[v] = None
    names = args.scenario or list(sm.SCENARIOS)
    jobs = [(v, n, s, args.latency, plans[v]) for n in names for s in range(1, args.seeds + 1) for v in args.variants]
    with Pool(args.jobs) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    if args.out:
        with args.out.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    lat = args.latency if isinstance(args.latency, float) else f"{args.latency[0]:g}-{args.latency[1]:g}"
    print(f"latency {lat} s, {args.seeds} seeds, {len(names)} scenarios")
    head = "".join(f"{v:>18s}" for v in args.variants)
    print(f"{'scenario':30s}{head}")
    for n in names:
        cells = []
        for v in args.variants:
            rr = [r for r in rows if r["variant"] == v and r["scenario"] == n]
            pts = statistics.mean(r["points"] for r in rr)
            lost = sum(r["lost"] for r in rr)
            cells.append(f"{pts:8.1f} lost {lost}/{len(rr)}")
        print(f"{n:30s}" + "".join(f"{c:>18s}" for c in cells))
    print(f"{'ALL (mean points, losses)':30s}" + "".join(
        f"{statistics.mean(r['points'] for r in rows if r['variant'] == v):8.1f} lost "
        f"{sum(r['lost'] for r in rows if r['variant'] == v):>2d}    " for v in args.variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
