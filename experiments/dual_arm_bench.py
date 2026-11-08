"""Sequential vs combined two-arm prompts for Dream-1, scored from MuJoCo ground truth (sim_station /objects).

Each trial resets the scene, then runs one condition for the same total budget:

  seq        left arm: apple -> bowl; switch the prompt to the right-arm task as soon as the apple is in
             (or at the half-way mark), the way the harness advances stages
  combined   one prompt naming both arms and both tasks
  combined_free  one prompt with both tasks and no arm assignment

Conditions are interleaved (seq, combined, combined_free, seq, ...) so model or network drift hits each equally.

    python experiments/dual_arm_bench.py --trials 4 --budget 120 --out ../runs_dual_arm

Needs sim_station.py running (--policy dream, or scripted for a dry check). Sim only: it calls /reset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.frames import FrameSource  # noqa: E402

LEFT_TASK = "with the left arm, put the apple in the bowl"
RIGHT_TASK = "with the right arm, put the banana on the plate"
PROMPTS = {
    "combined": "with the left arm put the apple in the bowl while the right arm puts the banana on the plate, "
                "both at the same time",
    "combined_free": "put the apple in the bowl and the banana on the plate",
}
GOALS = {  # object, target, xy radius (m), max height above target origin (m), arm expected to do it
    "apple_in_bowl": ("apple", "bowl", 0.06, 0.07, "left"),
    "banana_on_plate": ("banana", "plate", 0.07, 0.05, "right"),
}
MOVING_MPS = 0.02  # a TCP faster than this counts as moving


def call(base: str, method: str, route: str, body: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_state(base: str, states: tuple[str, ...], timeout: float) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = call(base, "GET", "/status")
        if status["state"] in states:
            return status
        time.sleep(0.25)
    raise TimeoutError(f"station never reached {states}")


def goal_met(snap: dict, goal: str) -> bool:
    obj, target, radius, height, _ = GOALS[goal]
    o, t = snap["bodies"][obj], snap["bodies"][target]
    return math.dist(o[:2], t[:2]) <= radius and o[2] - t[2] <= height


def goal_dist(snap: dict, goal: str) -> float:
    obj, target, *_ = GOALS[goal]
    return math.dist(snap["bodies"][obj][:2], snap["bodies"][target][:2])


def run_trial(base: str, view: FrameSource, condition: str, budget: float, out: Path, n: int) -> dict:
    if call(base, "GET", "/status")["state"] not in ("idle", "error"):
        call(base, "POST", "/stop")
        wait_state(base, ("idle", "error"), 60)
    call(base, "POST", "/reset", {})
    wait_state(base, ("idle",), 60)
    first = call(base, "GET", "/objects")
    prompt = LEFT_TASK if condition == "seq" else PROMPTS[condition]
    t0 = time.monotonic()
    call(base, "POST", "/start", {"prompt": prompt})
    samples, switched_at, done_at = [], None, {}
    first_chunk_s = None
    while (elapsed := time.monotonic() - t0) < budget:
        snap = call(base, "GET", "/objects")
        status = call(base, "GET", "/status")
        if status["state"] == "error":
            raise RuntimeError(f"station error: {status['last_error']}")
        if first_chunk_s is None and status["chunks"] > 0:
            first_chunk_s = round(elapsed, 1)
        for goal in GOALS:
            if goal not in done_at and goal_met(snap, goal):
                done_at[goal] = round(elapsed, 1)
        if condition == "seq" and switched_at is None and ("apple_in_bowl" in done_at or elapsed >= budget / 2):
            call(base, "POST", "/prompt", {"prompt": RIGHT_TASK})
            switched_at = round(elapsed, 1)
        samples.append({"s": round(elapsed, 2), **snap})
        time.sleep(0.25)
    final = call(base, "GET", "/objects")
    try:
        trip = view.latest_trip(fallback=True)
        if trip and "overhead" in trip.frames:
            (out / f"trial{n:02d}_{condition}_end_overhead.jpg").write_bytes(trip.frames["overhead"])
    except Exception as exc:  # frames are a nice-to-have
        print(f"  (no end frame: {exc})", flush=True)
    call(base, "POST", "/stop")
    wait_state(base, ("idle", "error"), 60)

    # Arm activity from TCP speed: how much each arm moved, and how often both moved at once.
    moving = {"left": 0.0, "right": 0.0, "both": 0.0}
    path = {"left": 0.0, "right": 0.0}
    for a, b in zip(samples, samples[1:]):
        dt = b["s"] - a["s"]
        if dt <= 0:
            continue
        fast = {}
        for side in ("left", "right"):
            d = math.dist(a["tcp"][side], b["tcp"][side])
            path[side] += d
            fast[side] = d / dt > MOVING_MPS
            moving[side] += dt if fast[side] else 0.0
        moving["both"] += dt if fast["left"] and fast["right"] else 0.0
    lift = {goal: round(max(s["bodies"][o][2] for s in samples) - first["bodies"][o][2], 3)
            for goal, (o, *_rest) in GOALS.items()}
    result = {
        "trial": n, "condition": condition, "prompt": prompt, "budget_s": budget, "first_chunk_s": first_chunk_s,
        "switched_at_s": switched_at, "done_at_s": done_at,
        "success": {goal: goal in done_at for goal in GOALS},
        "still_met_at_end": {goal: goal_met(final, goal) for goal in GOALS},
        "start_dist_m": {goal: round(goal_dist(first, goal), 3) for goal in GOALS},
        "min_dist_m": {goal: round(min(goal_dist(s, goal) for s in samples), 3) for goal in GOALS},
        "end_dist_m": {goal: round(goal_dist(final, goal), 3) for goal in GOALS},
        "max_lift_m": lift,
        "tcp_path_m": {k: round(v, 3) for k, v in path.items()},
        "moving_s": {k: round(v, 1) for k, v in moving.items()},
    }
    with (out / f"trial{n:02d}_{condition}_trace.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="http://127.0.0.1:8091")
    ap.add_argument("--view", default="http://127.0.0.1:8093")
    ap.add_argument("--trials", type=int, default=4, help="trials per condition")
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per trial, the same for every condition")
    ap.add_argument("--conditions", default="seq,combined,combined_free")
    ap.add_argument("--out", default="runs_dual_arm")
    args = ap.parse_args()

    if not call(args.station, "GET", "/status").get("sim"):
        raise SystemExit("refusing to run: the station does not report sim=true (this script resets the scene)")
    out = Path(args.out) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    conditions = args.conditions.split(",")
    view = FrameSource(args.view)
    results_path = out / "results.jsonl"
    print(f"logs: {out}", flush=True)
    n = 0
    for _ in range(args.trials):
        for condition in conditions:
            n += 1
            print(f"trial {n}: {condition} ...", flush=True)
            try:
                result = run_trial(args.station, view, condition, args.budget, out, n)
            except Exception as exc:
                result = {"trial": n, "condition": condition, "error": f"{type(exc).__name__}: {exc}"}
            with results_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
            print("  " + json.dumps({k: result.get(k) for k in ("success", "min_dist_m", "moving_s", "error")}),
                  flush=True)


if __name__ == "__main__":
    main()
