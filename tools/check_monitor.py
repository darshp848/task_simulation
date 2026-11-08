"""Live check of the Claude stage monitor: real MonitorVerifier, real prompt, real model, real table photo.

It asks plans/hill_climb_v4.yaml's own stage questions (and its illegal-ball guard) about a photo of the station
table whose answers are known, and checks every reply parses, answers correctly, and how long it took:

    python tools/check_monitor.py --backend subscription   # through the logged-in `claude` CLI (no API key)
    python tools/check_monitor.py --backend api            # ANTHROPIC_API_KEY
    python tools/check_monitor.py -n 3                     # each case 3 times (consistency)

Exit code 0 when every case is right, 1 otherwise. It costs a handful of Haiku calls.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.plan import load_plan  # noqa: E402
from harness.verifiers import MonitorVerifier, default_transport  # noqa: E402

PHOTO = ROOT / "tests" / "data" / "table_c_topdown.jpg"

# The photo: the table before a match. Both grippers open and empty at the near edge, nothing held; the hub at the
# top with its cables; loose blocks (no tower); four golf balls on the table (green, white, yellow, orange), none in
# the bin. (frames, stage, expected stage_done, expected guard or None when the stage is not guarded)
CASES = [
    ("photo", "release_cable", True, None),   # both grippers open and empty, no cable held
    ("photo", "stack_2", False, None),        # no block on another block
    ("photo", "tower_pass", False, None),     # no tower at all
    ("photo", "ball_1", False, False),        # no ball in the bin; nothing held, so the white-ball guard is no
]
# --sim: the MuJoCo table from the three policy cameras, empty grippers vs a black 8-ball held in the right one.
SIM_CASES = [
    ("sim_empty", "ball_1", False, False),
    ("sim_eight_held", "ball_1", False, True),  # the safety answer that matters: an illegal ball in a gripper
]


def sim_frames(hold_eight: bool) -> dict[str, bytes]:
    """The MuJoCo relay table (sim_scenes/hill_climb_table.py) rendered from the three policy cameras. With
    `hold_eight`, a black 8-ball is put between the right gripper's fingers (closed on it): the guard must say yes."""
    import io

    import mujoco
    from PIL import Image

    from sim_scenes import hill_climb_table as h
    from sim_station import load_example

    yam_env, _ = load_example(None)
    h.patch(yam_env)
    env = yam_env.YamEnv()
    try:
        m, d = env.model, env.data
        if hold_eight:
            for _ in range(30):   # lower the right arm a little so the wrist camera sees the ball in the fingers
                env.step(env.state()["left"], env.state()["right"])
            tcp = d.site("right_tcp").xpos.copy()
            adr = m.jnt_qposadr[m.body("eight_0").jntadr[0]]
            d.qpos[adr:adr + 3] = tcp
            d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            for s in (-1, 1):
                d.joint(f"right_finger_{s}").qpos[0] = h.EIGHT_R + 0.001
            mujoco.mj_forward(m, d)
        out = {}
        for name, rgb in env.images().items():
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
            out[{"top": "overhead", "left": "wrist_l", "right": "wrist_r"}[name]] = buf.getvalue()
        return out
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["api", "subscription"],
                    default=os.environ.get("HARNESS_CLAUDE_BACKEND")
                    or ("api" if os.environ.get("ANTHROPIC_API_KEY") else "subscription"))
    ap.add_argument("--plan", default=str(ROOT / "plans" / "hill_climb_v4.yaml"))
    ap.add_argument("--image", default=str(PHOTO), help="overhead JPEG to judge (default: the station photo)")
    ap.add_argument("-n", "--repeat", type=int, default=1, help="ask each case this many times")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the model's reasoning text too")
    ap.add_argument("--sim", action="store_true", help="also judge MuJoCo renders (needs mujoco + dreamlabs)")
    ap.add_argument("--save", metavar="DIR", help="write the frames each case was judged on here")
    args = ap.parse_args()

    os.environ["HARNESS_CLAUDE_BACKEND"] = args.backend
    plan = load_plan(args.plan)
    stages = {s.name: s for s in plan.stages}
    sources = {"photo": {"overhead": Path(args.image).read_bytes()}}
    cases = list(CASES)
    if args.sim:
        sources["sim_empty"], sources["sim_eight_held"] = sim_frames(False), sim_frames(True)
        cases += SIM_CASES
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for src, frames in sources.items():
            for role, data in frames.items():
                (out / f"{src}_{role}.jpg").write_bytes(data)
    raw_transport = default_transport(90.0)
    replies: list[str] = []

    def transport(payload: dict) -> dict:
        reply = raw_transport(payload)
        replies.append("".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text"))
        return reply

    print(f"monitor check: backend={args.backend} model={plan.claude.get('model')} image={Path(args.image).name}")
    failures, times = 0, []
    for src, name, want_done, want_guard in cases:
        stage = stages[name]
        for i in range(args.repeat):
            mon = MonitorVerifier(plan, transport=transport)
            t0 = time.monotonic()
            v = mon.check_now(stage, sources[src], 5.0)
            dt = time.monotonic() - t0
            times.append(dt)
            got_done = v.done is True
            errors = []
            if v.reason.startswith("monitor error"):
                errors.append("call/parse failed")
            elif got_done != want_done:
                errors.append(f"stage_done {got_done}, expected {want_done}")
            if want_guard is not None and v.guard is not want_guard:
                errors.append(f"guard {v.guard}, expected {want_guard}")
            failures += bool(errors)
            mark = "ok  " if not errors else "FAIL"
            guard = "" if want_guard is None else f" guard={v.guard}"
            print(f"  {mark} {src:14s} {name:14s} done={got_done!s:5s}{guard} conf={v.confidence:.2f} status={v.status} "
                  f"holding={v.holding} {dt:4.1f}s  {'; '.join(errors)}")
            print(f"       reason: {v.reason[:160]}")
            if args.verbose and replies:
                print("       reply: " + replies[-1].replace("\n", "\n              ")[:1500])
    print(f"{len(times) - failures}/{len(times)} right; latency median {statistics.median(times):.1f}s "
          f"max {max(times):.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
