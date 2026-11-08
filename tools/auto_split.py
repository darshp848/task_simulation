"""Propose a split spec for long multi-task takes, from the right gripper's grasps and where they happen.

    python tools/auto_split.py DATASET --out SPLIT_DATASET --spec spec.json [--hub X,Y] [--sheets DIR]
    python tools/split_episodes.py spec.json            # after reviewing the spec and the sheets

Per episode it finds every grasp of the right gripper (commanded closed, past half) and classifies it by the tool
point's position (Metal forward kinematics, Dream's TCP) relative to the hub:
- unplug: closes within --near of the hub, lifts at least --lift, releases more than --far from it
  -> T1 (approach + pull, ending once the plug is --lift above where it was grasped) and T2 (carry + release,
  ending --tail s after the gripper opens). A grasp still closed at the end of the episode gives T1 only.
- plug-in: closes more than --far from the hub and releases within --near of it -> T3 (approach + carry + insert,
  ending --tail s after it opens). Whether the plug went in is not known from the joints: CHECK THE SHEET.
- pick-and-place: with --picks, any other grasp that is released (on the left arm: every grasp, since the hub
  rules are in the right arm's frame) -> a segment from the approach to --tail s after the release, with the task
  "LABEL ME". Look at the sheet and set its task text (or drop it) in the spec: split_episodes.py refuses LABEL ME.
- anything else (an episode that starts closed, a grasp that never lifts, ...) is left out and reported.
The working arm is chosen per episode (--arm auto): the one whose joints move more. In leader/follower takes the
leader's side is recorded parked, so the follower is picked.
Each segment starts where the previous one ended (so the episodes chain like the harness's prompt switches), or
at most --max-approach s before the grasp, with leading stillness trimmed.

--hub defaults to the median of the grasp positions (most grasps are at the hub). With --sheets it writes one
review image per episode: for every segment, the right wrist camera at its first and last frame and the overhead
camera at its last frame. Look at them before splitting: a human hand in view, a plug that did not come out or
did not seat, means remove that segment from the spec.

Needs pyarrow, numpy (and ffmpeg for --sheets). Reads only; writes the spec and the sheets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.force import tool_point  # noqa: E402

T1 = "unplug the USB cable at the end of the hub nearest the right arm"
T2 = "put the held cable down on the table away from the hub"
T3 = "plug the loose USB cable on the table into a free port on the hub"
TCP = (0.135, 0.0, 0.0)            # Dream's tool point for the stock Metal gripper (the kit's station config)
RIGHT_JOINTS, RIGHT_GRIP = slice(7, 13), 13
ARMS = {"right": (slice(7, 13), 13), "left": (slice(0, 6), 6)}   # the Metal kit's layout: left arm first
LABEL = "LABEL ME"
MIN_FRAMES = 49


def grasps(cmd: np.ndarray, fps: float, min_s: float = 0.5, gap_s: float = 0.4) -> list[tuple[int, int | None]]:
    """(close, release) frame pairs where the commanded gripper is past half closed; release None = still closed at
    the end; a grasp already closed at frame 0 is returned with close 0. Short re-opens (< gap_s) are merged."""
    closed = cmd < 50
    out: list[list[int | None]] = []
    i, n = 0, len(cmd)
    while i < n:
        if closed[i]:
            j = i
            while j < n and closed[j]:
                j += 1
            if out and out[-1][1] is not None and i - out[-1][1] < gap_s * fps:
                out[-1][1] = j if j < n else None
            else:
                out.append([i, j if j < n else None])
            i = j
        else:
            i += 1
    return [(a, b) for a, b in out if (b if b is not None else n) - a >= min_s * fps]


def working_arm(state: np.ndarray, action: np.ndarray | None = None, fps: float = 30.0) -> str:
    """The arm that grasps (its gripper closes and opens again); if both or neither do, the one whose joints move
    more. Quest takes can move both arms while only one works."""
    rng = {side: float(np.ptp(state[:, j], axis=0).max()) for side, (j, _) in ARMS.items()}
    if action is not None:
        gripping = [side for side, (_, g) in ARMS.items() if any(c > 0 for c, _ in grasps(action[:, g], fps))]
        if len(gripping) == 1:
            return gripping[0]
    return max(rng, key=rng.get)


def plan_episode(state: np.ndarray, action: np.ndarray, fps: float, hub: np.ndarray, *, near: float, far: float,
                 lift: float, tail_s: float, max_approach_s: float, arm: str = "right",
                 picks: bool = False) -> tuple[list[dict], list[str]]:
    joints, grip = ARMS[arm]
    p = np.array([tool_point(list(q), TCP) for q in state[:, joints]])
    speed = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) * fps]
    n = len(p)
    segs, notes = [], []
    prev_end = 0

    def start_from(close: int) -> int:
        s = max(prev_end, close - int(max_approach_s * fps))
        moving = np.flatnonzero(speed[s:close] > 0.01)
        if len(moving) and s != prev_end:
            s = max(s, s + int(moving[0]) - int(0.5 * fps))
        elif len(moving) and s == prev_end and moving[0] > fps:   # still at the start of the episode: trim
            s = s + int(moving[0]) - int(0.5 * fps)
        return s

    for close, release in grasps(action[:, grip], fps):
        t = f"{close / fps:.1f}-{(release or n) / fps:.1f}s"
        if close == 0:
            notes.append(f"{t}: starts closed (no grasp seen), left out")
            prev_end = (release or n) - 1
            continue
        d_close = float(np.linalg.norm(p[close, :2] - hub))
        d_rel = float(np.linalg.norm(p[release, :2] - hub)) if release is not None else None
        end = min(n, (release + int(tail_s * fps)) if release is not None else n)
        if arm == "left":
            d_close = float("inf")      # the hub rules are in the right arm's frame: every left grasp is a pick
        if d_close <= near:
            up = np.flatnonzero(p[close:(release or n), 2] >= p[close, 2] + lift)
            if not len(up):
                notes.append(f"{t}: grasp at the hub never lifted {lift * 100:.0f} cm, left out")
                prev_end = end - 1
                continue
            pull_end = min(close + int(up[0]) + int(0.2 * fps), (release or n))
            s = start_from(close)
            segs.append({"start": s, "end": pull_end, "task": T1, "kind": "unplug"})
            if release is None:
                notes.append(f"{t}: unplug still holding at the end: T1 only")
            elif d_rel > far:
                segs.append({"start": pull_end, "end": end, "task": T2, "kind": "put down"})
            else:
                notes.append(f"{t}: unplug released {d_rel * 100:.0f} cm from the hub (not 'away'): T1 only")
                end = pull_end
            prev_end = end
        elif d_close > far and d_rel is not None and d_rel <= near:
            s = start_from(close)
            segs.append({"start": s, "end": end, "task": T3, "kind": "plug-in (check it seated)"})
            prev_end = end
        elif picks and release is not None:
            s = start_from(close)
            segs.append({"start": s, "end": end, "task": LABEL, "kind": "pick-place (label it)"})
            prev_end = end
        else:
            where = "never released" if d_rel is None or d_rel == float("inf") else f"released {d_rel * 100:.0f} cm from the hub"
            notes.append(f"{t}: grasp {d_close * 100:.0f} cm from the hub, {where}: unclear, left out")
            prev_end = end - 1 if release is not None else n
    kept = []
    for sg in segs:
        if sg["end"] - sg["start"] < MIN_FRAMES:
            notes.append(f"{sg['start'] / fps:.1f}-{sg['end'] / fps:.1f}s {sg['kind']}: shorter than 1.6 s, left out")
        else:
            kept.append(sg)
    return kept, notes


def _sheet(ds: Path, info: dict, ep: int, segs: list[dict], fps: float, dst: Path, arm: str = "right") -> None:
    vp = info["video_path"]
    wrist = ds / vp.format(episode_chunk=0, video_key=f"observation.images.{arm}_wrist", episode_index=ep)
    scene_key = next(k for k, f in info["features"].items() if f.get("dtype") == "video" and "scene" in k)
    scene = ds / vp.format(episode_chunk=0, video_key=scene_key, episode_index=ep)
    rows = []
    for k, sg in enumerate(segs):
        picks = [(wrist, sg["start"]), (wrist, sg["end"] - 1), (scene, sg["end"] - 1)]
        sel = []
        for j, (v, f) in enumerate(picks):
            out = dst.with_name(f"{dst.stem}_{k}_{j}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), "-vf", f"select='eq(n\\,{f})',scale=320:-1",
                            "-fps_mode", "passthrough", "-frames:v", "1", str(out)], check=True)
            sel.append(out)
        row = dst.with_name(f"{dst.stem}_{k}.png")
        # rows are in segment order, as printed; no text overlay (drawtext needs fonts not every ffmpeg has)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *sum((["-i", str(x)] for x in sel), []), "-filter_complex",
                        "[0][1][2]hstack=inputs=3,pad=iw:ih+6:0:0:white", str(row)], check=True)
        rows.append(row)
        for x in sel:
            x.unlink()
    if not rows:
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", *sum((["-i", str(r)] for r in rows), []),
                    "-filter_complex", f"{''.join(f'[{i}]' for i in range(len(rows)))}vstack=inputs={len(rows)}"
                    if len(rows) > 1 else "null", str(dst)], check=True)
    for r in rows:
        r.unlink()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True, help="the split dataset split_episodes.py will write (a new folder)")
    ap.add_argument("--spec", required=True, help="where to write the spec JSON")
    ap.add_argument("--episodes", help="comma list (default: all)")
    ap.add_argument("--hub", help="hub position X,Y in the right arm's base frame, metres (default: estimated)")
    ap.add_argument("--near", type=float, default=0.06, help="'at the hub' radius, m")
    ap.add_argument("--far", type=float, default=0.08, help="'away from the hub' radius, m")
    ap.add_argument("--lift", type=float, default=0.06, help="pull height that counts as unplugged, m")
    ap.add_argument("--tail", type=float, default=0.7, help="seconds kept after the gripper opens")
    ap.add_argument("--max-approach", type=float, default=10.0, help="longest approach kept before a grasp, s")
    ap.add_argument("--sheets", help="folder for one review image per episode")
    ap.add_argument("--arm", choices=("auto", "left", "right"), default="auto", help="the arm that works")
    ap.add_argument("--picks", action="store_true", help="also cut other grasp-and-release cycles, task LABEL ME")
    args = ap.parse_args(argv)

    ds = Path(args.dataset).expanduser()
    info = json.loads((ds / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    eps = [json.loads(x) for x in (ds / "meta" / "episodes.jsonl").read_text().splitlines() if x.strip()]
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        eps = [e for e in eps if e["episode_index"] in wanted]
    data = {}
    for e in eps:
        t = pq.read_table(ds / info["data_path"].format(episode_chunk=0, episode_index=e["episode_index"]))
        data[e["episode_index"]] = (np.array(t["observation.state"].to_pylist()), np.array(t["action"].to_pylist()))
    if args.hub:
        hub = np.array([float(v) for v in args.hub.split(",")])
    else:
        pts = []
        for s, a in data.values():
            if working_arm(s, a, fps) != "right":
                continue
            for c, _ in grasps(a[:, RIGHT_GRIP], fps):
                if c:
                    pts.append(tool_point(list(s[c, RIGHT_JOINTS]), TCP)[:2])
        if not pts:
            hub = np.array([0.38, 0.13])   # the 7 Nov station layout; only used for right-arm grasps
        hub = np.median(np.array(pts), axis=0)
    print(f"hub at x {hub[0]:+.3f} y {hub[1]:+.3f} m (right arm base frame){'' if args.hub else ', estimated'}")

    segments = []
    sheets = Path(args.sheets).expanduser() if args.sheets else None
    if sheets:
        sheets.mkdir(parents=True, exist_ok=True)
    for e in eps:
        i = e["episode_index"]
        arm = working_arm(*data[i], fps) if args.arm == "auto" else args.arm
        segs, notes = plan_episode(*data[i], fps, hub, near=args.near, far=args.far, lift=args.lift,
                                   tail_s=args.tail, max_approach_s=args.max_approach, arm=arm, picks=args.picks)
        print(f"\nepisode {i} ({e['length'] / fps:.1f} s, {arm} arm, {e['tasks'][0]!r})")
        for k, sg in enumerate(segs):
            print(f"  [{k}] {sg['start'] / fps:6.1f}-{sg['end'] / fps:6.1f} s  {sg['kind']:<26} {sg['task']!r}")
            segments.append({"episode": i, "start_s": round(sg["start"] / fps, 3), "end_s": round(sg["end"] / fps, 3),
                             "task": sg["task"], "kind": sg["kind"], "arm": arm})
        for note in notes:
            print(f"  - {note}")
        if sheets and segs:
            _sheet(ds, info, i, segs, fps, sheets / f"episode_{i:03d}.png", arm)
    spec = {"source": str(ds), "out": str(Path(args.out).expanduser()), "hub_xy": [round(float(v), 3) for v in hub],
            "segments": segments}
    Path(args.spec).write_text(json.dumps(spec, indent=2) + "\n")
    counts = {k: sum(s["task"] == k for s in segments) for k in (T1, T2, T3, LABEL)}
    print(f"\n{len(segments)} segments -> {args.spec}: T1 {counts[T1]}, T2 {counts[T2]}, T3 {counts[T3]}, "
          f"to label {counts[LABEL]}"
          + (f"; review images in {sheets}" if sheets else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
