"""Replace one camera's video with a still "parked" view in chosen episodes of a LeRobot v2.1 dataset.

Why: in leader/follower recording (follower_code/metal/leader_collect.py) the left arm is the leader, moved by hand,
so its wrist camera swings around and films the operator. In a match that arm stays parked at home and its camera
sees a still view. A fine-tune trains on every camera, so leader episodes would teach it a view it never gets.
This writes a COPY of the dataset where that camera shows the parked view in every frame of the chosen episodes.
Everything else (data, other cameras, other episodes) is copied unchanged; that camera's stats are recomputed.

    # 1. look before you write: per episode, how much the view wanders and how far it is from the parked view
    python tools/freeze_camera.py SRC --camera left_wrist --ref REF --list
    # 2. write the copy (episodes are always named: a leader-only dataset is simply `all`)
    python tools/freeze_camera.py SRC OUT --camera left_wrist --episodes all --ref REF

--ref is the parked view: a frame of an episode recorded IN THE SAME TABLE LAYOUT with the left arm parked at home
(a Quest episode), as DATASET:EPISODE:FRAME, or an image file. The view depends on the layout: the 13:20 and 16:34
sessions on 7 Nov had different parked views, so take the reference from the session you are freezing.

Choose episodes yourself. The scores are a guide, not a test: a parked camera that watches the right arm working
also "moves" (unplug-v1 episode 3 scores 25 and is a genuine parked view). Record leader episodes into their own
folder and freeze `all` of them. A still frame drops what a parked camera would see change (the right arm, cables
leaving the strip); that is the price of hiding the operator. Needs pyarrow, numpy and ffmpeg/ffprobe (run it on the
laptop: the station has no ffmpeg).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_episodes import _ffprobe_frames, _image_stats  # noqa: E402



def _video(ds: Path, info: dict, key: str, ep: int) -> Path:
    return ds / info["video_path"].format(episode_chunk=0, video_key=key, episode_index=ep)


def _frames(path: Path, w: int, h: int, every: int = 10) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='not(mod(n\\,{every}))',scale={w}:{h}",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def motion(path: Path) -> float:
    """How far the view wanders from the episode's first frame: mean abs pixel difference (0..255) at 64x36."""
    f = _frames(path, 64, 36)
    return float(np.abs(f - f[0]).mean()) if len(f) > 1 else 0.0


def ref_distance(path: Path, ref: Path) -> float:
    """Median over the episode of the mean abs pixel difference (0..255, at 64x36) from the parked reference view."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ref), "-vf", "scale=64:36", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    r = np.frombuffer(raw, np.uint8).reshape(36, 64, 3).astype(np.float32)
    f = _frames(path, 64, 36)
    return float(np.median(np.abs(f - r).mean(axis=(1, 2, 3))))


def _camera_key(info: dict, camera: str) -> str:
    keys = [k for k, f in info["features"].items() if f.get("dtype") == "video" and camera in k]
    if len(keys) != 1:
        raise SystemExit(f"--camera {camera!r} matches {keys or 'no video key'}; choose one of "
                         f"{[k for k, f in info['features'].items() if f.get('dtype') == 'video']}")
    return keys[0]


def _ref_image(ref: str, camera: str, dst: Path, w: int, h: int) -> Path:
    path = Path(ref).expanduser()
    if path.is_file():
        src, select = path, None
    else:
        ds, ep, frame = ref.rsplit(":", 2)
        ds_path = Path(ds).expanduser()
        info = json.loads((ds_path / "meta" / "info.json").read_text())
        src, select = _video(ds_path, info, _camera_key(info, camera), int(ep)), int(frame)
    vf = (f"select='eq(n\\,{select})'," if select is not None else "") + f"scale={w}:{h}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-fps_mode", "passthrough",
                    "-frames:v", "1", str(dst)], check=True)
    if not dst.is_file():
        raise SystemExit(f"--ref {ref!r}: no frame came out (episode or frame number out of range?)")
    return dst


def _still_video(image: Path, dst: Path, n: int, fps: int) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(fps), "-i", str(image),
                    "-frames:v", str(n), "-r", str(fps), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-g", "30", str(dst)], check=True)
    got = _ffprobe_frames(dst)
    if got != n:
        raise RuntimeError(f"{dst}: {got} frames, expected {n}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("source")
    p.add_argument("out", nargs="?")
    p.add_argument("--camera", default="left_wrist", help="camera role (substring of the video key)")
    p.add_argument("--list", action="store_true", help="print each episode's scores and stop")
    p.add_argument("--episodes", help="comma list of episode numbers, or 'all'")
    p.add_argument("--ref", help="the parked view: DATASET:EPISODE:FRAME, or an image file")
    args = p.parse_args(argv)

    src = Path(args.source).expanduser()
    info = json.loads((src / "meta" / "info.json").read_text())
    key = _camera_key(info, args.camera)
    eps = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    h, w = info["features"][key]["shape"][:2]
    tmp = tempfile.TemporaryDirectory()
    ref = _ref_image(args.ref, args.camera, Path(tmp.name) / "ref.png", w, h) if args.ref else None
    scores = {}
    print(f"{key}: motion = how far the view wanders from the episode's first frame; "
          f"to_ref = how far it is from the parked view (0..255)")
    for e in eps:
        i = e["episode_index"]
        vid = _video(src, info, key, i)
        scores[i] = motion(vid)
        dist = f"  to_ref {ref_distance(vid, ref):5.1f}" if ref is not None else ""
        print(f"episode {i:3d}  {e['length'] / info['fps']:6.1f} s  motion {scores[i]:5.1f}{dist}  {e['tasks'][0]}")
    if args.list:
        return 0
    if not (args.out and args.episodes and ref is not None):
        p.error("writing a copy needs OUT, --episodes and --ref (or use --list)")
    if args.episodes == "all":
        chosen = sorted(scores)
    else:
        chosen = sorted({int(x) for x in args.episodes.split(",") if x.strip()})
    missing = [i for i in chosen if i not in scores]
    if missing:
        raise SystemExit(f"no episode(s) {missing} in {src}")
    out = Path(args.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists: choose a new folder")

    fps = int(info["fps"])
    shutil.copytree(src, out)
    try:
        with tmp:
            lengths = {e["episode_index"]: e["length"] for e in eps}
            stats_path = out / "meta" / "episodes_stats.jsonl"
            stats = {s["episode_index"]: s for s in map(json.loads, stats_path.read_text().splitlines()) if s}
            for i in chosen:
                dst = _video(out, info, key, i)
                _still_video(ref, dst, lengths[i], fps)
                stats[i]["stats"][key] = _image_stats(dst, w, h, lengths[i])
                print(f"episode {i}: {key} frozen to the parked view ({lengths[i]} frames)")
            stats_path.write_text("".join(json.dumps(stats[i]) + "\n" for i in sorted(stats)))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    print(f"wrote {out}: {len(chosen)} of {len(eps)} episodes frozen; run add_eef on it, not on {src.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
