"""Cut LeRobot v2.1 episodes into new, shorter episodes, each with its own task text.

    python tools/split_episodes.py SPEC.json

SPEC.json:
    {"source": "path/to/dataset", "out": "path/to/new-dataset",
     "segments": [{"episode": 3, "start_s": 3.0, "end_s": 17.9, "task": "unplug the USB cable ..."}, ...]}

Writes a NEW dataset at `out` (refuses to overwrite one), never touches `source`. Each segment becomes one
episode, in spec order: parquet rows sliced with frame_index / timestamp / index / episode_index / task_index
rewritten, every camera's video cut frame-exactly with ffmpeg (h264, yuv420p, the source fps) and checked for
its frame count, and meta/ (info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl) rebuilt.

Needs pyarrow, numpy and ffmpeg/ffprobe on PATH. Not part of the harness package (which is stdlib only).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MIN_FRAMES = 49          # Dream refuses episodes shorter than 1.6 s at 30 fps (one training sample)
IMAGE_STAT_EVERY = 10    # sample every Nth frame for image stats, like the kit's writer (922 frames -> 93)


def _num_stats(values: np.ndarray) -> dict:
    v = values.astype(np.float64)
    if v.ndim == 1:
        v = v[:, None]
    return {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(),
            "std": v.std(0).tolist(), "count": [int(len(v))]}


def _ffprobe_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout.strip()
    return int(out)


def _cut_video(src: Path, dst: Path, first: int, n: int, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select='between(n\\,{first}\\,{first + n - 1})',setpts=N/({fps}*TB)"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-r", str(fps), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "30", str(dst)], check=True)
    got = _ffprobe_frames(dst)
    if got != n:
        raise RuntimeError(f"{dst}: {got} frames after the cut, expected {n}")


def _image_stats(path: Path, width: int, height: int, n: int) -> dict:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='not(mod(n\\,{IMAGE_STAT_EVERY}))'",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3).astype(np.float64) / 255.0
    px = frames.reshape(-1, 3)
    wrap = lambda a: [[[float(x)]] for x in a]  # noqa: E731  (channel, 1, 1) as LeRobot stores it
    return {"min": wrap(px.min(0)), "max": wrap(px.max(0)), "mean": wrap(px.mean(0)), "std": wrap(px.std(0)),
            "count": [int(len(frames))]}


def split(spec: dict) -> Path:
    src, out = Path(spec["source"]).expanduser(), Path(spec["out"]).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists: choose a new folder")
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    videos = {k: f for k, f in info["features"].items() if f.get("dtype") == "video"}
    data_path, video_path = info["data_path"], info["video_path"]
    tasks: list[str] = list(dict.fromkeys(s["task"] for s in spec["segments"]))
    episodes, stats_lines, total = [], [], 0
    for new_ep, seg in enumerate(spec["segments"]):
        src_ep = int(seg["episode"])
        table = pq.read_table(src / data_path.format(episode_chunk=0, episode_index=src_ep))
        first, last = round(seg["start_s"] * fps), round(seg["end_s"] * fps)
        last = min(last, table.num_rows)
        n = last - first
        if first < 0 or n < MIN_FRAMES:
            raise SystemExit(f"segment {new_ep} ({seg['start_s']}-{seg['end_s']} s of episode {src_ep}): "
                             f"{n} frames, needs at least {MIN_FRAMES}")
        part = table.slice(first, n)
        cols = {name: part.column(name) for name in part.column_names}
        task_index = tasks.index(seg["task"])
        idx = np.arange(n)
        cols["frame_index"] = pa.array(idx, pa.int64())
        cols["timestamp"] = pa.array((idx / fps).astype(np.float32), pa.float32())
        cols["episode_index"] = pa.array(np.full(n, new_ep), pa.int64())
        cols["index"] = pa.array(total + idx, pa.int64())
        cols["task_index"] = pa.array(np.full(n, task_index), pa.int64())
        new = pa.table({name: cols[name] for name in table.column_names}).cast(table.schema)
        dst = out / data_path.format(episode_chunk=0, episode_index=new_ep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new, dst)

        stats = {}
        for key, feat in videos.items():
            vsrc = src / video_path.format(episode_chunk=0, video_key=key, episode_index=src_ep)
            vdst = out / video_path.format(episode_chunk=0, video_key=key, episode_index=new_ep)
            _cut_video(vsrc, vdst, first, n, fps)
            h, w = feat["shape"][0], feat["shape"][1]
            stats[key] = _image_stats(vdst, w, h, n)
        for key in ("observation.state", "action"):
            stats[key] = _num_stats(np.array(new.column(key).to_pylist()))
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            stats[key] = _num_stats(new.column(key).to_numpy())
        stats_lines.append({"episode_index": new_ep, "stats": stats})
        episodes.append({"episode_index": new_ep, "tasks": [seg["task"]], "length": n})
        total += n
        print(f"episode {new_ep}: source ep {src_ep} {seg['start_s']:.1f}-{seg['end_s']:.1f} s, {n} frames, {seg['task']!r}")

    info = dict(info)
    info.update(total_episodes=len(episodes), total_frames=total, total_tasks=len(tasks),
                total_videos=len(episodes) * len(videos), total_chunks=1, splits={"train": f"0:{len(episodes)}"})
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks)))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats_lines))
    print(f"wrote {out}: {len(episodes)} episodes, {total} frames ({total / fps:.1f} s)")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    split(json.loads(Path(sys.argv[1]).read_text()))
