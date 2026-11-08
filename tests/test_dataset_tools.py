"""tools/split_episodes.py and tools/freeze_camera.py on a tiny synthetic LeRobot v2.1 dataset."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    pytest.skip("needs ffmpeg and ffprobe", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import freeze_camera  # noqa: E402
import split_episodes  # noqa: E402

FPS, W, H = 30, 64, 36
CAMS = ("scene", "left_wrist", "right_wrist")


def _make_dataset(root: Path, lengths=(90, 120)) -> Path:
    info = {"codebase_version": "v2.1", "robot_type": "metal", "fps": FPS, "chunks_size": 1000,
            "total_episodes": len(lengths), "total_frames": sum(lengths), "total_tasks": 1,
            "total_videos": 3 * len(lengths), "total_chunks": 1, "splits": {"train": f"0:{len(lengths)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"action": {"dtype": "float32", "shape": [14]},
                         "observation.state": {"dtype": "float32", "shape": [14]},
                         **{f"observation.images.{c}": {"dtype": "video", "shape": [H, W, 3],
                                                        "names": ["height", "width", "channels"]} for c in CAMS}}}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "do it (to be split)"}) + "\n")
    eps, stats, index = [], [], 0
    for ep, n in enumerate(lengths):
        state = np.arange(n * 14, dtype=np.float32).reshape(n, 14) + 1000 * ep
        table = pa.table({"observation.state": pa.array(state.tolist(), pa.list_(pa.float32(), 14)),
                          "action": pa.array((state + 0.5).tolist(), pa.list_(pa.float32(), 14)),
                          "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
                          "frame_index": pa.array(np.arange(n), pa.int64()),
                          "episode_index": pa.array(np.full(n, ep), pa.int64()),
                          "index": pa.array(index + np.arange(n), pa.int64()),
                          "task_index": pa.array(np.zeros(n, np.int64))})
        path = root / f"data/chunk-000/episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for c in CAMS:   # a moving test pattern: every frame differs
            v = root / f"videos/chunk-000/observation.images.{c}/episode_{ep:06d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}",
                            "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)], check=True)
        eps.append({"episode_index": ep, "tasks": ["do it (to be split)"], "length": n})
        stats.append({"episode_index": ep, "stats": {}})
        index += n
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (root / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    return root


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_split_cuts_rows_videos_and_meta(tmp_path):
    src = _make_dataset(tmp_path / "src", lengths=(90, 150))
    out = tmp_path / "out"
    spec = {"source": str(src), "out": str(out), "segments": [
        {"episode": 1, "start_s": 0.5, "end_s": 2.5, "task": "a"},
        {"episode": 1, "start_s": 2.5, "end_s": 4.5, "task": "b"},
        {"episode": 0, "start_s": 0.0, "end_s": 2.0, "task": "a"}]}
    split_episodes.split(spec)
    eps = _jsonl(out / "meta/episodes.jsonl")
    assert [(e["tasks"][0], e["length"]) for e in eps] == [("a", 60), ("b", 60), ("a", 60)]
    assert [t["task"] for t in _jsonl(out / "meta/tasks.jsonl")] == ["a", "b"]
    info = json.loads((out / "meta/info.json").read_text())
    assert info["total_episodes"] == 3 and info["total_frames"] == 180 and info["splits"] == {"train": "0:3"}
    src_state = np.array(pq.read_table(src / "data/chunk-000/episode_000001.parquet")["observation.state"].to_pylist())
    t = pq.read_table(out / "data/chunk-000/episode_000001.parquet")
    assert np.array_equal(np.array(t["observation.state"].to_pylist()), src_state[75:135])
    assert t["frame_index"].to_pylist()[0] == 0 and t["index"].to_pylist()[0] == 60
    assert t["episode_index"].to_pylist() == [1] * 60 and t["task_index"].to_pylist() == [1] * 60
    assert t.schema == pq.read_table(src / "data/chunk-000/episode_000001.parquet").schema
    for e in eps:
        for c in CAMS:
            v = out / f"videos/chunk-000/observation.images.{c}/episode_{e['episode_index']:06d}.mp4"
            assert split_episodes._ffprobe_frames(v) == e["length"]
    stats = _jsonl(out / "meta/episodes_stats.jsonl")
    assert stats[1]["stats"]["frame_index"]["max"] == [59.0]
    assert len(stats[1]["stats"]["observation.images.scene"]["mean"]) == 3


def test_split_refuses_short_segments_and_existing_out(tmp_path):
    src = _make_dataset(tmp_path / "src", lengths=(90,))
    with pytest.raises(SystemExit, match="at least"):
        split_episodes.split({"source": str(src), "out": str(tmp_path / "o"),
                              "segments": [{"episode": 0, "start_s": 0, "end_s": 1.0, "task": "a"}]})
    with pytest.raises(SystemExit, match="exists"):
        split_episodes.split({"source": str(src), "out": str(src), "segments": []})


def test_freeze_replaces_only_the_chosen_camera_and_episodes(tmp_path, capsys):
    src = _make_dataset(tmp_path / "src")
    ref = tmp_path / "ref.png"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=red:size={W}x{H}", "-frames:v", "1",
                    str(ref)], check=True)
    out = tmp_path / "out"
    assert freeze_camera.main([str(src), str(out), "--camera", "left_wrist", "--episodes", "1", "--ref", str(ref)]) == 0
    key = "observation.images.left_wrist"
    frozen = out / f"videos/chunk-000/{key}/episode_000001.mp4"
    assert split_episodes._ffprobe_frames(frozen) == 120
    assert freeze_camera.motion(frozen) < 1.0                                  # a still picture now
    assert freeze_camera.motion(out / f"videos/chunk-000/{key}/episode_000000.mp4") > 2.0      # untouched
    assert (out / "videos/chunk-000/observation.images.scene/episode_000001.mp4").read_bytes() == \
        (src / "videos/chunk-000/observation.images.scene/episode_000001.mp4").read_bytes()
    stats = {s["episode_index"]: s["stats"] for s in _jsonl(out / "meta/episodes_stats.jsonl")}
    red = stats[1][key]["mean"]
    assert red[0][0][0] > 0.8 and red[1][0][0] < 0.2 and key not in stats[0]
    assert (out / "data/chunk-000/episode_000001.parquet").read_bytes() == \
        (src / "data/chunk-000/episode_000001.parquet").read_bytes()
    # a reference taken from a dataset frame works too, and --list writes nothing
    assert freeze_camera.main([str(src), "--ref", f"{src}:0:5", "--list"]) == 0
    assert "to_ref" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exists"):
        freeze_camera.main([str(src), str(out), "--episodes", "all", "--ref", str(ref)])


def test_auto_split_classifies_unplug_put_down_and_plug_in(monkeypatch):
    import auto_split
    monkeypatch.setattr(auto_split, "tool_point", lambda q, tcp: [q[0], q[1], q[2]])   # joints 0-2 = x, y, z
    hub, fps = np.array([0.38, 0.13]), 30.0
    keys = [  # (t s, x, y, z, gripper cmd 0 closed .. 100 open)
        (0, 0.2, 0.0, 0.2, 100), (1, 0.2, 0.0, 0.2, 100), (4, 0.38, 0.13, 0.05, 100), (5, 0.38, 0.13, 0.05, 0),
        (7, 0.38, 0.13, 0.25, 0), (9, 0.35, -0.15, 0.08, 0), (9.5, 0.35, -0.15, 0.08, 100),        # unplug, put down
        (12, 0.30, -0.05, 0.03, 100), (12.5, 0.30, -0.05, 0.03, 0), (16, 0.38, 0.13, 0.06, 0),
        (17, 0.38, 0.13, 0.06, 100), (19, 0.38, 0.0, 0.2, 100),                                       # plug-in
        (21, 0.38, 0.13, 0.05, 100), (22, 0.38, 0.13, 0.05, 0), (23, 0.38, 0.13, 0.06, 0),
        (24, 0.38, 0.13, 0.06, 100), (26, 0.3, 0.0, 0.2, 100)]                                        # never lifts
    t = np.arange(0, 26 * fps) / fps
    cols = [np.interp(t, [k[0] for k in keys], [k[i] for k in keys]) for i in range(1, 5)]
    cols[3] = np.array([next(k[4] for k in reversed(keys) if k[0] <= x) for x in t])               # step, not ramp
    state = np.zeros((len(t), 14))
    state[:, 7:10] = np.stack(cols[:3], axis=1)
    action = state.copy()
    action[:, 13] = cols[3]
    segs, notes = auto_split.plan_episode(state, action, fps, hub, near=0.06, far=0.08, lift=0.06,
                                          tail_s=0.7, max_approach_s=10.0)
    assert [s["kind"] for s in segs] == ["unplug", "put down", "plug-in (check it seated)"]
    unplug, down, plug = segs
    assert 0.4 * fps <= unplug["start"] <= 0.6 * fps                 # stillness trimmed to 0.5 s before moving
    assert 5.4 * fps <= unplug["end"] <= 6.0 * fps                   # ends once lifted 6 cm, not at the release
    assert down["start"] == unplug["end"] and down["end"] == int(9.5 * fps + 0.7 * fps)
    assert plug["start"] == down["end"] and plug["task"] == auto_split.T3
    assert any("never lifted" in n for n in notes)
    assert [(a, b) for a, b in auto_split.grasps(action[:, 13], fps)][0] == (int(5 * fps), int(9.5 * fps))
