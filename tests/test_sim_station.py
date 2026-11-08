"""sim_station.py in scripted mode (no key, no network). Skipped without mujoco / Dream's yam_mujoco example / GL."""

import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if Path("/dev/dri").exists() else "osmesa"

pytest.importorskip("mujoco")
pytest.importorskip("PIL")
np = pytest.importorskip("numpy")

import sim_station  # noqa: E402
from conftest import wait_until  # noqa: E402

EXAMPLE_FALLBACKS = ["/tmp/claude-0/dl/x_dreamlabs-1.0.37-py3-none-any.whl/dream_examples/yam_mujoco"]


def _example_dir():
    try:
        sim_station.load_example(None)
        return None
    except SystemExit:
        pass
    for path in EXAMPLE_FALLBACKS:
        if (Path(path) / "yam_env.py").is_file():
            return path
    pytest.skip("Dream's yam_mujoco example is not installed (pip install dreamlabs) and no --example-dir")


@pytest.fixture(scope="module")
def sim():
    example = _example_dir()
    try:
        station = sim_station.SimStation(policy="scripted", api_port=0, view_port=0, realtime=False,
                                         example_dir=example).start()
    except Exception as exc:  # no usable GL, etc.
        pytest.skip(f"simulator could not start here: {exc}")
    yield station
    station.close()


def call(station, method, route, body=None, base=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request((base or station.api_url) + route, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=70) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def settle(sim):
    """Back to idle whatever a previous test left."""
    call(sim, "POST", "/stop", {})
    assert wait_until(lambda: sim.state in ("idle", "error"), 30)


def test_scripted_policy_chunk_shape_and_limits():
    anchor = {"left": np.r_[np.zeros(6), 1.0], "right": np.r_[np.zeros(6), 1.0]}
    policy = sim_station.ScriptedPolicy(anchor, latency_s=0)
    first, second = policy.get_action(), policy.get_action()
    for reply in (first, second):
        for key in ("q_l_chunk", "q_r_chunk"):
            chunk = np.array(reply[key])
            assert chunk.shape == (48, 7) and np.isfinite(chunk).all()
            assert chunk[:, 6].min() >= 0 and chunk[:, 6].max() <= 1
    again = sim_station.ScriptedPolicy(anchor, latency_s=0).get_action()
    assert again == first  # deterministic
    assert abs(np.array(second["q_l_chunk"])[0] - np.array(first["q_l_chunk"])[-1]).max() < 0.05  # chunks join up


def test_status_keys_idle(sim):
    settle(sim)
    code, status = call(sim, "GET", "/status")
    assert code == 200 and status["state"] == "idle"
    for key in ("prompt", "checkpoint", "run", "chunks", "stalls", "clamps", "arms", "cameras", "last_error",
                "run_elapsed_s"):
        assert key in status
    assert status["arms"] == {"left": True, "right": True}
    assert set(status["cameras"]) == {"overhead", "wrist_l", "wrist_r"}
    assert all(age is not None for age in status["cameras"].values())


def test_bad_requests(sim):
    settle(sim)
    assert call(sim, "POST", "/prompt", {"prompt": "  "})[0] == 400
    assert call(sim, "POST", "/prompt", {})[0] == 400
    assert call(sim, "GET", "/nope")[0] == 404
    assert call(sim, "POST", "/nope", {})[0] == 404


def test_run_prompt_frames_stop_home_reset(sim):
    settle(sim)
    code, status = call(sim, "POST", "/start", {"prompt": "put the apple in the bowl"})
    assert code == 200 and status["state"] == "starting"
    assert call(sim, "POST", "/start", {"prompt": "again"})[0] == 409
    assert call(sim, "POST", "/home", {})[0] == 409
    assert call(sim, "POST", "/reset", {})[0] == 409
    assert wait_until(lambda: sim.state == "running", 60)
    assert wait_until(lambda: sim.chunks >= 1, 60)

    # trips: 9 real JPEG frames per camera, the observation carries the prompt
    trips = call(sim, "GET", "/trips", base=sim.view_url)[1]
    assert trips and set(trips[0]) >= {"n", "sent_s", "error", "rtt_ms"}
    trip = call(sim, "GET", f"/trip/{trips[-1]['n']}", base=sim.view_url)[1]
    assert set(trip["frames"]) == {"video.overhead", "video.wrist_l", "video.wrist_r"}
    from PIL import Image

    for frames in trip["frames"].values():
        assert len(frames) == 9
        img = Image.open(io.BytesIO(base64.b64decode(frames[-1])))
        assert img.format == "JPEG" and img.size == (320, 176)
        assert np.asarray(img).std() > 5  # a rendered scene, not a flat image
    assert trip["observation"]["prompt"] == "put the apple in the bowl"
    assert call(sim, "GET", "/trip/999999", base=sim.view_url)[0] == 404

    # a new prompt reaches later observations
    assert call(sim, "POST", "/prompt", {"prompt": "put the banana in the bowl"})[0] == 200
    seen = trips[-1]["n"]
    assert wait_until(lambda: any(t["n"] > seen for t in call(sim, "GET", "/trips", base=sim.view_url)[1]), 60)
    last = call(sim, "GET", "/trips", base=sim.view_url)[1][-1]["n"]
    newest = call(sim, "GET", f"/trip/{last}", base=sim.view_url)[1]
    assert newest["observation"]["prompt"] == "put the banana in the bowl"

    # stop holds the pose
    code, status = call(sim, "POST", "/stop", {})
    assert code == 200 and status["state"] == "idle"
    before = call(sim, "GET", "/status")[1]["chunks"]
    import time

    time.sleep(0.5)
    assert call(sim, "GET", "/status")[1]["chunks"] == before

    # home moves both arms to the reset pose; reset restores the scene
    code, status = call(sim, "POST", "/home", {})
    assert code == 200 and status["state"] == "idle"
    assert call(sim, "POST", "/reset", {})[0] == 200
    assert call(sim, "GET", "/status")[1]["state"] == "idle"


def test_state_after_home_is_home_pose(sim):
    settle(sim)
    assert call(sim, "POST", "/home", {})[0] == 200
    # the sim thread owns the env; the hold targets are what is commanded while idle
    for side in ("left", "right"):
        assert np.allclose(sim.hold[side], sim.yam_env.HOME)


def test_run_start_error_is_reported(sim, monkeypatch):
    settle(sim)

    def boom(*a, **k):
        raise RuntimeError("no model today")

    monkeypatch.setattr(sim, "_open", boom)
    assert call(sim, "POST", "/start", {"prompt": "go"})[0] == 200
    assert wait_until(lambda: sim.state == "error", 60)
    error = call(sim, "GET", "/status")[1]["last_error"]
    assert error["message"] == "no model today" and error["type"] == "RuntimeError" and error["error_code"]
    monkeypatch.undo()
    assert call(sim, "POST", "/start", {"prompt": "go"})[0] == 200  # error state allows a new start
    settle(sim)


def test_live_view_serves_page_state_and_cameras(sim):
    with urllib.request.urlopen(sim.view_url + "/live", timeout=10) as resp:
        assert resp.status == 200 and b"Sim station" in resp.read()
    with urllib.request.urlopen(sim.view_url + "/live/state", timeout=10) as resp:
        state = json.load(resp)
    assert state["status"]["sim"] is True and set(state["cameras"]) == {"top", "left", "right"}
    assert "bodies" in state["objects"] and "tcp" in state["objects"]
    with urllib.request.urlopen(sim.view_url + "/live/cam/top.jpg", timeout=10) as resp:
        assert resp.headers["Content-Type"] == "image/jpeg" and resp.read()[:2] == b"\xff\xd8"
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(sim.view_url + "/live/cam/nope.jpg", timeout=10)
