import socket

import pytest

import fake_station
from conftest import make_station, stage, wait_until

from harness.frames import FrameSource
from harness.plan import load_plan
from harness.station_client import StationClient, StationError


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_client_roundtrip(fake):
    client = StationClient(fake.api_url)
    assert client.status()["state"] == "idle"
    assert client.start("hello", "ckpt")["state"] in {"starting", "running"}
    assert client.wait_for_state({"running"}, 5, poll_s=0.02)["state"] == "running"
    assert client.set_prompt("again")["prompt"] == "again"
    assert client.stop()["state"] == "idle"
    assert client.home()["state"] == "idle"
    assert client.reset()["state"] in {"resetting", "homing", "idle"}


def test_client_errors(fake):
    client = StationClient(fake.api_url)
    client.start("x")
    with pytest.raises(StationError) as exc:
        client.start("x")
    assert exc.value.status_code == 409
    with pytest.raises(StationError) as exc:
        client.set_prompt("")
    assert exc.value.status_code == 400


def test_wait_for_state_times_out(fake):
    with pytest.raises(TimeoutError):
        StationClient(fake.api_url).wait_for_state({"running"}, 0.2, poll_s=0.05)


def test_dry_run_sends_no_posts(fake):
    client = StationClient(fake.api_url, dry_run=True)
    assert client.status()["state"] == "idle"          # GETs are sent
    for result in (client.start("x"), client.set_prompt("y"), client.stop(), client.home(), client.reset()):
        assert isinstance(result, dict) and "state" in result
    assert fake.posts() == []
    assert fake.calls and all(method == "GET" for method, _, _ in fake.calls)


def test_dry_run_unreachable_station_is_synthetic():
    client = StationClient(f"http://127.0.0.1:{free_port()}", dry_run=True, timeout_s=1)
    assert client.start("x")["state"] == "running"


def test_frames_latest(fake):
    frames = FrameSource(fake.view_url)
    assert frames.latest() == {} and frames.latest_n() is None
    StationClient(fake.api_url).start("go")
    assert wait_until(lambda: frames.latest())
    got = frames.latest()
    assert set(got) == {"scene", "wrist_l", "wrist_r"}
    assert all(v[:2] == b"\xff\xd8" for v in got.values())
    assert frames.latest_n() >= 1


def test_frames_never_raise():
    frames = FrameSource(f"http://127.0.0.1:{free_port()}", timeout_s=0.5)
    assert frames.latest() == {} and frames.latest_n() is None


# ---------------------------------------------------------------- load_plan
def test_load_plan_ok(write_plan):
    plan = load_plan(write_plan([stage("a", recovery=["open grippers"]), stage("b", on_fail="skip")]))
    assert [s.name for s in plan.stages] == ["a", "b"]
    assert plan.stages[0].recovery == ["open grippers"]
    assert plan.stages[1].on_fail == "skip"


@pytest.mark.parametrize("override,field", [
    ({"prompt": ""}, "prompt"),
    ({"prompt": "   "}, "prompt"),
    ({"prompt": "two\nlines"}, "prompt"),
    ({"prompt": "x" * 513}, "prompt"),
    ({"timeout_s": 0}, "timeout_s"),
    ({"timeout_s": -3}, "timeout_s"),
    ({"max_attempts": 0}, "max_attempts"),
    ({"min_run_s": -1}, "min_run_s"),
    ({"on_fail": "explode"}, "on_fail"),
    ({"verifier": "oracle"}, "verifier"),
])
def test_load_plan_rejects(write_plan, override, field):
    path = write_plan([stage("good"), stage("bad_stage", **override)])
    with pytest.raises(ValueError) as exc:
        load_plan(path)
    assert "bad_stage" in str(exc.value) and field in str(exc.value)


def test_load_plan_bad_top_level(write_plan):
    with pytest.raises(ValueError):
        load_plan(write_plan([stage("a")], time_budget_s=0))
    with pytest.raises(ValueError):
        load_plan(write_plan([]))
