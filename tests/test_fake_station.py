import json
import urllib.error
import urllib.request

import pytest

import fake_station
from conftest import make_station, wait_until


def call(station, method, route, body=None, base=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request((base or station.api_url) + route, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def running(station):
    assert call(station, "POST", "/start", {"prompt": "go"})[0] == 200
    assert wait_until(lambda: station.state == "running")


def test_status_keys(fake):
    code, status = call(fake, "GET", "/status")
    assert code == 200 and status["state"] == "idle"
    for key in ("prompt", "checkpoint", "run", "chunks", "stalls", "clamps", "arms", "cameras", "last_error",
                "run_elapsed_s"):
        assert key in status
    assert status["arms"] == {"left": True, "right": True}
    assert set(status["cameras"]) == {"scene", "wrist_l", "wrist_r"}


def test_start_goes_through_starting_to_running(fake):
    code, status = call(fake, "POST", "/start", {"prompt": "pick it", "checkpoint": "dream-1-ft:x"})
    assert code == 200 and status["state"] == "starting" and status["run"] == 1
    assert wait_until(lambda: fake.state == "running")
    _, status = call(fake, "GET", "/status")
    assert status["prompt"] == "pick it" and status["checkpoint"] == "dream-1-ft:x"
    assert wait_until(lambda: call(fake, "GET", "/status")[1]["chunks"] > 2)


def test_start_without_prompt_is_refused(fake):
    code, body = call(fake, "POST", "/start", {})
    assert code == 409 and "prompt" in body["error"]


def test_start_twice_409(fake):
    running(fake)
    code, body = call(fake, "POST", "/start", {"prompt": "again"})
    assert code == 409 and "running" in body["error"]


def test_stop_and_home(fake):
    running(fake)
    code, status = call(fake, "POST", "/home", {})
    assert code == 409
    code, status = call(fake, "POST", "/stop", {})
    assert code == 200 and status["state"] == "idle"
    assert call(fake, "POST", "/stop", {})[0] == 200  # nothing to stop is a no-op
    code, status = call(fake, "POST", "/home", {})
    assert code == 200 and status["state"] == "idle"


def test_stop_refused_during_home():
    station = make_station(home_s=0.6)
    try:
        import threading
        t = threading.Thread(target=call, args=(station, "POST", "/home", {}))
        t.start()
        assert wait_until(lambda: station.state == "homing")
        assert call(station, "POST", "/stop", {})[0] == 409
        assert call(station, "POST", "/start", {"prompt": "x"})[0] == 409
        t.join()
        assert station.state == "idle"
    finally:
        station.close()


def test_prompt_route(fake):
    assert call(fake, "POST", "/prompt", {})[0] == 400
    assert call(fake, "POST", "/prompt", {"prompt": "  "})[0] == 400
    assert call(fake, "POST", "/prompt", {"prompt": 5})[0] == 400
    code, status = call(fake, "POST", "/prompt", {"prompt": "new task"})
    assert code == 200 and status["prompt"] == "new task"


def test_bad_routes_and_bodies(fake):
    assert call(fake, "GET", "/nope")[0] == 404
    assert call(fake, "POST", "/nope", {})[0] == 404
    req = urllib.request.Request(fake.api_url + "/start", data=b"{not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400


def test_reset_reopens_dead_arm():
    station = make_station(scenario="arm_dies", arm_dies_after_s=0.1)
    try:
        running(station)
        assert wait_until(lambda: station.state == "error")
        _, status = call(station, "GET", "/status")
        assert status["arms"]["right"] is False and status["last_error"]
        assert call(station, "POST", "/start", {"prompt": "x"})[0] == 409
        assert call(station, "POST", "/home", {})[0] == 409
        code, status = call(station, "POST", "/reset", {})
        assert code == 200 and status["state"] == "resetting"
        assert wait_until(lambda: station.state == "idle")
        assert all(station._status()["arms"].values())
    finally:
        station.close()


def test_start_error_scenario():
    station = make_station(scenario="start_error")
    try:
        call(station, "POST", "/start", {"prompt": "x"})
        assert wait_until(lambda: station.state == "error")
        assert call(station, "GET", "/status")[1]["last_error"]
    finally:
        station.close()


def test_slow_start_stays_starting():
    station = make_station(scenario="slow_start", slow_start_s=0.4)
    try:
        call(station, "POST", "/start", {"prompt": "x"})
        assert call(station, "GET", "/status")[1]["state"] == "starting"
        assert wait_until(lambda: station.state == "running", timeout=3)
    finally:
        station.close()


def test_camera_stale_scenario():
    station = make_station(scenario="camera_stale")
    try:
        ages = call(station, "GET", "/status")[1]["cameras"]
        assert any(age is None or age > 2 for age in ages.values())
    finally:
        station.close()


def test_trips_and_frames_carry_prompt(fake):
    assert call(fake, "GET", "/trips", base=fake.view_url) == (200, [])
    running(fake)
    assert wait_until(lambda: len(call(fake, "GET", "/trips", base=fake.view_url)[1]) > 1)
    call(fake, "POST", "/prompt", {"prompt": "second"})
    wait_until(lambda: call(fake, "GET", "/trips", base=fake.view_url)[1][-1]["n"] > 0)
    trips = call(fake, "GET", "/trips", base=fake.view_url)[1]
    assert [t["n"] for t in trips] == sorted(t["n"] for t in trips)
    assert wait_until(lambda: call(fake, "GET", f"/trip/{call(fake, 'GET', '/trips', base=fake.view_url)[1][-1]['n']}",
                                   base=fake.view_url)[1]["observation"]["prompt"] == "second")
    n = trips[-1]["n"]
    code, trip = call(fake, "GET", f"/trip/{n}", base=fake.view_url)
    assert code == 200 and set(trip["frames"]) == set(fake_station.ROLES)
    import base64
    jpeg = base64.b64decode(trip["frames"]["video.scene"][0])
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"
    assert call(fake, "GET", "/trip/99999", base=fake.view_url)[0] == 404


def test_calls_recorded(fake):
    call(fake, "GET", "/status")
    call(fake, "POST", "/prompt", {"prompt": "p"})
    assert fake.calls[0] == ("GET", "/status", None)
    assert fake.calls[1] == ("POST", "/prompt", {"prompt": "p"})


def test_fake_verifier():
    from harness.verifiers import Verdict
    fv = fake_station.FakeVerifier({"a": [None, True, Verdict(False, 1.0, "x")]})
    class S:  # noqa
        name = "a"
    assert fv.check(S, {}, 0).done is None
    assert fv.check(S, {"scene": b""}, 1).done is True
    assert fv.check(S, {}, 2).done is False
    assert fv.check(S, {}, 3).done is None  # exhausted
    assert fv.checks_for("a") == 4
