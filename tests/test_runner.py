import threading

import pytest

from conftest import make_station, stage, wait_until
from fake_station import FakeVerifier

from harness.frames import FrameSource
from harness.log import RunLog
from harness.plan import load_plan
from harness.runner import Runner
from harness.station_client import StationClient
from harness.verifiers import Verdict


def build(fake, plan_path, tmp_path, verifier):
    plan = load_plan(plan_path)
    verifiers = {"manual": verifier, "claude": verifier, "time": verifier}
    verifiers.update({s.name: verifier for s in plan.stages})
    client = StationClient(fake.api_url, timeout_s=10)
    return Runner(plan, client, FrameSource(fake.view_url), verifiers, RunLog(tmp_path / "runs"))


def routes(fake):
    return [r for _, r, _ in fake.posts()]


def prompts(fake):
    return [b["prompt"] for _, r, b in fake.posts() if r in ("/start", "/prompt")]


class Station:
    """Verifier that answers from the fake's live prompt: False once, then True when the station is on
    the stage's own prompt again after having seen a recovery prompt (or any other prompt)."""

    def __init__(self, fake, fail_first=True, need_other_prompt=False):
        self.fake, self.failed, self.need_other = fake, not fail_first, need_other_prompt
        self.seen_other = False
        self.calls = []

    def check(self, stage, frames, elapsed_s):
        self.calls.append(stage.name)
        if not self.failed:
            self.failed = True
            return Verdict(False, 1.0, "scripted fail")
        self.seen_other = self.seen_other or any(
            r == "/prompt" and b["prompt"] != stage.prompt for _, r, b in self.fake.posts())
        if self.fake.prompt != stage.prompt:
            return Verdict(None, 0.0, "recovering")
        if self.need_other and not self.seen_other:
            return Verdict(None, 0.0, "waiting for recovery")
        return Verdict(True, 1.0, "scripted ok")


def test_happy_three_stages(fake, write_plan, tmp_path):
    names = ["a", "b", "c"]
    path = write_plan([stage(n) for n in names])
    fv = FakeVerifier({n: [None, True] for n in names})
    result = build(fake, path, tmp_path, fv).run()
    assert result.ok, result.reason
    assert [s.name for s in result.stages] == names and all(s.ok for s in result.stages)
    assert prompts(fake) == ["do a", "do b", "do c"]
    assert routes(fake) == ["/start", "/prompt", "/prompt", "/stop"]   # one stop, at the end only
    assert fake.state == "idle"
    assert (tmp_path / "runs").exists()


def test_retry_in_place_with_recovery(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=4, recovery=["open the grippers"], on_fail="retry_in_place")])
    fv = Station(fake, need_other_prompt=True)
    result = build(fake, path, tmp_path, fv).run()
    assert result.ok, result.reason
    assert result.stages[0].attempts == 2
    assert prompts(fake) == ["do a", "open the grippers", "do a"]
    assert routes(fake) == ["/start", "/prompt", "/prompt", "/stop"]


def test_retry_in_place_without_recovery_resends_prompt(fake, write_plan, tmp_path):
    path = write_plan([stage("a", on_fail="retry_in_place")])
    result = build(fake, path, tmp_path, FakeVerifier({"a": [False, True, True, True]})).run()
    assert result.ok and result.stages[0].attempts == 2
    assert prompts(fake)[0] == "do a" and prompts(fake)[-1] == "do a"
    assert "/home" not in routes(fake)


def test_home_and_retry(fake, write_plan, tmp_path):
    path = write_plan([stage("a", on_fail="home_and_retry")])
    result = build(fake, path, tmp_path, FakeVerifier({"a": [False, True, True, True]})).run()
    assert result.ok, result.reason
    assert result.stages[0].attempts == 2
    assert routes(fake) == ["/start", "/stop", "/home", "/start", "/stop"]
    assert prompts(fake) == ["do a", "do a"]


def test_skip(fake, write_plan, tmp_path):
    path = write_plan([stage("a", on_fail="skip", max_attempts=1), stage("b")])
    result = build(fake, path, tmp_path, FakeVerifier({"a": [False], "b": [True, True]})).run()
    assert [s.ok for s in result.stages] == [False, True]
    assert prompts(fake) == ["do a", "do b"]
    assert routes(fake)[-1] == "/stop"


def test_skip_does_not_retry(fake, write_plan, tmp_path):
    path = write_plan([stage("a", on_fail="skip", max_attempts=3), stage("b")])
    result = build(fake, path, tmp_path, FakeVerifier({"a": [False, False, False], "b": [True, True]})).run()
    assert result.stages[0].ok is False and result.stages[0].attempts == 1
    assert result.stages[1].ok is True


def test_abort(fake, write_plan, tmp_path):
    path = write_plan([stage("a", on_fail="abort"), stage("b")])
    result = build(fake, path, tmp_path, FakeVerifier({"a": [False]})).run()
    assert result.ok is False
    assert "do b" not in prompts(fake)
    assert routes(fake)[-1] == "/stop" and fake.state == "idle"


def test_out_of_attempts_fails_run(fake, write_plan, tmp_path):
    path = write_plan([stage("a", max_attempts=1, on_fail="retry_in_place"), stage("b")])
    result = build(fake, path, tmp_path, FakeVerifier({"a": [False]})).run()
    assert result.ok is False and "do b" not in prompts(fake)


def test_timeout_fails_attempt(write_plan, tmp_path):
    fake = make_station()
    try:
        path = write_plan([stage("a", timeout_s=0.4, max_attempts=1)])
        result = build(fake, path, tmp_path, FakeVerifier({})).run()   # verifier never says anything
        assert result.ok is False
        assert routes(fake)[-1] == "/stop"
    finally:
        fake.close()


def test_arm_dies_aborts(write_plan, tmp_path):
    fake = make_station(scenario="arm_dies", arm_dies_after_s=0.3)
    try:
        path = write_plan([stage("a", timeout_s=20), stage("b")])
        result = build(fake, path, tmp_path, FakeVerifier({})).run()
        assert result.ok is False
        assert "/stop" in routes(fake)
        assert "do b" not in prompts(fake)
    finally:
        fake.close()


def test_start_error_reports_last_error(write_plan, tmp_path):
    fake = make_station(scenario="start_error")
    try:
        path = write_plan([stage("a")])
        result = build(fake, path, tmp_path, FakeVerifier({"a": [True]})).run()
        assert result.ok is False
        assert "checkpoint not found" in result.reason
        assert "/stop" in routes(fake)
    finally:
        fake.close()


def test_slow_start_is_waited_out(write_plan, tmp_path):
    fake = make_station(scenario="slow_start", slow_start_s=0.6)
    try:
        path = write_plan([stage("a")])
        result = build(fake, path, tmp_path, FakeVerifier({"a": [True, True]})).run()
        assert result.ok, result.reason
    finally:
        fake.close()


def test_preflight_refuses_running_station(fake, write_plan, tmp_path):
    StationClient(fake.api_url).start("someone else")
    assert wait_until(lambda: fake.state == "running")
    before = len(fake.posts())
    result = build(fake, write_plan([stage("a")]), tmp_path, FakeVerifier({"a": [True]})).run()
    assert result.ok is False
    assert len(fake.posts()) == before and fake.prompt == "someone else"


def test_preflight_refuses_stale_camera(write_plan, tmp_path):
    fake = make_station(scenario="camera_stale")
    try:
        result = build(fake, write_plan([stage("a")]), tmp_path, FakeVerifier({"a": [True]})).run()
        assert result.ok is False and fake.posts() == []
    finally:
        fake.close()


def test_time_budget(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=60)], time_budget_s=0.8)
    result = build(fake, path, tmp_path, FakeVerifier({})).run()
    assert result.ok is False
    assert result.seconds < 10
    assert routes(fake)[-1] == "/stop" and fake.state == "idle"


def test_request_stop(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=60), stage("b")])
    runner = build(fake, path, tmp_path, FakeVerifier({}))
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", runner.run()))
    thread.start()
    assert wait_until(lambda: fake.state == "running" and fake.chunks > 2)
    runner.request_stop()
    thread.join(10)
    assert not thread.is_alive()
    assert out["r"].ok is False
    assert fake.state == "idle" and routes(fake)[-1] == "/stop"
    assert "do b" not in prompts(fake)


def test_verifier_gets_frames(fake, write_plan, tmp_path):
    path = write_plan([stage("a")])
    fv = FakeVerifier({"a": [None, None, True]})
    build(fake, path, tmp_path, fv).run()
    assert fv.calls and fv.calls[-1][2] == ["scene", "wrist_l", "wrist_r"]
    assert all(c[1] >= 0.05 for c in fv.calls)   # never asked before min_run_s


class FaultOnce:
    """Puts the fake station into error with `message` at the first check of stage b, then verifies everything."""

    def __init__(self, fake, message):
        self.fake, self.message, self.faulted = fake, message, False

    def check(self, stage, frames, elapsed_s):
        if stage.name == "b" and not self.faulted:
            self.faulted = True
            with self.fake._lock:
                self.fake.last_error = self.message
                self.fake.state = "error"   # the run died mid-stage
            return Verdict(None, 0.0, "running")
        return Verdict(True, 1.0, "scripted ok")


def test_inference_fault_restarts_the_run_with_the_same_prompt(fake, write_plan, tmp_path):
    # Cluster sim matches lost minutes to one DA-6000 gateway timeout; that must not end a match.
    path = write_plan([stage(n, min_run_s=0.05) for n in ("a", "b", "c")], confirm=1)
    msg = "DreamAPI refused the inference request: DA-6000 GATEWAY_FAULT (connection_timed_out)"
    runner = build(fake, path, tmp_path, FaultOnce(fake, msg))
    result = runner.run()
    assert result.ok, result.reason
    events = (runner.log.path / "events.jsonl").read_text()
    assert '"station_restart"' in events and "DA-6000" in events
    starts = [b["prompt"] for _, r, b in fake.posts() if r == "/start"]
    assert starts == ["do a", "do b"]


def test_hardware_fault_still_ends_the_run(fake, write_plan, tmp_path):
    path = write_plan([stage(n, min_run_s=0.05) for n in ("a", "b", "c")], confirm=1)
    runner = build(fake, path, tmp_path, FaultOnce(fake, "arm 'right' motor 4 over-current"))
    result = runner.run()
    assert not result.ok and "over-current" in result.reason
    assert '"station_restart"' not in (runner.log.path / "events.jsonl").read_text()
