"""Harness v2 (DESIGN_v2.md): monitor verifier, debouncing, stall, recovery menu, settle, entry check, data loop.

No network anywhere: the monitor takes an injected `transport`; the one test of the real transport patches urlopen.
"""

import json
import threading
import time

import pytest

from conftest import stage, wait_until
from fake_station import FakeVerifier

from harness import report
from harness.cli import main as cli_main
from harness.frames import FrameSource
from harness.log import RunLog
from harness.plan import Recovery, load_plan, parse_plan
from harness.runner import Runner
from harness.station_client import StationClient
from harness.verifiers import (MonitorVerdict, MonitorVerifier, Verdict, anthropic_transport, make_verifier,
                               parse_monitor_reply)

JPEG = b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"


# -- helpers -------------------------------------------------------------------------------------
def mv(done=None, status="ok", conf=0.9, **kw):
    return MonitorVerdict(done, conf, kw.pop("reason", "scripted"), status=status, **kw)


class Script:
    """Verifier returning scripted verdicts in order; the last one repeats if repeat=True."""

    def __init__(self, items, repeat=True):
        self.items, self.repeat, self.calls, self.prompts = list(items), repeat, [], []

    def check(self, stage, frames, elapsed_s):
        self.calls.append((stage.name, elapsed_s, bool(frames)))
        if len(self.items) > 1 or (self.items and not self.repeat):
            item = self.items.pop(0)
        else:
            item = self.items[0] if self.items else None
        return item(stage, frames, elapsed_s) if callable(item) else (item or Verdict(None, 0.0, "script empty"))


def build(fake, plan_path, tmp_path, verifier, log=None):
    plan = load_plan(plan_path)
    verifiers = {n: verifier for n in ("manual", "claude", "time", "monitor", "claude+manual", "monitor+manual")}
    verifiers.update({s.name: verifier for s in plan.stages})
    return Runner(plan, StationClient(fake.api_url, timeout_s=10), FrameSource(fake.view_url), verifiers,
                  log or RunLog(tmp_path / "runs"))


def prompts(fake):
    return [b["prompt"] for _, r, b in fake.posts() if r in ("/start", "/prompt")]


def routes(fake):
    return [r for _, r, _ in fake.posts()]


def events(log, name=None):
    recs = [json.loads(line) for line in (log.path / "events.jsonl").read_text().splitlines()]
    return [e for e in recs if name is None or e["event"] == name]


def segments(log):
    return [json.loads(line) for line in (log.path / "segments.jsonl").read_text().splitlines()]


def reply_text(**fields):
    base = {"stage_done": False, "progress": 0.5, "status": "ok", "holding": {"left": "cup", "right": "nothing"},
            "recovery": None, "scene_ok": True, "confidence": 0.9, "reason": "r"}
    base.update(fields)
    return "Q1: is the cup held? yes.\n<json>" + json.dumps(base) + "</json>"


def api_reply(text):
    return {"content": [{"type": "text", "text": text}]}


class Transport:
    """Scripted stand-in for the Anthropic call: pops replies (text or exception), repeats the last one."""

    def __init__(self, *texts):
        self.texts, self.payloads, self.lock = list(texts), [], threading.Lock()

    def __call__(self, payload):
        with self.lock:
            self.payloads.append(payload)
            item = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(item, Exception):
            raise item
        return api_reply(item)


FRAMES = {"video.scene": JPEG, "video.wrist_l": JPEG + b"1", "video.wrist_r": JPEG + b"2"}


# -- plan ----------------------------------------------------------------------------------------
def _plan(stage_dict, **top):
    return parse_plan({"task": "t", "stages": [stage_dict], **top})


def test_v1_defaults_unchanged():
    plan = _plan({"name": "a", "prompt": "p", "recovery": ["open"]})
    s = plan.stages[0]
    assert plan.confirm == 1 and plan.claude_user == {}
    assert s.stall_s is None and s.settle_s is None and s.skip_if_done is False and s.expect is None
    assert s.effective_settle() == s.min_run_s
    assert s.recovery == ["open"]
    (r,) = s.menu()
    assert r.legacy and r.prompt == "open"


def test_v2_fields_parse():
    plan = _plan({"name": "a", "prompt": "p", "stall_s": 10, "settle_s": 2, "skip_if_done": True, "expect": " e ",
                  "recovery": [{"name": "x", "kind": "retry", "when": "w", "max_uses": 2},
                               {"prompt": "fix", "when": "w2", "settle_s": 3}, "plain"]},
                 confirm=2, verifier="monitor+manual", claude={"max_tokens": 700, "history_frames": 3})
    s = plan.stages[0]
    assert plan.confirm == 2 and plan.verifier == "monitor+manual" and s.verifier == "monitor+manual"
    assert (s.stall_s, s.settle_s, s.skip_if_done, s.expect) == (10.0, 2.0, True, "e")
    x, fix, plain = s.menu()
    assert (x.kind, x.max_uses, x.when) == ("retry", 2, "w")
    assert (fix.kind, fix.prompt, fix.settle_s, fix.name) == ("reset", "fix", 3.0, "recovery2")
    assert plain.legacy and isinstance(x, Recovery)
    assert plan.claude_user == {"max_tokens": 700, "history_frames": 3}


@pytest.mark.parametrize("bad, text", [
    ({"recovery": [{"kind": "retry", "prompt": "x"}]}, "retry"),
    ({"recovery": [{"kind": "reset"}]}, "prompt"),
    ({"recovery": [{"kind": "wobble", "prompt": "x"}]}, "kind"),
    ({"recovery": [{"prompt": "x", "bogus": 1}]}, "unknown"),
    ({"recovery": [{"name": "n", "kind": "retry"}, {"name": "n", "kind": "retry"}]}, "duplicate"),
    ({"recovery": [{"prompt": "x", "max_uses": 0}]}, "max_uses"),
    ({"recovery": [5]}, "string or a mapping"),
    ({"stall_s": 0}, "stall_s"),
    ({"settle_s": -1}, "settle_s"),
    ({"skip_if_done": "yes"}, "skip_if_done"),
    ({"expect": ""}, "expect"),
])
def test_bad_v2_stage_fields(bad, text):
    with pytest.raises(ValueError, match=text):
        _plan({"name": "a", "prompt": "p", **bad})


def test_bad_confirm():
    with pytest.raises(ValueError, match="confirm"):
        _plan({"name": "a", "prompt": "p"}, confirm=0)
    with pytest.raises(ValueError, match="confirm"):
        _plan({"name": "a", "prompt": "p"}, confirm=1.5)


def test_example_v2_plan_loads():
    from pathlib import Path
    plan = load_plan(Path(__file__).resolve().parent.parent / "plans" / "example_hill_climb_v2.yaml")
    kinds = {r.kind for s in plan.stages for r in s.menu()}
    assert plan.confirm == 2 and {"retry", "reset"} <= kinds
    assert any(s.skip_if_done for s in plan.stages) and any(s.stall_s for s in plan.stages)
    assert any(s.expect for s in plan.stages) and any(s.settle_s for s in plan.stages)


# -- parsing the monitor reply -------------------------------------------------------------------
def test_parse_valid_reply():
    v = parse_monitor_reply(reply_text(stage_done=True, progress=0.9, recovery=1), 2)
    assert isinstance(v, MonitorVerdict)
    assert (v.done, v.progress, v.status, v.recovery, v.scene_ok) == (True, 0.9, "ok", 1, True)
    assert v.holding == {"left": "cup", "right": "nothing"}


def test_parse_clamps_and_validates():
    v = parse_monitor_reply(reply_text(progress=7, confidence=-3, recovery=5, status="weird"), 2)
    assert v.progress == 1.0 and v.confidence == 0.0 and v.recovery is None and v.status is None
    assert parse_monitor_reply(reply_text(recovery=-1), 2).recovery is None
    assert parse_monitor_reply(reply_text(recovery=True), 2).recovery is None
    assert parse_monitor_reply(reply_text(recovery=0), 0).recovery is None   # empty menu


def test_parse_low_confidence_done_is_not_done():
    v = parse_monitor_reply(reply_text(stage_done=True, confidence=0.5), 0)
    assert v.done is None and "low confidence" in v.reason


@pytest.mark.parametrize("text", [
    "no markers {\"stage_done\": true}", "<json>{not json</json>", "<json>[1]</json>",
    "<json>{\"progress\": 0.5}</json>", "<json>{\"stage_done\": \"yes\"}</json>", "", "</json><json>"])
def test_parse_garbage_gives_none(text):
    v = parse_monitor_reply(text, 1)
    assert v.done is None and isinstance(v, MonitorVerdict) and v.status is None


# -- MonitorVerifier -----------------------------------------------------------------------------
def _stage_obj(**kw):
    plan = _plan({"name": "a", "prompt": "do a", "success": "is a done?", "expect": "cup lifts", "timeout_s": 40,
                  "recovery": [{"name": "r0", "kind": "retry", "when": "still on path"},
                               {"name": "r1", "prompt": "fix", "when": "dropped"}], **kw},
                 task="clear the table")
    return plan, plan.stages[0]


def test_monitor_payload_contents_and_history():
    plan, st = _stage_obj()
    t = Transport(reply_text())
    mon = MonitorVerifier(plan, transport=t, history_frames=2)
    frames = lambda n: {"video.scene": JPEG + bytes([n]), "video.wrist_l": JPEG[:2] + b"w" + bytes([n]) + JPEG[-2:]}  # noqa: E731
    for i in range(5):
        v = mon.check_now(st, frames(i), 3.0 * i)
        assert v.done is None and v.status == "ok"
    first, last = t.payloads[0], t.payloads[-1]
    assert last["model"] == "claude-haiku-5-5" and last["max_tokens"] == 4000   # 600 truncated Haiku 5.5 replies
    content = last["messages"][0]["content"]
    labels = [b["text"] for b in content if b["type"] == "text"][:-1]
    assert labels[0] == "t=+0s overhead (stage start)"
    assert labels[1:3] == ["t=+6s overhead", "t=+9s overhead"]       # history_frames=2, oldest first
    assert labels[3] == "t=+12s overhead (newest)" and labels[4] == "t=+12s video.wrist_l (newest)"
    text = content[-1]["text"]
    for needle in ("clear the table", "<-- CURRENT STAGE", "is a done?", "cup lifts", "12 s of timeout 40",
                   "Last prompt sent to the robot: do a", "[0] r0: still on path", "[1] r1: dropped",
                   "<json>", "Do not call the stage failing unless"):
        assert needle in text, needle
    assert sum(b["type"] == "image" for b in first["messages"][0]["content"]) == 2   # first call: overhead + wrist


def test_monitor_notify_prompt_change_updates_last_prompt_and_stage_change_resets_history():
    plan = _plan({"name": "a", "prompt": "do a"}, task="t")
    plan.stages.append(plan.stages[0].__class__(name="b", prompt="do b", success="b?"))
    a, b = plan.stages
    t = Transport(reply_text())
    mon = MonitorVerifier(plan, transport=t)
    mon.check_now(a, FRAMES, 0)
    mon.notify_prompt_change("fix it")
    mon.check_now(a, FRAMES, 5)
    assert "Last prompt sent to the robot: fix it" in t.payloads[1]["messages"][0]["content"][-1]["text"]
    mon.check_now(b, FRAMES, 0)
    labels = [x["text"] for x in t.payloads[2]["messages"][0]["content"] if x["type"] == "text"]
    assert not any("stage start" in lbl and "+5s" in lbl for lbl in labels)
    assert sum("overhead" in lbl for lbl in labels) == 1      # fresh history for the new stage


def test_monitor_errors_give_none_and_never_raise():
    plan, st = _stage_obj()
    for t in (Transport(RuntimeError("boom")), Transport("no json at all")):
        v = MonitorVerifier(plan, transport=t).check_now(st, FRAMES, 1)
        assert v.done is None and "monitor error" in v.reason


def test_monitor_no_frames():
    plan, st = _stage_obj()
    t = Transport(reply_text())
    mon = MonitorVerifier(plan, transport=t)
    assert mon.check(st, {}, 1).done is None and mon.check_now(st, {}, 1).done is None
    assert t.payloads == []


def test_monitor_async_never_blocks_and_one_in_flight():
    plan, st = _stage_obj()
    release, started = threading.Event(), threading.Event()
    calls = []

    def slow(payload):
        calls.append(payload)
        started.set()
        release.wait(5)
        return api_reply(reply_text(stage_done=True, progress=0.9))

    mon = MonitorVerifier(plan, transport=slow)
    t0 = time.monotonic()
    assert mon.check(st, FRAMES, 1).done is None
    assert started.wait(2)
    for _ in range(3):
        assert mon.check(st, FRAMES, 2).reason == "monitor call in flight"
    assert time.monotonic() - t0 < 1.0 and len(calls) == 1      # still one call, nobody blocked
    release.set()
    assert mon.wait_idle(3)
    v = mon.check(st, FRAMES, 3)
    assert v.done is True and v.progress == 0.9
    assert mon.check(st, FRAMES, 4).done is None                # a result is consumed exactly once
    mon.wait_idle(3)


def test_monitor_drops_result_computed_before_prompt_change():
    plan, st = _stage_obj()
    release, started = threading.Event(), threading.Event()

    def slow(payload):
        started.set()
        release.wait(5)
        return api_reply(reply_text(stage_done=True))

    mon = MonitorVerifier(plan, transport=slow)
    mon.check(st, FRAMES, 1)
    assert started.wait(2)
    mon.notify_prompt_change("new prompt")
    release.set()
    mon.wait_idle(3)
    assert mon.check(st, FRAMES, 2).done is None                # the old answer was discarded
    mon.wait_idle(3)


def test_monitor_needs_key_or_transport(monkeypatch):
    plan, _ = _stage_obj()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        MonitorVerifier(plan)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_verifier("monitor", plan)
    assert isinstance(make_verifier("monitor", plan, transport=Transport(reply_text())), MonitorVerifier)
    assert make_verifier("monitor+manual", plan, manual=object(), transport=Transport(reply_text())) is not None


def test_real_transport_sends_key_only_in_header_to_anthropic(monkeypatch):
    import urllib.request
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(api_reply(reply_text())).encode()

    def fake_urlopen(req, timeout=None):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), req.data
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    plan, st = _stage_obj()
    mon = MonitorVerifier(plan, api_key="sk-ant-SECRET123")
    assert "SECRET123" not in repr(mon) and "SECRET123" not in str(vars(mon).get("model"))
    assert mon.check_now(st, FRAMES, 1).status == "ok"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["X-api-key"] == "sk-ant-SECRET123"
    assert b"SECRET123" not in seen["body"]
    assert callable(anthropic_transport("k"))


# -- runner: debounce, frames, stall -------------------------------------------------------------
def test_confirm_two_needs_consecutive_done(fake, write_plan, tmp_path):
    path = write_plan([stage("a")], confirm=2)
    sv = Script([mv(True), mv(None, "ok"), mv(True), mv(True)], repeat=False)
    result = build(fake, path, tmp_path, sv).run()
    assert result.ok, result.reason
    assert len(sv.calls) == 3          # kaushik-fixes: a second done within DONE_WINDOW_S passes despite the flicker


def test_done_far_apart_still_needs_confirm(fake, write_plan, tmp_path, monkeypatch):
    """Two done verdicts further apart than DONE_WINDOW_S do not pass on their own."""
    import harness.runner as R
    monkeypatch.setattr(R, "DONE_WINDOW_S", 0.0)
    path = write_plan([stage("a")], confirm=2)
    sv = Script([mv(True), mv(None, "ok"), mv(True), mv(True)], repeat=False)
    assert build(fake, path, tmp_path, sv).run().ok and len(sv.calls) == 4


def test_confirm_ignores_in_flight_ticks(fake, write_plan, tmp_path):
    """Plain Verdict(None) ticks (monitor call still in flight) neither count nor reset."""
    path = write_plan([stage("a")], confirm=2)
    sv = Script([mv(True), Verdict(None, 0, "in flight"), Verdict(None, 0, "in flight"), mv(True)], repeat=False)
    assert build(fake, path, tmp_path, sv).run().ok and len(sv.calls) == 4


def test_confirm_one_is_v1(fake, write_plan, tmp_path):
    sv = Script([mv(True)], repeat=False)
    assert build(fake, write_plan([stage("a")]), tmp_path, sv).run().ok and len(sv.calls) == 1


def test_manual_y_is_immediate_even_with_confirm_2(fake, write_plan, tmp_path):
    sv = Script([Verdict(True, 1.0, "operator typed y", immediate=True)], repeat=False)
    assert build(fake, write_plan([stage("a")], confirm=3), tmp_path, sv).run().ok and len(sv.calls) == 1


def test_manual_n_is_immediate_even_with_confirm_2(fake, write_plan, tmp_path):
    path = write_plan([stage("a", on_fail="skip")], confirm=3)
    sv = Script([Verdict(False, 1.0, "operator typed n", immediate=True)], repeat=False)
    result = build(fake, path, tmp_path, sv).run()
    assert result.stages[0].ok is False and len(sv.calls) == 1


def test_done_without_frames_never_passes(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=0.6, on_fail="skip")], cameras_for_check=["no_such_camera"])
    sv = Script([mv(True)])
    result = build(fake, path, tmp_path, sv).run()
    assert sv.calls and not any(c[2] for c in sv.calls)
    assert result.stages[0].ok is False and "timeout" in result.stages[0].reason


def test_immediate_done_without_frames_is_fine(fake, write_plan, tmp_path):
    path = write_plan([stage("a")], cameras_for_check=["no_such_camera"])
    sv = Script([Verdict(True, 1.0, "operator typed y", immediate=True)], repeat=False)
    assert build(fake, path, tmp_path, sv).run().ok


def test_failing_needs_confirm_and_half_the_timeout(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=1.0, on_fail="skip")], confirm=2)
    sv = Script([mv(None, "failing", reason="stuck")])
    log = RunLog(tmp_path / "runs")
    result = build(fake, path, tmp_path, sv, log).run()
    r = result.stages[0]
    assert r.ok is False and r.reason.startswith("monitor says failing")
    first_fail_checks = [e for e in events(log, "verdict")]
    assert len(first_fail_checks) >= 3                       # early failing verdicts were not enough
    assert first_fail_checks[-1]["elapsed_s"] >= 0.5         # ended only past 50% of timeout_s
    assert all(e["elapsed_s"] < 0.5 for e in first_fail_checks[:-1])


def test_failing_streak_broken_by_ok(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=1.2, on_fail="skip")], confirm=2)
    # failing/ok alternate forever: never two in a row, so the stage can only time out
    sv = Script([mv(None, "failing"), mv(None, "ok")] * 30, repeat=False)
    r = build(fake, path, tmp_path, sv).run().stages[0]
    assert r.reason.startswith("timeout")


def test_stall_by_progress_clock(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=10, stall_s=0.4, on_fail="skip")])
    sv = Script([mv(None, "ok", progress=0.2)])
    t0 = time.monotonic()
    r = build(fake, path, tmp_path, sv).run().stages[0]
    assert r.ok is False and r.reason.startswith("stalled") and time.monotonic() - t0 < 5


def test_rising_progress_does_not_stall(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=10, stall_s=0.3)])
    items = [mv(None, "ok", progress=min(1.0, 0.1 * i)) for i in range(1, 9)] + [mv(True, "ok", progress=1.0)]
    assert build(fake, path, tmp_path, Script(items, repeat=False)).run().ok


def test_stall_by_monitor_status_needs_confirm_in_a_row(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=10, stall_s=100, on_fail="skip")], confirm=2)
    sv = Script([mv(None, "stalled"), mv(None, "ok"), mv(None, "stalled"), mv(None, "stalled")], repeat=False)
    r = build(fake, path, tmp_path, sv).run().stages[0]
    assert r.reason.startswith("stalled") and len(sv.calls) == 4


def test_stalled_status_ignored_when_stall_s_off(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=0.6, on_fail="skip")])
    r = build(fake, path, tmp_path, Script([mv(None, "stalled")])).run().stages[0]
    assert r.reason.startswith("timeout")


def test_regression_logged_and_counts_as_failing(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=1.0, on_fail="skip")])
    sv = Script([mv(None, "ok", progress=0.8), mv(None, "ok", progress=0.4)])
    log = RunLog(tmp_path / "runs")
    r = build(fake, path, tmp_path, sv, log).run().stages[0]
    assert events(log, "regression") and r.reason.startswith("regressed")
    assert segments(log)[0]["max_progress"] == 0.8


def test_small_drop_is_not_regression(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=5)])
    sv = Script([mv(None, "ok", progress=0.8), mv(None, "ok", progress=0.6), mv(True, "ok", progress=0.7)], repeat=False)
    log = RunLog(tmp_path / "runs")
    assert build(fake, path, tmp_path, sv, log).run().ok and not events(log, "regression")


# -- runner: settle ------------------------------------------------------------------------------
def test_settle_blocks_verifier_after_prompt_change(fake, write_plan, tmp_path):
    path = write_plan([stage("a", settle_s=0.5, timeout_s=4)])
    fv = FakeVerifier({"a": [None, True]})
    assert build(fake, path, tmp_path, fv).run().ok
    assert fv.calls and min(c[1] for c in fv.calls) >= 0.5


def test_settle_defaults_to_min_run(fake, write_plan, tmp_path):
    path = write_plan([stage("a", min_run_s=0.3)])
    fv = FakeVerifier({"a": [True]})
    build(fake, path, tmp_path, fv).run()
    assert fv.calls[0][1] >= 0.3


def test_prompt_change_events_carry_chunks(fake, write_plan, tmp_path):
    log = RunLog(tmp_path / "runs")
    fv = FakeVerifier({"a": [True], "b": [True]})
    assert build(fake, write_plan([stage("a"), stage("b")]), tmp_path, fv, log).run().ok
    pcs = events(log, "prompt_change")
    assert [(e["prompt"], e["how"]) for e in pcs] == [("do a", "start"), ("do b", "set_prompt")]
    assert all("chunks" in e for e in pcs)


def test_progress_moved_event_after_prompt_change(fake, write_plan, tmp_path):
    log = RunLog(tmp_path / "runs")
    sv = Script([mv(None, "ok", progress=0.2), mv(None, "ok", progress=0.2), mv(None, "ok", progress=0.5),
                 mv(True, "ok", progress=0.9)], repeat=False)
    assert build(fake, write_plan([stage("a", timeout_s=5)]), tmp_path, sv, log).run().ok
    (moved,) = events(log, "progress_moved")
    assert moved["before"] == 0.2 and moved["after"] == 0.5 and "since_change_s" in moved


# -- runner: recovery menu -----------------------------------------------------------------------
class SceneVerifier:
    """Answers from the fake's live prompt. `rules` maps a prompt -> verdict factory; default is `base`."""

    def __init__(self, fake, base, rules=None):
        self.fake, self.base, self.rules = fake, base, rules or {}

    def check(self, stage, frames, elapsed_s):
        fn = self.rules.get(self.fake.prompt)
        if fn is None:
            return self.base(self)
        return fn(self)

    def used(self, prompt):
        return any(r == "/prompt" and b["prompt"] == prompt for _, r, b in self.fake.posts())


RECS = [{"name": "again", "kind": "retry", "when": "on path"},
        {"name": "fix_b", "kind": "reset", "prompt": "fix b", "when": "b broke", "settle_s": 0.1},
        {"name": "fix_c", "kind": "reset", "prompt": "fix c", "when": "c broke", "settle_s": 0.1}]


def test_monitor_recovery_hint_is_used(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=0.8, max_attempts=3, recovery=RECS)])
    sv = SceneVerifier(
        fake,
        base=lambda s: mv(True) if s.used("fix c") else mv(None, "failing", recovery=2),
        rules={"fix c": lambda s: mv(None, "ok", scene_ok=True)})
    log = RunLog(tmp_path / "runs")
    result = build(fake, path, tmp_path, sv, log).run()
    assert result.ok, result.reason
    assert prompts(fake) == ["do a", "fix c", "do a"]          # hint 2 beat list order
    assert [s["kind"] for s in segments(log)] == ["stage", "recovery", "stage"]
    assert segments(log)[1]["name"] == "fix_c" and segments(log)[1]["outcome"] == "ok"


def test_recovery_order_retry_reset_and_scene_not_ok_burns_attempt(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=0.8, max_attempts=4, recovery=RECS)])
    sv = SceneVerifier(
        fake,
        base=lambda s: mv(True) if s.used("fix c") else Verdict(False, 1.0, "scripted no"),
        rules={"fix b": lambda s: mv(None, "ok", scene_ok=False),
               "fix c": lambda s: mv(None, "ok", scene_ok=True)})
    log = RunLog(tmp_path / "runs")
    result = build(fake, path, tmp_path, sv, log).run()
    assert result.ok, result.reason
    # again (retry: stage prompt re-issued), then fix_b (scene never ok), then fix_c, then the stage prompt
    assert prompts(fake) == ["do a", "do a", "fix b", "fix c", "do a"]
    assert result.stages[0].attempts == 4                      # the broken scene counted as an attempt
    outcomes = [(s["kind"], s.get("name"), s["outcome"]) for s in segments(log)]
    assert ("recovery", "fix_b", "scene_not_ok") in outcomes and ("recovery", "fix_c", "ok") in outcomes


def test_reset_scene_not_ok_to_the_end_fails_stage(fake, write_plan, tmp_path):
    recs = [{"name": "fix_b", "kind": "reset", "prompt": "fix b", "settle_s": 0.1}]
    path = write_plan([stage("a", timeout_s=0.6, max_attempts=2, recovery=recs)])
    sv = SceneVerifier(fake, base=lambda s: Verdict(False, 1.0, "no"),
                       rules={"fix b": lambda s: mv(None, "ok", scene_ok=False)})
    result = build(fake, path, tmp_path, sv).run()
    assert result.ok is False and "scene still not ok" in result.reason
    assert "do a" == prompts(fake)[0] and prompts(fake).count("do a") == 1   # stage prompt not re-issued


def test_menu_exhausted_falls_back_to_on_fail(fake, write_plan, tmp_path):
    recs = [{"name": "again", "kind": "retry", "max_uses": 1}]
    path = write_plan([stage("a", max_attempts=3, recovery=recs, on_fail="home_and_retry")])
    fv = FakeVerifier({"a": [False, False, True, True]})
    result = build(fake, path, tmp_path, fv).run()
    assert result.ok and result.stages[0].attempts == 3
    # attempt 1 fails -> retry item (in place), attempt 2 fails -> menu empty -> home_and_retry
    assert routes(fake) == ["/start", "/prompt", "/stop", "/home", "/start", "/stop"]


def test_max_uses_two_but_never_same_item_twice_in_a_row(fake, write_plan, tmp_path):
    recs = [{"name": "again", "kind": "retry", "max_uses": 2}]
    path = write_plan([stage("a", max_attempts=4, recovery=recs, on_fail="home_and_retry")])
    fv = FakeVerifier({"a": [False, False, False, True, True]})
    log = RunLog(tmp_path / "runs")
    assert build(fake, path, tmp_path, fv, log).run().ok
    used = [e["name"] for e in events(log, "recovery")]
    assert used == ["again"]                                    # second use blocked right after the first
    assert "/home" in routes(fake)


def test_legacy_string_recovery_still_in_order(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=0.8, max_attempts=3, recovery=["r1", "r2"])])
    fv = FakeVerifier({"a": [False, False, True]})
    assert build(fake, path, tmp_path, fv).run().ok
    assert prompts(fake) == ["do a", "r1", "do a", "r2", "do a"]


def test_legacy_string_recovery_ignored_under_home_and_retry(fake, write_plan, tmp_path):
    path = write_plan([stage("a", recovery=["r1"], on_fail="home_and_retry")])
    fv = FakeVerifier({"a": [False, True, True]})
    assert build(fake, path, tmp_path, fv).run().ok
    assert "r1" not in prompts(fake) and "/home" in routes(fake)


# -- runner: entry check -------------------------------------------------------------------------
def test_skip_if_done(fake, write_plan, tmp_path):
    path = write_plan([stage("a"), stage("b", skip_if_done=True), stage("c")])
    fv = FakeVerifier({"a": [True], "b": [Verdict(True, 0.9, "already there")], "c": [True]})
    log = RunLog(tmp_path / "runs")
    result = build(fake, path, tmp_path, fv, log).run()
    assert result.ok and [(s.name, s.attempts) for s in result.stages] == [("a", 1), ("b", 0), ("c", 1)]
    assert prompts(fake) == ["do a", "do c"]
    assert events(log, "entry_check")[0]["skipped"] is True


def test_skip_if_done_needs_confidence(fake, write_plan, tmp_path):
    path = write_plan([stage("a"), stage("b", skip_if_done=True)])
    fv = FakeVerifier({"a": [True], "b": [Verdict(True, 0.75, "maybe"), True]})
    result = build(fake, path, tmp_path, fv).run()
    assert result.ok and result.stages[1].attempts == 1 and "do b" in prompts(fake)


def test_skip_if_done_manual_y_counts(fake, write_plan, tmp_path):
    path = write_plan([stage("a"), stage("b", skip_if_done=True)])
    fv = FakeVerifier({"a": [True], "b": [Verdict(True, 1.0, "operator typed y", immediate=True)]})
    result = build(fake, path, tmp_path, fv).run()
    assert result.stages[1].attempts == 0 and "do b" not in prompts(fake)


def test_skip_if_done_uses_check_now_when_available(fake, write_plan, tmp_path):
    path = write_plan([stage("a"), stage("b", skip_if_done=True)])
    plan = load_plan(path)
    t = Transport(reply_text(stage_done=True, confidence=0.95, progress=1.0))
    mon = MonitorVerifier(plan, transport=t)
    ok = FakeVerifier({"a": [True]})
    runner = Runner(plan, StationClient(fake.api_url, timeout_s=10), FrameSource(fake.view_url),
                    {"manual": ok, "a": ok, "b": mon}, RunLog(tmp_path / "runs"))
    plan.stages[1].verifier = "b"
    result = runner.run()
    assert result.stages[1].attempts == 0 and len(t.payloads) == 1   # exactly one synchronous call


# -- end to end with the scripted monitor --------------------------------------------------------
def test_end_to_end_monitor_with_scripted_transport(fake, write_plan, tmp_path):
    path = write_plan([stage("a", timeout_s=5, expect="done looks like X")], verifier="monitor", confirm=2)
    plan = load_plan(path)
    t = Transport(reply_text(progress=0.3), reply_text(stage_done=True, progress=0.9, confidence=0.9))
    mon = MonitorVerifier(plan, transport=t)
    runner = Runner(plan, StationClient(fake.api_url, timeout_s=10), FrameSource(fake.view_url), {"monitor": mon},
                    RunLog(tmp_path / "runs"))
    result = runner.run()
    assert result.ok, result.reason
    assert len(t.payloads) >= 3                                 # one not-done, then two done in a row
    assert "done looks like X" in t.payloads[0]["messages"][0]["content"][-1]["text"]


# -- data loop -----------------------------------------------------------------------------------
def test_segments_and_manifest(fake, write_plan, tmp_path):
    path = write_plan([stage("a", recovery=["fix"], timeout_s=1.0)])
    fv = FakeVerifier({"a": [False, True]})
    log = RunLog(tmp_path / "runs")
    t_before = time.time()
    assert build(fake, path, tmp_path, fv, log).run().ok
    segs = segments(log)
    assert [(s["kind"], s["outcome"]) for s in segs] == [("stage", "failed"), ("recovery", "ok"), ("stage", "ok")]
    for s in segs:
        assert {"stage", "attempt", "kind", "prompt", "t_start", "t_end", "outcome", "reason", "max_progress"} <= set(s)
        assert t_before <= s["t_start"] <= s["t_end"]
    assert segs[1]["prompt"] == "fix" and segs[0]["prompt"] == "do a"
    man = json.loads((log.path / "manifest.json").read_text())
    assert man["station_status"]["state"] == "idle" and man["start_time"] and man["end_time"]


def test_segment_written_when_run_aborts(write_plan, tmp_path):
    from conftest import make_station
    fake = make_station(scenario="arm_dies", arm_dies_after_s=3.0)
    try:
        log = RunLog(tmp_path / "runs")
        result = build(fake, write_plan([stage("a", timeout_s=20)]), tmp_path, FakeVerifier({}), log).run()
        assert result.ok is False
        assert segments(log)[-1]["outcome"] == "aborted"
    finally:
        fake.close()


def _fake_run(root, name, ok, stages, segs, evs=()):
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"ok": ok, "stages": stages}))
    (d / "segments.jsonl").write_text("\n".join(json.dumps(s) for s in segs))
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in evs))
    return d


def test_report_tables_and_false_success(tmp_path, capsys):
    root = tmp_path / "runs"
    seg = lambda stg, att, kind, out, t0, t1, reason="", name=None: {   # noqa: E731
        "stage": stg, "attempt": att, "kind": kind, "outcome": out, "t_start": t0, "t_end": t1, "reason": reason,
        **({"name": name} if name else {})}
    _fake_run(root, "20260101T000000Z", True,
              [{"name": "a", "ok": True, "attempts": 2, "seconds": 20.0, "reason": ""},
               {"name": "b", "ok": True, "attempts": 1, "seconds": 10.0, "reason": ""}],
              [seg("a", 1, "stage", "failed", 0, 10, "timeout after 45 s"), seg("a", 1, "recovery", "ok", 10, 14, name="fix"),
               seg("a", 2, "stage", "ok", 14, 20), seg("b", 1, "stage", "ok", 20, 30)])
    _fake_run(root, "20260102T000000Z", False,
              [{"name": "a", "ok": True, "attempts": 1, "seconds": 8.0, "reason": ""},
               {"name": "b", "ok": False, "attempts": 1, "seconds": 5.0, "reason": "x"}],
              [seg("a", 1, "stage", "ok", 100, 108), seg("b", 1, "stage", "failed", 100.5 + 7.5, 113,
                                                         "monitor says failing: dropped")],
              [{"t": 110.0, "event": "verdict", "stage": "b", "scene_ok": False}])
    rep = report.build_report(report.run_dirs(root))
    assert rep["runs"] == 2 and rep["runs_ok"] == 1
    assert rep["stages"]["a"]["ok"] == 2 and rep["stages"]["a"]["attempts"] == 3
    assert rep["recoveries"]["a/fix"] == {"used": 1, "ok": 1}
    assert rep["false_success_leads"] == 1
    assert {r for r, _ in rep["failure_reasons"]} == {"timeout after N s", "monitor says failing"}
    assert report.main([str(root), "--last", "1"]) == 0
    out = capsys.readouterr().out
    assert "runs: 1" in out and "false-success leads" in out
    assert report.main([str(root)]) == 0
    assert "a/fix" in capsys.readouterr().out


def test_report_empty_dir(tmp_path, capsys):
    assert report.main([str(tmp_path)]) == 1


def test_report_on_real_run(fake, write_plan, tmp_path, capsys):
    path = write_plan([stage("a", recovery=["fix"], timeout_s=1.0)])
    build(fake, path, tmp_path, FakeVerifier({"a": [False, True]})).run()
    assert report.main([str(tmp_path / "runs")]) == 0
    assert "a/recovery1" in capsys.readouterr().out


# -- CLI -----------------------------------------------------------------------------------------
def test_cli_monitor_without_key_exits_2(write_plan, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = write_plan([stage("a")])
    rc = cli_main([str(path), "--mock", "--verifier", "monitor", "--runs-dir", str(tmp_path / "r")])
    assert rc == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err
    rc = cli_main([str(path), "--mock", "--verifier", "monitor+manual", "--runs-dir", str(tmp_path / "r")])
    assert rc == 2


def test_cli_writes_manifest_and_never_the_key(write_plan, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TOPSECRET")
    path = write_plan([stage("a", timeout_s=1.0)], verifier="time")
    runs = tmp_path / "r"
    rc = cli_main([str(path), "--mock", "--runs-dir", str(runs)])
    assert rc == 0
    (run,) = runs.iterdir()
    man = json.loads((run / "manifest.json").read_text())
    import hashlib
    assert man["plan_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert man["plan_file"] == str(path.resolve()) and man["harness_version"] and man["host"]
    assert man["args"]["mock"] is True and man["station_status"] and man["end_time"]
    assert (run / "segments.jsonl").exists()
    assert "TOPSECRET" not in capsys.readouterr().out
    assert all("TOPSECRET" not in p.read_text() for p in run.iterdir() if p.suffix in (".json", ".jsonl"))
