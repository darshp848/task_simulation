"""plan.guard: a safety question the monitor answers on guarded stages; a confirmed yes stops the run at once."""

import pytest

from test_v2 import FRAMES, Script, Transport, build, events, mv, reply_text, routes
from conftest import stage
from harness.cli import main as cli_main
from harness.plan import parse_plan
from harness.verifiers import GUARD_SCHEMA, MONITOR_SCHEMA, MonitorVerifier, parse_monitor_reply

Q = "Is a gripper holding a black ball?"


def _guarded(**guard):
    return parse_plan({"task": "t", "verifier": "monitor", "guard": {"question": Q, **guard},
                       "stages": [{"name": "stack", "prompt": "s", "success": "s?"},
                                  {"name": "ball_1", "prompt": "b", "success": "b?"}]})


def test_guard_parses_and_covers_by_prefix():
    g = _guarded(stages=["ball_"], confirm=2).guard
    assert g.question == Q and g.confirm == 2 and "CALL STOP" in g.message
    assert g.covers("ball_1") and not g.covers("stack")
    assert _guarded().guard.covers("stack")                     # no stages = every stage


@pytest.mark.parametrize("guard, verifier, match", [
    ({"question": ""}, "monitor", "question"),
    ({"question": Q, "stages": ["nope_"]}, "monitor", "matches no stage"),
    ({"question": Q, "confirm": 0}, "monitor", "confirm"),
    ({"question": Q, "colour": "black"}, "monitor", "unknown key"),
    ({"question": Q}, "manual", "needs verifier monitor"),
])
def test_bad_guard_rejected(guard, verifier, match):
    with pytest.raises(ValueError, match=match):
        parse_plan({"task": "t", "verifier": verifier, "guard": guard,
                    "stages": [{"name": "ball_1", "prompt": "b", "success": "b?"}]})


def test_monitor_asks_the_guard_only_on_guarded_stages():
    plan = _guarded(stages=["ball_"])
    t = Transport(reply_text(guard=True))
    mon = MonitorVerifier(plan, transport=t)
    assert mon.check_now(plan.stages[1], FRAMES, 5.0).guard is True
    text = t.payloads[-1]["messages"][0]["content"][-1]["text"]
    assert Q in text and GUARD_SCHEMA in text
    mon.check_now(plan.stages[0], FRAMES, 5.0)
    text = t.payloads[-1]["messages"][0]["content"][-1]["text"]
    assert Q not in text and MONITOR_SCHEMA in text and '"guard"' not in text


def test_guard_answer_survives_a_broken_reply():
    v = parse_monitor_reply('<json>{"guard": true}</json>', 0)
    assert v.done is None and v.guard is True
    assert parse_monitor_reply(reply_text(guard="yes"), 0).guard is None   # not a boolean: no answer


def _plan_file(write_plan, **guard):
    return write_plan([stage("stack", timeout_s=1.0, on_fail="skip"), stage("ball_1", timeout_s=3.0, on_fail="skip"),
                       stage("ball_2", timeout_s=1.0, on_fail="skip")],
                      verifier="monitor", guard={"question": Q, "stages": ["ball_"], "message": "BLACK BALL", **guard})


def test_guard_yes_stops_the_run(fake, write_plan, tmp_path):
    sv = Script([lambda st, f, e: mv(None, guard=True) if st.name == "ball_1" else mv(True)])
    runner = build(fake, _plan_file(write_plan), tmp_path, sv)
    result = runner.run()
    assert not result.ok and result.guard_tripped and "ball_1" in result.reason
    assert [r.name for r in result.stages] == ["stack"]                  # ball_2 never started
    assert routes(fake)[-1] == "/stop"
    trip = events(runner.log, "guard_tripped")
    assert len(trip) == 1 and trip[0]["stage"] == "ball_1" and trip[0]["question"] == Q


def test_guard_needs_confirm_in_a_row(fake, write_plan, tmp_path):
    seq = [mv(None, guard=True), mv(None, guard=False), mv(None, guard=True), mv(None, guard=True)]
    sv = Script([lambda st, f, e: (seq.pop(0) if seq else mv(None, guard=False)) if st.name == "ball_1" else mv(True)])
    runner = build(fake, _plan_file(write_plan, confirm=2), tmp_path, sv)
    result = runner.run()
    assert result.guard_tripped
    assert [e["count"] for e in events(runner.log, "guard_yes")] == [1, 1, 2]   # a "no" resets the count


def test_guard_ignored_on_unguarded_stages(fake, write_plan, tmp_path):
    sv = Script([lambda st, f, e: mv(True, guard=True) if st.name == "stack" else mv(True)])
    result = build(fake, _plan_file(write_plan), tmp_path, sv).run()
    assert result.ok and not result.guard_tripped


def test_cli_warns_when_the_guard_cannot_be_answered(write_plan, tmp_path, capsys, monkeypatch):
    path = _plan_file(write_plan)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert cli_main([str(path), "--mock", "--verifier", "time", "--runs-dir", str(tmp_path / "r")]) in (0, 1)
    assert "WARNING: plan.guard" in capsys.readouterr().err       # time verifier never answers the guard
