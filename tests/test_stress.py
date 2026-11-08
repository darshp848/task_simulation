"""Whole v4 matches on a virtual clock (experiments/stress_match.py): faults seen on the station NUC on 7 Nov, a
policy that won't let go of the cable, Claude API outages and bad camera frames. No station, no key, ~5 s each."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import stress_match as sm  # noqa: E402

from harness.runner import _fault_kind  # noqa: E402
from harness.verifiers import MonitorVerdict, valid_jpeg  # noqa: E402


def match(name, seed=1, tmp_path=None):
    return sm.run_match(sm.SCENARIOS[name], seed, tmp_path)


@pytest.mark.parametrize("err, kind", [
    # the four kinds in the NUC's robots-inference.jsonl, worded as the station reports them
    ({"type": "DreamBadRequest", "error_code": "DA-4000", "message": "DreamAPI refused the inference request: "
      "DA-4000 BAD_REQUEST (malformed): unsafe instruction in this scene, refused: ..."}, "refusal"),
    ({"type": "ControlLoopError", "message": "action buffer empty for 241 consecutive ticks (12.05 s)"}, "transient"),
    ({"type": "LinAlgError", "message": "SVD did not converge"}, "transient"),
    ("DreamAPI refused the inference request: DA-6000 GATEWAY_FAULT (connection_timed_out)", "transient"),
    ({"type": "DreamSessionExpired", "error_code": "DA-0001", "message": "session replaced"}, "transient"),
    ("arm 'right' motor 4 over-current", None),
])
def test_fault_kinds(err, kind):
    assert _fault_kind(err) == kind


def test_valid_jpeg():
    assert valid_jpeg(b"\xff\xd8\xff\xe0abc\xff\xd9")
    assert not valid_jpeg(b"\x00\x00not-a-jpeg\xff\xd9") and not valid_jpeg(b"\xff\xd8truncated") and not valid_jpeg(b"")


def test_happy_match_scores_and_never_dqs(tmp_path):
    r = match("happy", 1, tmp_path)
    assert not r["dq"] and r["black_in_bin"] == 0 and r["points"] >= 30 and r["t_end"] >= 280


@pytest.mark.parametrize("name", ["refusal_at_start", "refusal_twice", "starve_mid", "linalg_mid"])
def test_station_faults_seen_on_the_nuc_do_not_end_the_match(name, tmp_path):
    r = match(name, 1, tmp_path)
    assert r["t_end"] >= 280, r["reason"]      # the harness kept going to the end of the clock
    assert r["out"] >= 1 and not r["dq"]


def test_refused_skip_stage_prompt_skips_the_stage_not_the_match(tmp_path):
    r = match("refusal_put_down", 1, tmp_path)
    assert r["t_end"] >= 280, r["reason"]
    stages = {n: ok for n, ok, _ in r["stages"]}
    assert stages["put_cable_down"] is False and stages.get("stack_2") is True, r["stages"]


def test_arm_loss_still_ends_the_run_without_a_restart(tmp_path):
    r = match("arm_loss", 1, tmp_path)
    assert r["t_end"] < 150 and "arm" in r["reason"]
    starts_after = [t for t, route, _ in r["posts"] if route == "/start" and t > 100]
    assert not starts_after


@pytest.mark.parametrize("name", ["no_release", "stuck_cable"])
def test_a_held_unplugged_cable_is_never_plugged_back_in(name, tmp_path):
    for seed in (1, 2):
        r = match(name, seed, tmp_path)
        assert r["replugged"] == 0 and not r["dq"]


def test_a_corrupt_stage_start_frame_does_not_blind_the_stage(tmp_path):
    r = match("corrupt_first_frame", 1, tmp_path)
    passed = {n for n, ok, _ in r["stages"] if ok}
    assert {"unplug_cable", "stack_2"} <= passed, r["stages"]


def test_no_guard_answers_on_ball_stages_stops_the_run(tmp_path):
    sc = sm.Scenario("outage_on_balls", sm.Behaviour(black_p=0.5, block_in=2.0), outages=[(120, 1e9)])
    r = sm.run_match(sc, 1, tmp_path)
    if any(n.startswith("ball_") for n, _, _ in r["stages"]) or "guard" in r["reason"]:
        assert "guard blind" in r["reason"], r["reason"]
    assert r["black_in_bin"] == 0


class _SameFrames:
    """Says done on every look, always on the same camera images (a frozen debug view)."""

    def __init__(self, sig):
        self.sig, self.n = sig, 0

    def check(self, stage, frames, elapsed_s):
        self.n += 1
        return MonitorVerdict(True, 0.9, "frozen", progress=1.0, status="ok", frames_sig=self.sig(self.n))


def _one_stage(tmp_path, verifier):
    from harness.log import RunLog
    from harness.plan import parse_plan
    from harness.runner import Runner
    plan = parse_plan({"task": "t", "verifier": "monitor", "confirm": 2, "check_every_s": 1, "time_budget_s": 40,
                       "stages": [{"name": "a", "prompt": "do a", "success": "a?", "timeout_s": 10, "min_run_s": 1,
                                   "max_attempts": 1, "on_fail": "skip"}]})
    clock = sm.VClock()
    table = sm.Table(sm.Behaviour(), 1)
    station = sm.SimStation(clock, table, [])
    return Runner(plan, station, sm.SimFrames(clock), {"monitor": verifier}, RunLog(tmp_path),
                  clock=clock, sleep=clock.sleep).run()


def test_two_done_verdicts_on_the_same_frames_do_not_confirm(tmp_path):
    assert not _one_stage(tmp_path, _SameFrames(lambda n: "frozen")).stages[0].ok


def test_two_done_verdicts_on_new_frames_confirm(tmp_path):
    assert _one_stage(tmp_path, _SameFrames(lambda n: f"f{n}")).stages[0].ok
