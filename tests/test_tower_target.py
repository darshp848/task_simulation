"""tower_target: the v4 plan's target tower (Drive IMG_2697) reaches the monitor and the log, never gates a stage."""

from pathlib import Path

import pytest

from harness.plan import load_plan, parse_plan
from harness.runner import tower_matches
from harness.verifiers import MonitorVerifier, parse_monitor_reply

V4 = Path(__file__).resolve().parents[1] / "plans" / "hill_climb_v4.yaml"


def test_v4_target_is_the_photo_tower():
    t = load_plan(V4).tower_target
    assert [p.split(",")[0] for p in t] == ["blue cube", "pink cube", "blue cube", "yellow square stick",
                                           "pink round stick"]
    assert "upright" in t[3] and "upright" in t[4] and all("sideways" in p for p in t[:3])


def test_target_goes_to_stacking_stages_only():
    plan = load_plan(V4)
    mon = MonitorVerifier(plan, transport=lambda payload: {})
    stage = {s.name: s for s in plan.stages}
    assert "Target tower, bottom to top" in mon._stage_text(stage["stack_2"], 5)
    assert '"tower": [str]' in mon._stage_text(stage["higher_1"], 5)
    assert "Target tower" not in mon._stage_text(stage["unplug_cable"], 5)
    assert "Target tower" not in mon._stage_text(stage["ball_1"], 5)


@pytest.mark.parametrize("seen, ok", [
    ([], True),
    (["blue cube"], True),
    (["Blue cube (hole to the side)", "pink cube", "blue cube", "yellow stick standing up"], True),
    (["blue cube", "blue cube"], False),           # wrong second piece
    (["pink cube"], False),
    (["blue cube", "pink cube", "blue cube", "pink stick", "yellow stick"], False),   # sticks swapped
    (["blue cube"] * 6, False),                    # taller than the target
])
def test_tower_matches(seen, ok):
    assert tower_matches(seen, load_plan(V4).tower_target) is ok


def test_monitor_reply_carries_the_tower():
    v = parse_monitor_reply('<json>{"stage_done": false, "tower": ["blue cube", "pink cube"]}</json>', 2)
    assert v.tower == ["blue cube", "pink cube"]


def test_bad_tower_target_rejected():
    with pytest.raises(ValueError, match="tower_target"):
        parse_plan({"stages": [{"name": "a", "prompt": "p"}], "tower_target": "blue cube"})


# -- tower_prompts: name the next piece -------------------------------------------------------------------------
class _TowerSays:
    """Monitor stand-in: reports a fixed tower, never done."""

    def __init__(self, tower):
        self.tower = tower

    def check(self, stage, frames, elapsed_s):
        from harness.verifiers import MonitorVerdict
        return MonitorVerdict(None, 0.5, "x", progress=0.3, status="ok", tower=self.tower)

    check_now = check


def _prompt_for(tower, tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import stress_match as sm
    from harness.log import RunLog
    from harness.runner import Runner
    plan = load_plan(V4)
    clock = sm.VClock()
    runner = Runner(plan, sm.SimStation(clock, sm.Table(sm.Behaviour(), 1), []), sm.SimFrames(clock),
                    {"monitor": _TowerSays(tower)}, RunLog(tmp_path), clock=clock, sleep=clock.sleep)
    stage = next(s for s in plan.stages if s.name == "stack_2")
    return runner._stage_prompt(stage, runner.verifiers["monitor"])


@pytest.mark.parametrize("tower, prompt", [
    (["blue cube"], "put the pink cube on top of the stack"),
    ([], "put the pink cube on top of the stack"),
    (["blue cube", "pink cube", "blue cube"], "stand the yellow square stick upright on top of the stack"),
    (["blue cube", "pink cube", "blue cube", "yellow square stick"], "stand the pink round stick upright on top of the stack"),
    (["blue cube"] * 5, "put a block on top of the stack"),       # target complete: the generic prompt
    (None, "put a block on top of the stack"),                    # no tower reported
])
def test_tower_prompts_name_the_next_piece(tower, prompt, tmp_path):
    assert _prompt_for(tower, tmp_path) == prompt


def test_tower_prompts_need_a_template_for_every_piece():
    with pytest.raises(ValueError, match="no template"):
        parse_plan({"stages": [{"name": "a", "prompt": "p"}], "tower_target": ["red ball"],
                    "tower_prompts": {"cube": "put the <piece> on the stack"}})
