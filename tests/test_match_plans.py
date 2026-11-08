"""The Hill Climb match plan (rubric v4), --var placeholders, done_message and the task check."""

import json
from pathlib import Path

import pytest

from harness.cli import main
from harness.plan import load_plan, substitute
from harness import tasks

PLANS = Path(__file__).resolve().parent.parent / "plans"
V4 = PLANS / "hill_climb_v4.yaml"

# The task texts of TELEOP_PLAN.md the match plan sends (T3, plug-in, was dropped on 7 Nov).
TELEOP_TASKS = {
    "unplug the USB cable at the end of the hub nearest the right arm",
    "put the held cable down on the table away from the hub",
    "put a block on top of the stack",
    "put a golf ball in the bin",
    "open both grippers and move the arms up",
}
PASS_BAR = ("unplug_cable", "stack_2", "stack_3", "tower_pass")


def test_substitute_fills_and_leaves_flow_maps():
    assert substitute("a {color} ball, {name: x}", {"color": "orange"}) == "a orange ball, {name: x}"


def test_substitute_missing_var_is_an_error():
    with pytest.raises(ValueError, match="color"):
        substitute("put a {color} ball", {})


def test_v4_plan_follows_the_rules():
    plan = load_plan(V4)
    assert plan.verifier == "monitor"            # no human input after "go"
    assert plan.confirm >= 2                     # a false "done" is a DQ
    assert plan.time_budget_s <= 300
    assert plan.done_message
    assert all(s.verifier == "monitor" for s in plan.stages)
    names = [s.name for s in plan.stages]
    assert names[0] == "unplug_cable"
    for n in PASS_BAR:                           # pass bars are never skipped
        assert plan.stages[names.index(n)].on_fail != "skip"
    # nothing of Stage 2 before the cable is out, nothing of Stage 3 before the tower passes
    first_ball = min(i for i, n in enumerate(names) if n.startswith("ball_"))
    assert names.index("tower_pass") < first_ball
    # every stage after the last pass bar is optional
    assert all(s.on_fail == "skip" for s in plan.stages[names.index("tower_pass") + 1:])


def test_v4_plan_sends_only_recorded_task_texts():
    assert set(tasks.plan_prompts(load_plan(V4))) == TELEOP_TASKS


def test_done_message_printed_on_success(tmp_path, capsys):
    plan = tmp_path / "p.yaml"
    plan.write_text(
        'task: t\ncheck_every_s: 0.2\ndone_message: "CALL STOP NOW"\nverifier: time\nstages:\n'
        '  - {name: a, prompt: "do {thing}", success: "done?", timeout_s: 1, min_run_s: 0.1}\n')
    rc = main([str(plan), "--mock", "--var", "thing=it", "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert rc == 0 and "CALL STOP NOW" in out


def test_bad_var_rejected():
    assert main([str(V4), "--mock", "--var", "plug_cam"]) == 2


def _dataset(root: Path, name: str, episodes: list[tuple[str, int]], fps: int = 30) -> Path:
    d = root / name / "meta"
    d.mkdir(parents=True)
    (d / "info.json").write_text(json.dumps({"fps": fps}))
    (d / "episodes.jsonl").write_text("\n".join(
        json.dumps({"episode_index": i, "tasks": [t], "length": n}) for i, (t, n) in enumerate(episodes)))
    texts = list(dict.fromkeys(t for t, _ in episodes))
    (d / "tasks.jsonl").write_text("\n".join(json.dumps({"task_index": i, "task": t}) for i, t in enumerate(texts)))
    return root / name


def test_task_check_counts_and_flags(tmp_path, capsys):
    plan = tmp_path / "p.yaml"
    plan.write_text(
        'task: t\nstages:\n'
        '  - {name: a, prompt: "unplug the cable", success: "ok?",\n'
        '     recovery: [{name: r, kind: reset, prompt: "open both grippers"}, {name: k, kind: retry}]}\n'
        '  - {name: b, prompt: "Put A Block  on the stack", success: "ok?"}\n')
    parent = tmp_path / "ft"
    _dataset(parent, "one-eef", [("unplug the cable", 300)] * 3)
    _dataset(parent, "two-eef", [("unplug the cable", 600), ("put a block on the stack", 450), ("wave", 90)])
    rc = tasks.main([str(plan), str(parent)])
    out = capsys.readouterr().out
    assert rc == 1                                            # "open both grippers" has no demos
    assert "   4 eps" in out
    assert "MISSING" in out and "'open both grippers'" in out and "a/r" in out
    assert "differs only in case/spaces from: 'put a block on the stack'" in out
    assert "'wave'" in out                                    # recorded but never prompted
    plan.write_text('task: t\nstages:\n  - {name: a, prompt: "unplug the cable", success: "ok?"}\n')
    assert tasks.main([str(plan), str(parent / "one-eef")]) == 0
    assert tasks.main([str(plan), str(parent / "one-eef"), "--min", "5"]) == 1
    assert tasks.main([str(plan), str(tmp_path / "nothing")]) == 2


def test_v4_plan_guards_every_ball_stage_and_nothing_else():
    plan = load_plan(V4)
    assert plan.guard is not None and "black" in plan.guard.question and plan.guard.confirm == 1
    for s in plan.stages:
        assert plan.guard.covers(s.name) == s.name.startswith("ball_"), s.name
