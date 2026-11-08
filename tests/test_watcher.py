"""Watcher-arm support: per-stage cameras, camera notes, frame history for a watching wrist camera, plan vars."""

from pathlib import Path

import pytest

from harness.plan import load_plan, parse_plan
from harness.verifiers import MonitorVerifier
from test_v2 import JPEG, Transport, reply_text


def _st(**kw):
    plan = parse_plan({"task": "t", "stages": [{"name": "plug", "prompt": "plug it", "success": "in?", **kw}]})
    return plan, plan.stages[0]


def test_stage_camera_fields_parse_and_validate():
    _, s = _st(cameras=["scene", "wrist_l"], history_cameras=["wrist_l"],
               camera_notes={"wrist_l": "side view", "scene": "top view"})
    assert s.cameras == ["scene", "wrist_l"] and s.history_cameras == ["wrist_l"]
    assert s.note_for("wrist_l") == "side view" and s.note_for("wrist_r") == ""
    with pytest.raises(ValueError, match="camera_notes"):
        _st(camera_notes={"wrist_l": ""})
    with pytest.raises(ValueError, match="history_cameras"):
        _st(history_cameras="wrist_l")


def test_monitor_keeps_history_for_watcher_camera_and_labels_notes():
    plan, st = _st(history_cameras=["wrist_l"], camera_notes={"wrist_l": "side view from the watching arm"})
    t = Transport(reply_text())
    mon = MonitorVerifier(plan, transport=t, history_frames=2)
    frames = lambda n: {"video.scene": JPEG + bytes([n]), "video.wrist_l": JPEG[:2] + b"l" + bytes([n]) + JPEG[-2:],  # noqa: E731
                        "video.wrist_r": JPEG[:2] + b"r" + bytes([n]) + JPEG[-2:]}
    for i in range(4):
        mon.check_now(st, frames(i), 3.0 * i)
    content = t.payloads[-1]["messages"][0]["content"]
    labels = [b["text"] for b in content if b["type"] == "text"][:-1]
    watch = [lbl for lbl in labels if "wrist_l" in lbl]
    assert watch[0].startswith("t=+0s video.wrist_l [side view from the watching arm] (stage start)")
    assert len(watch) == 4                                   # start + 2 history + newest
    assert sum("wrist_r" in lbl for lbl in labels) == 1      # not tracked: newest only
    assert sum("overhead" in lbl for lbl in labels) == 4
    assert "Camera notes" in content[-1]["text"] and "side view from the watching arm" in content[-1]["text"]


def test_stage_cameras_filter_frames(tmp_path):
    from harness.runner import Runner

    class Frames:
        def latest(self):
            return {"scene": b"s", "wrist_l": b"l", "wrist_r": b"r"}

    plan, st = _st(cameras=["wrist_l"])
    r = Runner(plan, station=None, frames=Frames(), verifiers={}, log=None)
    assert set(r._frames_for_check(st)) == {"wrist_l"}
    plan.cameras_for_check = ["scene"]
    _, st2 = _st()
    assert set(r._frames_for_check(st2)) == {"scene"}


def test_plan_vars_defaults_and_override(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text('task: t\nvars: {arm: left}\nstages:\n  - {name: a, prompt: "use the {arm} arm", success: "ok?"}\n')
    assert load_plan(p).stages[0].prompt == "use the left arm"
    assert load_plan(p, {"arm": "right"}).stages[0].prompt == "use the right arm"


def test_v4_plan_has_no_plug_in_and_notes_the_stacking_camera():
    plan = load_plan(Path(__file__).resolve().parent.parent / "plans" / "hill_climb_v4.yaml")
    names = [s.name for s in plan.stages]
    assert not any(n.startswith("plug") for n in names)
    assert not any(s.prompt.startswith("plug") for s in plan.stages)
    assert names.index("unplug_cable") < names.index("stack_2")
    assert "wrist_r" in plan.stages[names.index("stack_2")].camera_notes


# -- the watcher is kept watching -----------------------------------------------------------------------------
WATCH = {"camera": "wrist_l", "check": "Does wrist_l show the hub ports from the side?",
         "prompt": "move the left arm back beside the hub and hold it still", "settle_s": 0.1, "max_fixes": 1}


def test_watch_field_parses_and_adds_history():
    _, s = _st(watch=WATCH)
    assert s.watch.camera == "wrist_l" and s.watch.max_fixes == 1 and "wrist_l" in s.history_cameras
    with pytest.raises(ValueError, match="watch"):
        _st(watch={"camera": "wrist_l"})


def test_monitor_asks_for_watch_ok_and_parses_it():
    plan, st = _st(watch=WATCH)
    t = Transport(reply_text(watch_ok=False))
    v = MonitorVerifier(plan, transport=t).check_now(st, {"video.scene": JPEG, "video.wrist_l": JPEG}, 5.0)
    text = t.payloads[0]["messages"][0]["content"][-1]["text"]
    assert "Watcher check" in text and '"watch_ok": bool' in text and v.watch_ok is False


def test_runner_puts_the_watcher_back_then_resumes_the_stage(write_plan, tmp_path):
    from conftest import make_station
    from test_v2 import Script, build, events, mv, prompts

    fake = make_station()
    try:
        lost = mv(None, status="ok", watch_ok=False)
        ok = mv(None, status="ok", watch_ok=True)
        script = Script([lost, lost, ok, mv(True, watch_ok=True), mv(True, watch_ok=True)])
        plan = write_plan([dict(name="plug", prompt="plug it in", success="in?", timeout_s=10, min_run_s=0.05,
                                verifier="monitor", watch=WATCH)], confirm=2)
        result = build(fake, plan, tmp_path, script).run()
        assert result.ok, result.reason
        assert prompts(fake)[:3] == ["plug it in", WATCH["prompt"], "plug it in"]
        assert len(events(_last_log(tmp_path), "watch_lost")) == 1
    finally:
        fake.close()


def _last_log(tmp_path):
    runs = sorted((tmp_path / "runs").iterdir())

    class L:
        path = runs[-1]
    return L
