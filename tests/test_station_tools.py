"""integrations.station_tools against fake_station: the primitives an agent (DimOS / MCP) can call."""

import json

from conftest import make_station, stage
from fake_station import FakeVerifier
from harness.verifiers import Verdict
from integrations.station_tools import TOOL_NAMES, StationTools


def _tools(fake, write_plan, **kw):
    plan = write_plan([stage("a", recovery=[{"name": "lift", "kind": "reset", "prompt": "lift the arms"}]),
                       stage("b")])
    return StationTools(plan, station_url=fake.api_url, view_url=fake.view_url, **kw)


def test_tools_drive_the_station_through_its_api_only(write_plan, tmp_path):
    fake = make_station()
    try:
        t = _tools(fake, write_plan)
        assert "a: prompt='do a'" in t.list_stages() and "lift (reset)" in t.list_stages()
        assert "stage a prompt sent" in t.start_stage("a")
        assert json.loads(t.status())["state"] in ("starting", "running")
        assert "recovery lift sent: 'lift the arms'" in t.recover("a", "lift")
        assert "refused: free prompts are off" in t.send_prompt("dance")
        assert "stage b prompt sent" in t.start_stage("b")
        assert t.stop().startswith("stopped")
        routes = [r for _, r, _ in fake.posts()]
        assert set(routes) <= {"/start", "/prompt", "/stop"}           # never /home or /reset
        assert [b["prompt"] for _, r, b in fake.posts() if r in ("/start", "/prompt")] == ["do a", "lift the arms", "do b"]
    finally:
        fake.close()


def test_check_stage_uses_the_verifier(write_plan, tmp_path):
    fake = make_station()
    try:
        t = _tools(fake, write_plan)
        t._verifiers["manual"] = t._verifiers["time"] = FakeVerifier({"a": [Verdict(True, 0.9, "looks done")]})
        t._verifier_name = "time"
        t.start_stage("a")
        import time
        time.sleep(0.3)
        out = json.loads(t.check_stage("a"))
        assert out["done"] is True and out["reason"] == "looks done"
    finally:
        fake.close()


def test_free_prompts_and_unknown_names(write_plan):
    fake = make_station()
    try:
        t = _tools(fake, write_plan, allow_free_prompts=True)
        assert t.send_prompt("pick up the cube").startswith("prompt sent")
        assert "refused" in t.send_prompt("x" * 600)
        assert "no recovery" in t.recover("a", "nope")
        assert set(TOOL_NAMES) <= set(dir(t))
    finally:
        fake.close()


def test_start_refused_when_an_arm_is_dead(write_plan):
    fake = make_station()
    try:
        fake.arms["left"] = False
        t = _tools(fake, write_plan)
        try:
            t.start_stage("a")
            raise AssertionError("should refuse")
        except RuntimeError as exc:
            assert "arm left not alive" in str(exc)
    finally:
        fake.close()
