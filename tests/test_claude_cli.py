"""The subscription backend: the monitor's request goes through the `claude` CLI instead of the API.

A fake `claude` script stands in for the CLI, so these run offline and spend nothing."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from harness.verifiers import MonitorVerifier, claude_cli_transport, default_transport, make_verifier

FAKE = """#!{python}
import json, os, sys
args = sys.argv[1:]
lines = [json.loads(l) for l in sys.stdin if l.strip()]
with open(os.environ["FAKE_CLAUDE_LOG"], "w") as fh:
    json.dump({{"args": args, "lines": lines, "has_key": "ANTHROPIC_API_KEY" in os.environ, "cwd": os.getcwd()}}, fh)
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
print(json.dumps({{"type": "system", "subtype": "init", "apiKeySource": "none"}}))
if mode == "ok":
    text = 'Q: held? no.\\n<json>{{"stage_done": true, "progress": 1, "status": "ok", "holding": {{"left": "nothing", "right": "nothing"}}, "recovery": null, "scene_ok": true, "confidence": 0.9, "reason": "done", "guard": false}}</json>'
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": False, "result": text, "duration_ms": 1234}}))
elif mode == "error":
    print(json.dumps({{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "rate limited"}}))
else:
    sys.exit(3)
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "log.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("HARNESS_CLAUDE_BIN", str(exe))
    return log


def _plan():
    from harness.plan import parse_plan

    return parse_plan({"task": "t", "stages": [{"name": "a", "prompt": "do a", "success": "is a done?",
                                                 "timeout_s": 30}]})


def test_cli_transport_sends_the_request_unchanged(fake_claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SHOULD-NOT-LEAK")
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
               {"type": "text", "text": "question"}]
    reply = claude_cli_transport()({"model": "claude-haiku-5-5", "max_tokens": 4000,
                                    "output_config": {"effort": "low"},
                                    "messages": [{"role": "user", "content": content}]})
    assert "<json>" in reply["content"][0]["text"]
    seen = json.loads(fake_claude.read_text())
    args = seen["args"]
    assert args[args.index("--model") + 1] == "claude-haiku-5-5"
    assert args[args.index("--effort") + 1] == "low"
    assert args[args.index("--tools") + 1] == ""
    assert "-p" in args and "--no-session-persistence" in args and "--strict-mcp-config" in args
    assert seen["lines"] == [{"type": "user", "message": {"role": "user", "content": content}}]
    assert not seen["has_key"]          # the subscription pays, never the API key


@pytest.mark.parametrize("mode", ["error", "crash"])
def test_cli_failures_are_monitor_errors_not_crashes(fake_claude, monkeypatch, mode):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    plan = _plan()
    mon = MonitorVerifier(plan, transport=claude_cli_transport())
    v = mon.check_now(plan.stages[0], {"overhead": b"\xff\xd8\xff\xe0fake\xff\xd9"}, 5.0)
    assert v.done is None and v.reason.startswith("monitor error")


def test_backend_switch(fake_claude, monkeypatch):
    plan = _plan()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HARNESS_CLAUDE_BACKEND", "subscription")
    mon = make_verifier("monitor", plan)
    v = mon.check_now(plan.stages[0], {"overhead": b"\xff\xd8\xff\xe0fake\xff\xd9"}, 5.0)
    assert v.done is True and v.reason == "done"
    monkeypatch.setenv("HARNESS_CLAUDE_BACKEND", "api")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        default_transport()
    monkeypatch.setenv("HARNESS_CLAUDE_BACKEND", "bogus")
    with pytest.raises(ValueError, match="HARNESS_CLAUDE_BACKEND"):
        default_transport()


def test_subscription_without_cli_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("HARNESS_CLAUDE_BIN", "")
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setenv("HARNESS_CLAUDE_BACKEND", "subscription")
    with pytest.raises(ValueError, match="claude"):
        default_transport()
