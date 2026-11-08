"""Harness primitives as plain tool functions: what an agent (DimOS, MCP, anything) may call.

Every call goes through Dream's station agent HTTP API (StationClient), so Dream-1 drives the arms and every
speed cap, clamp, grip cap and home move in Dream's control loop stays in force. There is deliberately no tool
that moves a joint, homes, or resets: the tools can only give Dream-1 a prompt, stop it, look, and judge.

Tools return short strings (what an LLM reads). The plan file supplies the stage list, the success questions
and the recovery menu, so an agent composes the run from a closed set of prompts instead of inventing them
(FLARE / Hi Robot); `send_prompt` is the one free-text tool, and it is off unless allow_free_prompts=True.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from harness.frames import FrameSource
from harness.log import RunLog
from harness.plan import MAX_PROMPT_CHARS, Plan, load_plan
from harness.runner import MAX_CAMERA_AGE_S, Runner
from harness.station_client import StationClient, StationError
from harness.verifiers import make_verifier

LIVE = ("starting", "running")


class StationTools:
    def __init__(self, plan: str | Path | Plan, *, station_url: str = "http://127.0.0.1:8091",
                 view_url: str = "http://127.0.0.1:8093", variables: dict[str, str] | None = None,
                 runs_dir: str = "runs", allow_free_prompts: bool = False, dry_run: bool = False,
                 verifier: str | None = None, transport=None):
        self.plan = plan if isinstance(plan, Plan) else load_plan(plan, variables)
        self.station = StationClient(station_url, dry_run=dry_run)
        self.frames = FrameSource(view_url)
        self.runs_dir = runs_dir
        self.allow_free_prompts = allow_free_prompts
        self._verifier_name = verifier
        self._transport = transport
        self._verifiers: dict[str, Any] = {}
        self._stage_t0: dict[str, float] = {}
        self._runner: Runner | None = None
        self._run_thread: threading.Thread | None = None
        self._run_result: Any = None
        self._lock = threading.Lock()

    # -- helpers ----------------------------------------------------------------------------------------------
    def _stage(self, name: str):
        for s in self.plan.stages:
            if s.name == name:
                return s
        raise ValueError(f"no stage {name!r}; stages: {', '.join(s.name for s in self.plan.stages)}")

    def _verifier(self, stage):
        name = self._verifier_name or stage.verifier
        if name in ("manual", "monitor+manual", "claude+manual"):
            name = "monitor" if "monitor" in name else ("claude" if "claude" in name else "time")
        if name not in self._verifiers:
            self._verifiers[name] = make_verifier(name, self.plan, transport=self._transport)
        return self._verifiers[name]

    def _issue(self, prompt: str) -> dict[str, Any]:
        """Prompt change while a run is live, otherwise a start (after the same preflight the runner uses)."""
        st = self.station.status()
        if st.get("state") in LIVE:
            return self.station.set_prompt(prompt)
        problems = self.preflight_problems(st)
        if problems:
            raise RuntimeError("refused to start: " + "; ".join(problems))
        return self.station.start(prompt, self.plan.checkpoint)

    @staticmethod
    def preflight_problems(st: dict[str, Any]) -> list[str]:
        out = []
        if st.get("state") not in ("idle", "error") + LIVE:
            out.append(f"station is {st.get('state')!r}")
        out += [f"arm {a} not alive" for a, ok in (st.get("arms") or {}).items() if not ok]
        out += [f"camera {c} stale ({age})" for c, age in (st.get("cameras") or {}).items()
                if age is None or age > MAX_CAMERA_AGE_S]
        return out

    @staticmethod
    def _short(st: dict[str, Any]) -> str:
        keep = {k: st.get(k) for k in ("state", "prompt", "run", "chunks", "stalls", "clamps", "arms", "last_error")}
        return json.dumps(keep, default=str)

    def _busy_with_plan(self) -> bool:
        return self._run_thread is not None and self._run_thread.is_alive()

    # -- the tools --------------------------------------------------------------------------------------------
    def status(self) -> str:
        """The station's state, current prompt, run number, chunk/stall/clamp counts and arm health."""
        return self._short(self.station.status())

    def list_stages(self) -> str:
        """The plan: task, then each stage's name, prompt, success question, timeout and recovery menu."""
        lines = [f"task: {self.plan.task}"]
        for s in self.plan.stages:
            menu = "; ".join(f"{r.name} ({r.kind}): {r.when or r.prompt}" for r in s.menu()) or "none"
            lines.append(f"- {s.name}: prompt={s.prompt!r} success={s.success!r} timeout={s.timeout_s:g}s "
                         f"recoveries=[{menu}]")
        return "\n".join(lines)

    def start_stage(self, name: str) -> str:
        """Send a plan stage's prompt to Dream-1 (starts the run if none is live, else changes the prompt live)."""
        if self._busy_with_plan():
            return "refused: the full plan is running; call stop() first"
        stage = self._stage(name)
        st = self._issue(stage.prompt)
        self._stage_t0[name] = time.monotonic()
        return f"stage {name} prompt sent: {stage.prompt!r}. " + self._short(st)

    def recover(self, stage_name: str, recovery_name: str) -> str:
        """Send one recovery prompt from a stage's menu (a 'retry' item re-sends the stage prompt)."""
        if self._busy_with_plan():
            return "refused: the full plan is running; call stop() first"
        stage = self._stage(stage_name)
        rec = next((r for r in stage.menu() if r.name == recovery_name), None)
        if rec is None:
            return f"no recovery {recovery_name!r} for {stage_name}; menu: {[r.name for r in stage.menu()]}"
        prompt = stage.prompt if rec.kind == "retry" and not rec.legacy else rec.prompt
        st = self._issue(prompt)
        self._stage_t0[stage_name] = time.monotonic()
        return f"recovery {recovery_name} sent: {prompt!r}. " + self._short(st)

    def send_prompt(self, prompt: str) -> str:
        """Send free text to Dream-1. Off unless the operator enabled free prompts; one line, <= 512 chars."""
        if not self.allow_free_prompts:
            return "refused: free prompts are off; use start_stage or recover"
        prompt = " ".join(str(prompt).split())
        if not prompt or len(prompt) > MAX_PROMPT_CHARS:
            return f"refused: prompt must be 1..{MAX_PROMPT_CHARS} characters"
        if self._busy_with_plan():
            return "refused: the full plan is running; call stop() first"
        return "prompt sent. " + self._short(self._issue(prompt))

    def check_stage(self, name: str) -> str:
        """Ask the stage's checker (the Claude monitor in match plans) about the newest camera frames."""
        stage = self._stage(name)
        frames = self.frames.latest()
        if stage.cameras or self.plan.cameras_for_check:
            wanted = stage.cameras or self.plan.cameras_for_check
            frames = {r: j for r, j in frames.items() if any(w in r for w in wanted)}
        if not frames:
            return "no camera frames available"
        elapsed = time.monotonic() - self._stage_t0.get(name, time.monotonic())
        v = self._verifier(stage)
        verdict = getattr(v, "check_now", v.check)(stage, frames, elapsed)
        return json.dumps({k: getattr(verdict, k, None) for k in
                           ("done", "confidence", "reason", "progress", "status", "recovery", "scene_ok", "holding")},
                          default=str)

    def stop(self) -> str:
        """Stop Dream-1: the arms hold where they are (and stop the full plan if it is running)."""
        if self._runner is not None:
            self._runner.request_stop()
        try:
            return "stopped. " + self._short(self.station.stop())
        except StationError as exc:
            return f"stop answered {exc.status_code}: {exc.message}"

    def run_plan(self) -> str:
        """Run the whole plan with the harness's own state machine in the background; poll plan_status()."""
        with self._lock:
            if self._busy_with_plan():
                return "already running"
            verifiers = {name: self._verifier(s) for s in self.plan.stages for name in [s.verifier]}
            log = RunLog(self.runs_dir)
            self._runner = Runner(self.plan, self.station, self.frames, verifiers, log)
            self._run_result = None

            def go():
                self._run_result = self._runner.run()

            self._run_thread = threading.Thread(target=go, daemon=True, name="plan")
            self._run_thread.start()
            return f"plan started; log {log.path}"

    def plan_status(self) -> str:
        """Whether the background plan is running, and its result when it has finished."""
        if self._busy_with_plan():
            return "running"
        r = self._run_result
        if r is None:
            return "not started"
        stages = ", ".join(f"{s.name}:{'ok' if s.ok else 'FAIL'}" for s in r.stages)
        return f"{'SUCCESS' if r.ok else 'FAILED'}: {r.reason} ({r.seconds:.0f}s) [{stages}]"


TOOL_NAMES = ("status", "list_stages", "start_stage", "recover", "send_prompt", "check_stage", "stop", "run_plan",
              "plan_status")
