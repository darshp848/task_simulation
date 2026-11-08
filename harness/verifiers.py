"""Stage verifiers: decide from camera frames (or a person's keypress) whether a stage worked."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any, Protocol

from .plan import TOWER_STAGES, Plan, Stage

CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MIN_CONFIDENCE = 0.7


@dataclass
class Verdict:
    done: bool | None  # None = can't tell yet
    confidence: float = 0.0
    reason: str = ""
    # True for a person's key press or the clock: needs no frames and is never debounced by plan.confirm.
    immediate: bool = False


@dataclass
class MonitorVerdict(Verdict):
    progress: float | None = None
    status: str | None = None  # ok | stalled | failing
    recovery: int | None = None  # index into the stage's recovery list
    scene_ok: bool | None = None
    holding: dict | None = None
    guard: bool | None = None  # the plan guard's yes/no answer, on guarded stages only
    watch_ok: bool | None = None  # only for stages with a watcher arm
    frames_sig: str | None = None  # digest of the newest frames this verdict was computed on
    tower: list[str] | None = None  # the tower the monitor sees, bottom to top (stacking stages with a tower_target)


def valid_jpeg(data: bytes) -> bool:
    """SOI at the start and EOI near the end. A truncated or garbage frame makes the API answer 400 for the whole
    request, and the stage-start frame is in every request of the stage, so one bad frame would blind a stage."""
    return isinstance(data, (bytes, bytearray)) and len(data) > 4 and data[:2] == b"\xff\xd8" and b"\xff\xd9" in data[-32:]


def frames_digest(frames: dict[str, bytes]) -> str:
    h = hashlib.sha1()
    for role, jpeg in sorted(frames.items()):
        h.update(role.encode()); h.update(hashlib.sha1(jpeg).digest())
    return h.hexdigest()[:16]


class Verifier(Protocol):
    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict: ...


class ManualVerifier:
    """A person watches the robot and types y (done) or n (failed now) + Enter.

    A background thread reads lines. `s` + Enter calls on_stop (the CLI wires it to Runner.request_stop).
    """

    def __init__(self, input_stream: IO[str] | None = None, *, on_stop: Callable[[], None] | None = None,
                 out: IO[str] | None = None):
        self._stream = input_stream if input_stream is not None else sys.stdin
        self._on_stop = on_stop
        self._out = out if out is not None else sys.stdout
        self._lock = threading.Lock()
        self._pending: str | None = None
        self._announced: set[str] = set()
        self._thread = threading.Thread(target=self._read_loop, name="manual-verifier", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            for line in self._stream:
                key = line.strip().lower()
                if key in ("y", "n"):
                    with self._lock:
                        self._pending = key
                elif key == "s" and self._on_stop is not None:
                    self._on_stop()
        except (ValueError, OSError):
            return

    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        if stage.name not in self._announced:
            self._announced.add(stage.name)
            print(f"\n[stage {stage.name}] {stage.success}\n  type y = done, n = failed, s = stop everything (then Enter)",
                  file=self._out, flush=True)
        with self._lock:
            key, self._pending = self._pending, None
        if key == "y":
            return Verdict(True, 1.0, "operator typed y", immediate=True)
        if key == "n":
            return Verdict(False, 1.0, "operator typed n", immediate=True)
        return Verdict(None, 0.0, "waiting for operator")


class ClaudeVerifier:
    """Asks a Claude vision model a yes/no question about the cameras.

    The API key is read from ANTHROPIC_API_KEY, sent only to api.anthropic.com in the x-api-key
    header, and never logged or printed. A model "done: false" means "not yet", so it returns
    Verdict(None, ...); only a confident "done: true" ends a stage.
    """

    def __init__(self, model: str = "claude-haiku-5-5", max_tokens: int = 200, api_key: str | None = None,
                 timeout_s: float = 30.0):
        self._transport = default_transport(timeout_s, api_key, "ClaudeVerifier")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def __repr__(self) -> str:
        return f"ClaudeVerifier(model={self.model!r})"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._transport(payload)

    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        if not frames:
            return Verdict(None, 0.0, "no frames available")
        content: list[dict[str, Any]] = []
        for role, jpeg in sorted(frames.items()):
            content.append({"type": "text", "text": f"Camera: {role}"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}})
        content.append({"type": "text", "text": (
            f"{stage.success}\n\nAnswer with JSON only: "
            '{"done": true|false, "confidence": 0..1, "reason": "..."}')})
        try:
            reply = self._post({"model": self.model, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": content}]})
            text = "".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text")
            return self._parse(text)
        except urllib.error.HTTPError as exc:
            return Verdict(None, 0.0, f"verifier error: HTTP {exc.code}")
        except Exception as exc:  # any failure means "can't tell"
            return Verdict(None, 0.0, f"verifier error: {type(exc).__name__}: {str(exc)[:200]}")

    @staticmethod
    def _parse(text: str) -> Verdict:
        start = text.find("{")
        if start < 0:
            return Verdict(None, 0.0, f"verifier error: no JSON in reply: {text[:100]!r}")
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            done = obj["done"]
            confidence = float(obj.get("confidence", 0.0))
            reason = str(obj.get("reason", ""))
        except (ValueError, KeyError, TypeError, AttributeError):
            return Verdict(None, 0.0, f"verifier error: unparseable reply: {text[:100]!r}")
        if done is True and confidence >= MIN_CONFIDENCE:
            return Verdict(True, confidence, reason)
        if done is True:
            return Verdict(None, confidence, f"low confidence: {reason}")
        return Verdict(None, confidence, f"not yet: {reason}")


MONITOR_INSTRUCTION = (
    "First write 2-4 short questions about the robot and objects that decide this, and answer them from the "
    "images. Do not call the stage failing unless it is unlikely to finish within the time left. Then output "
    "JSON only between <json> and </json>:")
MONITOR_SCHEMA = (
    '{"stage_done": bool, "progress": 0..1, "status": "ok"|"stalled"|"failing", '
    '"holding": {"left": str, "right": str}, "recovery": <index into the recovery menu or null>, '
    '"scene_ok": bool, "confidence": 0..1, "reason": str}')
GUARD_SCHEMA = MONITOR_SCHEMA[:-1] + ', "guard": bool}'
Transport = Callable[[dict[str, Any]], dict[str, Any]]


BACKENDS = ("api", "subscription")
CLI_SYSTEM_PROMPT = ("You are a vision monitor for a robot station. Judge only from the images and text given, and "
                     "answer exactly in the output format the user asks for.")


def claude_backend() -> str:
    """Which way the Claude verifiers reach Claude: $HARNESS_CLAUDE_BACKEND, `api` (default) or `subscription`."""
    name = os.environ.get("HARNESS_CLAUDE_BACKEND", "api").strip().lower() or "api"
    if name not in BACKENDS:
        raise ValueError(f"HARNESS_CLAUDE_BACKEND must be one of {BACKENDS}, not {name!r}")
    return name


def claude_cli_transport(timeout_s: float = 90.0, claude_bin: str | None = None) -> Transport:
    """Sends the Messages request through the Claude Code CLI (`claude -p`), which runs on the Claude subscription
    you are logged in to (`claude` then /login), so no API key is needed.

    The request's content blocks (images + text) go in unchanged as one stream-json user message; the model and
    effort come from the payload. No tools, no MCP servers, no session saved, Claude Code's own system prompt
    replaced by a one-line one, and run from the temp dir so no CLAUDE.md is picked up. ANTHROPIC_API_KEY is
    removed from the child's environment so the subscription is what pays. Each call starts a fresh `claude`
    process: expect ~3-5 s per call (about 1-2 s more than the API)."""
    import shutil
    import subprocess
    import tempfile

    exe = claude_bin or os.environ.get("HARNESS_CLAUDE_BIN") or shutil.which("claude")
    if not exe:
        raise ValueError("HARNESS_CLAUDE_BACKEND=subscription needs the Claude Code CLI (`claude`) on PATH, "
                         "logged in to your subscription (run `claude`, then /login)")
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        cmd = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               "--model", str(payload["model"]), "--tools", "", "--strict-mcp-config", "--no-session-persistence",
               "--system-prompt", CLI_SYSTEM_PROMPT]
        effort = (payload.get("output_config") or {}).get("effort")
        if effort:
            cmd += ["--effort", str(effort)]
        lines = "".join(json.dumps({"type": "user", "message": {"role": m["role"], "content": m["content"]}}) + "\n"
                        for m in payload["messages"] if m.get("role") == "user")
        proc = subprocess.run(cmd, input=lines, capture_output=True, text=True, timeout=timeout_s,
                              cwd=tempfile.gettempdir(), env=env)
        result = None
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "result":
                result = event
        if result is None:
            raise RuntimeError(f"claude CLI exited {proc.returncode} with no result: {proc.stderr.strip()[-300:]}")
        if result.get("is_error") or result.get("subtype") != "success":
            raise RuntimeError(f"claude CLI: {result.get('subtype')}: {str(result.get('result'))[:300]}")
        return {"content": [{"type": "text", "text": result.get("result") or ""}], "usage": result.get("usage"),
                "stop_reason": result.get("stop_reason"), "duration_ms": result.get("duration_ms")}

    return send


def default_transport(timeout_s: float = 60.0, api_key: str | None = None, what: str = "MonitorVerifier") -> Transport:
    """The transport $HARNESS_CLAUDE_BACKEND asks for: the API (ANTHROPIC_API_KEY) or the subscription (claude CLI)."""
    if api_key is None and claude_backend() == "subscription":
        return claude_cli_transport(max(timeout_s, 90.0))
    key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(f"{what} needs ANTHROPIC_API_KEY to be set (or HARNESS_CLAUDE_BACKEND=subscription / "
                         "--claude-backend subscription to use your Claude subscription through the claude CLI)")
    return anthropic_transport(key, timeout_s)


def anthropic_transport(api_key: str, timeout_s: float = 60.0) -> Transport:
    """The real HTTP call. The key lives in this closure, goes only in the x-api-key header to api.anthropic.com,
    and is never logged or printed."""

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            CLAUDE_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())

    return send


def _clamp01(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return min(1.0, max(0.0, float(value)))


def parse_monitor_reply(text: str, menu_len: int) -> MonitorVerdict:
    """Text between <json> and </json> -> MonitorVerdict. Any problem -> done=None."""
    def bad(why: str) -> MonitorVerdict:
        return MonitorVerdict(None, 0.0, f"monitor error: {why}")

    start, end = text.rfind("<json>"), text.rfind("</json>")
    if start < 0 or end < start:
        return bad(f"no <json> block in reply: {text[-100:]!r}")
    try:
        obj = json.loads(text[start + len("<json>"):end])
    except ValueError:
        return bad("unparseable JSON")
    guard = obj.get("guard") if isinstance(obj, dict) and isinstance(obj.get("guard"), bool) else None
    if not isinstance(obj, dict) or not isinstance(obj.get("stage_done"), bool):
        verdict = bad("stage_done missing or not a boolean")
        verdict.guard = guard   # a safety answer counts even when the rest of the reply is broken
        return verdict
    confidence = _clamp01(obj.get("confidence")) or 0.0
    status = obj.get("status") if obj.get("status") in ("ok", "stalled", "failing") else None
    rec = obj.get("recovery")
    rec = rec if isinstance(rec, int) and not isinstance(rec, bool) and 0 <= rec < menu_len else None
    scene = obj.get("scene_ok") if isinstance(obj.get("scene_ok"), bool) else None
    watch_ok = obj.get("watch_ok") if isinstance(obj.get("watch_ok"), bool) else None
    tower = obj.get("tower")
    tower = [str(x) for x in tower] if isinstance(tower, list) else None
    holding = obj.get("holding")
    if isinstance(holding, dict):
        holding = {str(k): str(v) for k, v in holding.items()}
    else:
        holding = None
    reason = str(obj.get("reason", ""))
    done: bool | None = None
    if obj["stage_done"]:
        if confidence >= MIN_CONFIDENCE:
            done = True
        else:
            reason = f"low confidence: {reason}"
    return MonitorVerdict(done, confidence, reason, progress=_clamp01(obj.get("progress")), status=status,
                          recovery=rec, scene_ok=scene, holding=holding, guard=guard, watch_ok=watch_ok,
                          tower=tower)


class MonitorVerifier:
    """Claude-backed stage monitor: frame history in, structured verdict out (DESIGN_v2.md section 1).

    check() never blocks: it returns the result of an earlier call (once) and starts the next call in a
    background thread, at most one in flight. Results from before a prompt change or a stage change are
    discarded. check_now() is the synchronous variant for the entry check and tests.
    """

    def __init__(self, plan: Plan, *, model: str | None = None, max_tokens: int | None = None,
                 history_frames: int | None = None, api_key: str | None = None,
                 transport: Transport | None = None, timeout_s: float = 60.0):
        user = plan.claude_user
        self.plan = plan
        self.model = model or plan.claude.get("model", "claude-haiku-5-5")
        self.max_tokens = int(max_tokens if max_tokens is not None else user.get("max_tokens", 4000))
        self.grip_note = ""   # set by the runner from /arms before each check (station_addons)
        self.history_frames = int(history_frames if history_frames is not None else user.get("history_frames", 6))
        if transport is None:
            transport = default_transport(timeout_s, api_key, "MonitorVerifier")
        self._transport = transport
        self._lock = threading.Lock()
        self._gen = 0
        self._busy = False
        self._result: MonitorVerdict | None = None
        self._stage: str | None = None
        self._first: dict[str, tuple[float, bytes]] = {}
        self._history: dict[str, list[tuple[float, bytes]]] = {}
        self._last_prompt: str | None = None
        self._thread: threading.Thread | None = None
        self._guard_alert = False   # a guard "yes" whose verdict was dropped (stage or prompt changed in flight)
        # Optional local detector (harness.detect.TableDetector): draws labelled boxes on the frames sent to Claude
        self.annotator: Callable[[dict[str, bytes]], dict[str, bytes]] | None = None
        self.legend = ""
        self.bad_frames = 0

    def __repr__(self) -> str:
        return f"MonitorVerifier(model={self.model!r})"

    # -- hooks the runner calls -----------------------------------------------------------------
    def notify_prompt_change(self, prompt: str) -> None:
        with self._lock:
            self._last_prompt = prompt
            self._gen += 1
            self._result = None

    def take_guard_alert(self) -> bool:
        """True once after a dropped verdict answered the guard yes. A safety answer outlives the stage it was for."""
        with self._lock:
            alert, self._guard_alert = self._guard_alert, False
        return alert

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        """Tests: wait for the call in flight (if any)."""
        t = self._thread
        if t is not None:
            t.join(timeout_s)
        return not self._busy

    # -- frames and payload ---------------------------------------------------------------------
    @staticmethod
    def _split(frames: dict[str, bytes]) -> tuple[tuple[str, bytes] | None, dict[str, bytes]]:
        wrists = {r: j for r, j in sorted(frames.items()) if "wrist" in r}
        others = [(r, j) for r, j in sorted(frames.items()) if "wrist" not in r]
        overhead = next(((r, j) for r, j in others if any(k in r for k in ("overhead", "scene", "top"))), None)
        return (overhead or (others[0] if others else None)), wrists

    def _stage_text(self, stage: Stage, elapsed_s: float) -> str:
        lines = [f"Overall task: {self.plan.task or '(none given)'}", "Stages:"]
        for i, st in enumerate(self.plan.stages, 1):
            mark = "   <-- CURRENT STAGE" if st.name == stage.name else ""
            lines.append(f"  {i}. {st.name}: {st.prompt}{mark}")
        lines.append(f"Current stage question: {stage.success}")
        if stage.expect:
            lines.append(f"What correct behaviour looks like: {stage.expect}")
        if stage.camera_notes:
            lines.append("Camera notes (judge depth and alignment from the view that shows it best):")
            lines += [f"  {k}: {n}" for k, n in stage.camera_notes.items()]
        if self.legend:
            lines.append(self.legend)
        if getattr(self, "grip_note", ""):
            lines.append(f"Gripper sensors (measured, more reliable than the images for what is held): {self.grip_note}")
        lines.append(f"Elapsed in this stage: {elapsed_s:.0f} s of timeout {stage.timeout_s:g} s")
        lines.append(f"Last prompt sent to the robot: {self._last_prompt or stage.prompt}")
        menu = stage.menu()
        if menu:
            lines.append("Recovery menu (answer `recovery` with an index, or null):")
            for i, r in enumerate(menu):
                lines.append(f"  [{i}] {r.name}" + (f": {r.when}" if r.when else ""))
        else:
            lines.append("Recovery menu: (empty, answer recovery null)")
        guard = self.plan.guard
        guarded = guard is not None and guard.covers(stage.name)
        if guarded:
            lines.append(f"Safety question, answer it as `guard` (true = yes): {guard.question}")
        schema = GUARD_SCHEMA if guarded else MONITOR_SCHEMA
        if self.plan.tower_target and stage.name.startswith(TOWER_STAGES):
            target = "; ".join(f"{i}. {p}" for i, p in enumerate(self.plan.tower_target, 1))
            lines.append(f"Target tower, bottom to top: {target}. Report the tower you see as `tower`: a list of its "
                         "pieces bottom to top, worded like the target (empty list if there is no tower).")
            schema = schema.rstrip()[:-1] + ', "tower": [str]}'
        if stage.watch is not None:
            lines.append(f"Watcher check (answer it as watch_ok, from the {stage.watch.camera} images): "
                         f"{stage.watch.check}")
            schema = schema.rstrip()[:-1] + ', "watch_ok": bool}'
        lines += ["", MONITOR_INSTRUCTION, schema]
        return "\n".join(lines)

    def _build(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> dict[str, Any] | None:
        """Builds the request and advances the frame history. Caller holds the lock."""
        if stage.name != self._stage:
            self._stage, self._first, self._history = stage.name, {}, {}
            self._gen += 1
            self._result = None
        good = {r: j for r, j in frames.items() if valid_jpeg(j)}
        self.bad_frames += len(frames) - len(good)
        frames = good
        if self.annotator is not None:
            try:
                frames = self.annotator(frames)
            except Exception:  # noqa: BLE001 - a detector bug must not cost the monitor its frames
                pass
        overhead, wrists = self._split(frames)
        if overhead is None and not wrists:
            return None
        # Roles with a kept history: the overhead view, plus any the stage lists in history_cameras
        # (e.g. the wrist camera of an arm that is only watching, for a second viewpoint on depth).
        tracked = ([overhead] if overhead is not None else []) + [
            (r, j) for r, j in wrists.items() if any(h in r for h in stage.history_cameras)]
        tracked_roles = {r for r, _ in tracked}

        def label(t: float, role: str, tag: str = "") -> str:
            name = "overhead" if overhead is not None and role == overhead[0] else role
            note = stage.note_for(role)
            return f"t=+{t:.0f}s {name}" + (f" [{note}]" if note else "") + (f" ({tag})" if tag else "")

        images: list[tuple[str, bytes]] = []
        for role, _ in tracked:
            if role in self._first:
                images.append((label(self._first[role][0], role, "stage start"), self._first[role][1]))
            images += [(label(t, role), j) for t, j in self._history.get(role, [])]
            images.append((label(elapsed_s, role, "newest"), frames[role]))
        for role, _ in tracked:
            if role not in self._first:
                self._first[role] = (elapsed_s, frames[role])
            else:
                self._history[role] = (self._history.get(role, []) + [(elapsed_s, frames[role])])[-self.history_frames:]
        images += [(label(elapsed_s, r, "newest"), j) for r, j in wrists.items() if r not in tracked_roles]
        content: list[dict[str, Any]] = []
        for label, jpeg in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}})
        content.append({"type": "text", "text": self._stage_text(stage, elapsed_s)})
        return {"model": self.model, "max_tokens": self.max_tokens, "output_config": {"effort": "low"},
                "messages": [{"role": "user", "content": content}], "_frames_sig": frames_digest(frames)}

    def _call(self, payload: dict[str, Any], menu_len: int) -> MonitorVerdict:
        payload = dict(payload)
        sig = payload.pop("_frames_sig", None)
        try:
            reply = self._transport(payload)
            text = "".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text")
            verdict = parse_monitor_reply(text, menu_len)
            verdict.frames_sig = sig
            return verdict
        except urllib.error.HTTPError as exc:
            return MonitorVerdict(None, 0.0, f"monitor error: HTTP {exc.code}")
        except Exception as exc:  # any failure means "can't tell"
            return MonitorVerdict(None, 0.0, f"monitor error: {type(exc).__name__}: {str(exc)[:200]}")

    def _run_async(self, gen: int, payload: dict[str, Any], menu_len: int) -> None:
        verdict = self._call(payload, menu_len)
        with self._lock:
            if gen == self._gen:
                self._result = verdict
            elif verdict.guard is True:
                self._guard_alert = True
            self._busy = False

    # -- Verifier protocol ----------------------------------------------------------------------
    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        with self._lock:
            payload = None if self._busy else self._build(stage, frames, elapsed_s)
            if self._stage != stage.name:  # busy, so _build did not run
                self._stage, self._first, self._history = stage.name, {}, {}
                self._gen += 1
                self._result = None
            verdict, self._result = self._result, None
            if payload is not None:
                self._busy = True
                self._thread = threading.Thread(target=self._run_async, args=(self._gen, payload, len(stage.menu())),
                                                name="monitor-call", daemon=True)
                self._thread.start()
        if verdict is not None:
            return verdict
        if not frames:
            return Verdict(None, 0.0, "no frames available")
        return Verdict(None, 0.0, "monitor call in flight")

    def check_now(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        """Synchronous: waits for any call in flight, then asks once and returns that answer."""
        self.wait_idle(self.plan.claude_user.get("timeout_s", 60.0))
        with self._lock:
            payload = self._build(stage, frames, elapsed_s)
            self._result = None
        if payload is None:
            return Verdict(None, 0.0, "no frames available")
        return self._call(payload, len(stage.menu()))


class TimeVerifier:
    """For stages with no real check: done once 90% of the stage timeout has passed."""

    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        if elapsed_s >= stage.timeout_s * 0.9:
            return Verdict(True, 0.5, "time elapsed", immediate=True)
        return Verdict(None, 0.0, "waiting")


class EitherVerifier:
    """The operator's key press wins; otherwise Claude's verdict. Claude judges, a person can still overrule it."""

    def __init__(self, first: Verifier, second: Verifier) -> None:
        self.first, self.second = first, second

    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        verdict = self.first.check(stage, frames, elapsed_s)
        if verdict.done is not None:
            return verdict
        return self.second.check(stage, frames, elapsed_s)

    def check_now(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        verdict = self.first.check(stage, frames, elapsed_s)
        if verdict.done is not None:
            return verdict
        second = getattr(self.second, "check_now", self.second.check)
        return second(stage, frames, elapsed_s)

    def notify_prompt_change(self, prompt: str) -> None:
        for v in (self.first, self.second):
            fn = getattr(v, "notify_prompt_change", None)
            if fn is not None:
                fn(prompt)


def make_verifier(name: str, plan: Plan, *, manual: ManualVerifier | None = None,
                  transport: Transport | None = None) -> Verifier:
    """Builds a verifier by plan name. `manual` lets the CLI share one stdin reader."""
    if name == "manual":
        return manual if manual is not None else ManualVerifier()
    if name == "claude":
        return ClaudeVerifier(model=plan.claude.get("model", "claude-haiku-5-5"),
                              max_tokens=int(plan.claude.get("max_tokens", 200)))
    if name == "time":
        return TimeVerifier()
    if name == "claude+manual":
        return EitherVerifier(manual if manual is not None else ManualVerifier(),
                              make_verifier("claude", plan))
    if name == "monitor":
        return MonitorVerifier(plan, transport=transport)
    if name == "monitor+manual":
        return EitherVerifier(manual if manual is not None else ManualVerifier(),
                              make_verifier("monitor", plan, transport=transport))
    raise ValueError(f"unknown verifier {name!r}")
