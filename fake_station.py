"""Offline stand-in for Dream's station agent (API port) and debug view (view port).

Implements the routes of `robots/station_api.py` and `dream/debug_view.py` with the station controller's
real state machine: idle -> starting -> running; stop -> stopping -> idle; home -> homing -> idle;
reset -> resetting -> homing -> idle; 409 for refused transitions, 400 for bad bodies, 404 for unknown routes.

    fs = FakeStation(scenario="happy"); fs.start(); ... fs.api_url, fs.view_url, fs.calls ...; fs.close()

Run alone: `python fake_station.py [scenario]` serves on 8091 / 8093.
No hardware is ever touched.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

try:  # the harness's own Verdict when the package is importable, else a local twin
    from harness.verifiers import Verdict  # type: ignore
except Exception:  # pragma: no cover

    @dataclass
    class Verdict:  # type: ignore[no-redef]
        done: bool | None
        confidence: float
        reason: str


IDLE, STARTING, RUNNING, STOPPING, HOMING, RESETTING, ERROR = (
    "idle", "starting", "running", "stopping", "homing", "resetting", "error")

SCENARIOS = ("happy", "arm_dies", "start_error", "slow_start", "camera_stale")
ROLES = ("video.scene", "video.wrist_l", "video.wrist_r")
ARMS = ("left", "right")
MAX_BODY_BYTES = 64 * 1024

# A tiny valid 8x8 JPEG.
TINY_JPEG: bytes = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1x"
    "eXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAAR"
    "CAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEG"
    "E1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWG"
    "h4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEB"
    "AQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYk"
    "NOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0"
    "tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDHooorhPqD/9k="
)


class _Refused(Exception):
    """A refused transition: HTTP 409."""


class FakeStation:
    """Two ThreadingHTTPServers on 127.0.0.1 (API + debug view) around one state machine.

    Timing knobs (seconds, all scaled down from the real station so tests are fast): `start_s` (starting ->
    running), `slow_start_s` (the slow_start scenario's starting time, default 5), `stop_s`, `home_s`, `reset_s`,
    `arm_dies_after_s` (default 3), `tick_s` (chunk / trip cadence).
    """

    def __init__(self, api_port: int = 0, view_port: int = 0, *, scenario: str = "happy",
                 start_s: float = 0.05, slow_start_s: float = 5.0, stop_s: float = 0.03, home_s: float = 0.1,
                 reset_s: float = 0.05, arm_dies_after_s: float = 3.0, tick_s: float = 0.02,
                 prompt: str | None = None, checkpoint: str | None = "dream-1") -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; one of {SCENARIOS}")
        self.scenario = scenario
        self._api_port, self._view_port = api_port, view_port
        self.start_s = slow_start_s if scenario == "slow_start" else start_s
        self.stop_s, self.home_s, self.reset_s = stop_s, home_s, reset_s
        self.arm_dies_after_s, self.tick_s = arm_dies_after_s, tick_s
        self.calls: list[tuple[str, str, Any]] = []
        self._lock = threading.RLock()
        self.state = IDLE
        self.prompt = prompt
        self.checkpoint = checkpoint
        self.run_number = 0
        self.chunks = 0
        self.stalls = 0
        self.clamps = 0
        self.arms = {name: True for name in ARMS}
        self.last_error: str | None = None
        self._run_started: float | None = None
        self._run_epoch = 0  # bumped when a run ends, so stale timers do nothing
        self._trips: deque[dict[str, Any]] = deque(maxlen=50)
        self._trip_n = 0
        self._servers: list[ThreadingHTTPServer] = []
        self._threads: list[threading.Thread] = []
        self._closed = threading.Event()
        self.api_url = ""
        self.view_url = ""

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> "FakeStation":
        api = ThreadingHTTPServer(("127.0.0.1", self._api_port), self._api_handler())
        view = ThreadingHTTPServer(("127.0.0.1", self._view_port), self._view_handler())
        for server in (api, view):
            server.daemon_threads = True
            server.allow_reuse_address = True
            self._servers.append(server)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True, name="fake-station")
            thread.start()
            self._threads.append(thread)
        ticker = threading.Thread(target=self._tick_loop, daemon=True, name="fake-station-tick")
        ticker.start()
        self._threads.append(ticker)
        self.api_url = f"http://127.0.0.1:{api.server_address[1]}"
        self.view_url = f"http://127.0.0.1:{view.server_address[1]}"
        return self

    def close(self) -> None:
        self._closed.set()
        for server in self._servers:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(2.0)
        self._servers, self._threads = [], []

    def __enter__(self) -> "FakeStation":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def posts(self) -> list[tuple[str, str, Any]]:
        """The recorded POSTs only."""
        return [call for call in self.calls if call[0] == "POST"]

    # ---------------------------------------------------------------- state machine
    def _set_state(self, state: str) -> None:
        with self._lock:
            self.state = state

    def _dead_arm(self) -> str | None:
        return next((name for name, alive in self.arms.items() if not alive), None)

    def _status(self) -> dict[str, Any]:
        with self._lock:
            stale = self.scenario == "camera_stale"
            ages = {role.removeprefix("video."): (9.0 if stale and role == "video.scene" else 0.05) for role in ROLES}
            return {
                "state": self.state, "prompt": self.prompt, "checkpoint": self.checkpoint,
                "action_space": None, "steps": 1, "run": self.run_number, "chunks": self.chunks,
                "stalls": self.stalls, "clamps": self.clamps, "arms": dict(self.arms), "cameras": ages,
                "last_error": self.last_error, "started_at": None,
                "run_elapsed_s": None if self._run_started is None else round(time.monotonic() - self._run_started, 1),
            }

    @staticmethod
    def _checked_prompt(prompt: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return prompt.strip()

    def _do_start(self, body: dict) -> dict:
        prompt, checkpoint = body.get("prompt"), body.get("checkpoint")
        with self._lock:
            if self.state not in (IDLE, ERROR):
                raise _Refused(f"start refused: station is {self.state}")
            dead = self._dead_arm()
            if dead:
                raise _Refused(f"start refused: arm {dead!r} driver is dead; reset the arms first")
            if prompt is not None:
                self.prompt = self._checked_prompt(prompt)
            if checkpoint is not None:
                if not isinstance(checkpoint, str) or not checkpoint.strip():
                    raise ValueError("checkpoint must be a non-empty string")
                self.checkpoint = checkpoint
            if not self.prompt:
                raise _Refused("start refused: no task prompt; send the task with the start")
            self.last_error = None
            self.run_number += 1
            self.chunks = 0
            self._run_started = time.monotonic()
            self._set_state(STARTING)
            epoch = self._run_epoch
            result = self._status()
        threading.Thread(target=self._finish_start, args=(epoch,), daemon=True).start()
        return result

    def _finish_start(self, epoch: int) -> None:
        if self._closed.wait(self.start_s):
            return
        with self._lock:
            if self._run_epoch != epoch or self.state != STARTING:
                return
            if self.scenario == "start_error":
                self.last_error = "model refused: checkpoint not found (fake start_error scenario)"
                self._run_epoch += 1
                self._set_state(ERROR)
                return
            self._set_state(RUNNING)
        if self.scenario == "arm_dies":
            if self._closed.wait(self.arm_dies_after_s):
                return
            with self._lock:
                if self._run_epoch == epoch and self.state == RUNNING:
                    self.arms["right"] = False
                    self.last_error = "arm 'right' driver stopped answering (fake arm_dies scenario)"
                    self._run_epoch += 1
                    self._set_state(ERROR)

    def _do_stop(self) -> dict:
        with self._lock:
            if self.state == HOMING:
                raise _Refused("stop refused: a home move is in progress; it ends within seconds")
            if self.state == RESETTING:
                raise _Refused("stop refused: a reset is in progress; it ends within seconds")
            if self.state == STOPPING:
                raise _Refused("stop refused: already stopping")
            if self.state not in (STARTING, RUNNING):
                return self._status()  # nothing to do (idle, or error: the run already ended)
            self._run_epoch += 1
            self._set_state(STOPPING)
        time.sleep(self.stop_s)
        with self._lock:
            self._set_state(IDLE)
            return self._status()

    def _do_home(self) -> dict:
        with self._lock:
            if self.state not in (IDLE, ERROR):
                raise _Refused(f"home refused: station is {self.state}; stop the run first")
            dead = self._dead_arm()
            if dead:
                raise _Refused(f"home refused: arm {dead!r} driver is dead; reset the arms first")
            self.last_error = None
            self._set_state(HOMING)
        time.sleep(self.home_s)
        with self._lock:
            self._set_state(IDLE)
            return self._status()

    def _do_reset(self) -> dict:
        with self._lock:
            if self.state not in (IDLE, ERROR):
                raise _Refused(f"reset refused: station is {self.state}; stop the run first")
            self.last_error = None
            self._set_state(RESETTING)
            result = self._status()
        threading.Thread(target=self._finish_reset, daemon=True).start()
        return result

    def _finish_reset(self) -> None:
        if self._closed.wait(self.reset_s):
            return
        with self._lock:
            self.arms = {name: True for name in ARMS}
            self._set_state(HOMING)
        if self._closed.wait(self.home_s):
            return
        self._set_state(IDLE)

    def _do_prompt(self, body: dict) -> dict:
        if "prompt" not in body:
            raise ValueError("body must be a JSON object with a 'prompt' key")
        with self._lock:
            self.prompt = self._checked_prompt(body["prompt"])
            return self._status()

    # ---------------------------------------------------------------- trips (the debug view)
    def _tick_loop(self) -> None:
        while not self._closed.wait(self.tick_s):
            with self._lock:
                if self.state != RUNNING:
                    continue
                self.chunks += 1
                self._trip_n += 1
                # each trip is a new exposure: a COM segment with the trip number makes the bytes differ, as a
                # live camera's do (the harness treats byte-identical frames as one look)
                com = b"\xff\xfe\x00\x08" + self._trip_n.to_bytes(6, "big")
                jpeg = TINY_JPEG[:2] + com + TINY_JPEG[2:]
                frames = {role: [base64.b64encode(jpeg).decode()] for role in ROLES}
                self._trips.append({
                    "n": self._trip_n, "sent_s": round(time.monotonic() - (self._run_started or 0), 3),
                    "frames": frames, "observation": {"prompt": self.prompt, "run": self.run_number},
                    "action": [0.0] * 14, "timing": {"rtt_ms": 12.0}, "error": None, "trained": [],
                })

    def _trip_list(self) -> list[dict]:
        with self._lock:
            return [{"n": t["n"], "sent_s": t["sent_s"], "error": t["error"], "rtt_ms": 12.0} for t in self._trips]

    def _trip(self, n: int) -> dict | None:
        with self._lock:
            return next((t for t in self._trips if t["n"] == n), None)

    # ---------------------------------------------------------------- HTTP
    def _api_handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                route = self.path.split("?", 1)[0]
                fake.calls.append(("GET", route, None))
                if route != "/status":
                    self._reply(404, {"error": f"no such route: GET {self.path}"})
                else:
                    self._reply(200, fake._status())

            def do_POST(self) -> None:
                route = self.path.split("?", 1)[0]
                try:
                    body = self._read_json()
                except ValueError as exc:
                    fake.calls.append(("POST", route, None))
                    self._reply(400, {"error": str(exc)})
                    return
                fake.calls.append(("POST", route, body))
                try:
                    if route == "/start":
                        result = fake._do_start(body)
                    elif route == "/stop":
                        result = fake._do_stop()
                    elif route == "/home":
                        result = fake._do_home()
                    elif route == "/reset":
                        result = fake._do_reset()
                    elif route == "/prompt":
                        result = fake._do_prompt(body)
                    else:
                        self._reply(404, {"error": f"no such route: POST {route}"})
                        return
                except _Refused as exc:
                    self._reply(409, {"error": str(exc)})
                except ValueError as exc:
                    self._reply(400, {"error": str(exc)})
                except Exception as exc:  # pragma: no cover
                    self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})
                else:
                    self._reply(200, result)

            def do_PUT(self) -> None:
                self._reply(405, {"error": "method not allowed"})

            do_DELETE = do_PUT

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY_BYTES:
                    raise ValueError(f"request body of {length} bytes is larger than {MAX_BODY_BYTES}")
                raw = self.rfile.read(length) if length else b""
                if not raw.strip():
                    return {}
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"request body is not valid JSON: {exc.msg}") from exc
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                return body

            def _reply(self, code: int, payload: dict) -> None:
                data = json.dumps(payload, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        return Handler

    def _view_handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                number = path.removeprefix("/trip/")
                if path == "/trips":
                    self._send(200, json.dumps(fake._trip_list()).encode(), "application/json")
                elif path.startswith("/trip/") and number.isdigit() and (trip := fake._trip(int(number))):
                    self._send(200, json.dumps(trip).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def _send(self, code: int, body: bytes, kind: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        return Handler


class FakeVerifier:
    """Scripted verifier: `script` maps stage name -> list of Verdict | None | bool, consumed one per check().

    A bare bool becomes Verdict(bool, 0.9, "scripted"); None (or an exhausted list) is "can't tell yet".
    `.calls` records `(stage_name, elapsed_s, sorted frame roles)` for every check.
    """

    def __init__(self, script: dict[str, list[Any]] | None = None) -> None:
        self.script = {name: list(items) for name, items in (script or {}).items()}
        self.calls: list[tuple[str, float, list[str]]] = []
        self._lock = threading.Lock()

    def check(self, stage: Any, frames: dict[str, bytes], elapsed_s: float) -> Verdict:
        with self._lock:
            self.calls.append((stage.name, elapsed_s, sorted(frames)))
            queue = self.script.get(stage.name)
            item = queue.pop(0) if queue else None
        if isinstance(item, bool):
            return Verdict(item, 0.9, "scripted")
        if item is None:
            return Verdict(None, 0.0, "scripted: can't tell yet")
        return item

    def checks_for(self, name: str) -> int:
        return sum(1 for call in self.calls if call[0] == name)


def main(argv: list[str]) -> None:
    scenario = argv[1] if len(argv) > 1 else "happy"
    fake = FakeStation(8091, 8093, scenario=scenario).start()
    print(f"fake station ({scenario}): api {fake.api_url}  view {fake.view_url}  (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fake.close()


if __name__ == "__main__":
    main(sys.argv)
