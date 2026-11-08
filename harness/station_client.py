"""HTTP client for Dream's station agent (`dreamcli robot up`), urllib only.

It is the only module that sends anything that can move an arm, and it does so only by asking
the station agent, which owns every safety limit. With dry_run=True no POST is ever sent.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


class StationError(Exception):
    """A non-2xx reply, or no reply at all (status_code 0)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"station error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class StationClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8091",
        *,
        timeout_s: float = 70.0,
        dry_run: bool = False,
        log: Callable[[str], None] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.dry_run = dry_run
        self._log = log
        self.last_status: dict[str, Any] | None = None
        # Dry run only: the state the station would be in had the POSTs been sent.
        self._sim_state: str | None = None
        self._sim_prompt: str | None = None

    # -- helpers -------------------------------------------------------------------------------
    def _say(self, message: str) -> None:
        if self._log is not None:
            self._log(message)

    def _request(self, method: str, route: str, body: dict | None = None) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if method == "POST":
            data = json.dumps(body or {}).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + route, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            raise StationError(exc.code, _error_message(raw, exc.reason)) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise StationError(0, f"cannot reach {self.base_url}{route}: {reason}") from None
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise StationError(0, f"bad JSON from {route}: {exc.msg}") from None
        if not isinstance(payload, dict):
            raise StationError(0, f"unexpected reply from {route}")
        return payload

    def _post(self, route: str, body: dict) -> dict[str, Any]:
        if self.dry_run:
            self._say(f"DRY RUN: would POST {route} {json.dumps(body)}")
            return self._simulate(route, body)
        return self._request("POST", route, body)

    def _simulate(self, route: str, body: dict) -> dict[str, Any]:
        if route == "/start":
            self._sim_state, self._sim_prompt = "running", body.get("prompt")
        elif route == "/prompt":
            self._sim_state, self._sim_prompt = "running", body.get("prompt")
        elif route in ("/stop", "/home", "/reset"):
            self._sim_state = "idle"
        return self._overlay(dict(self.last_status or {"state": "running", "arms": {}, "cameras": {}}))

    def _overlay(self, status: dict[str, Any]) -> dict[str, Any]:
        if self._sim_state is not None:
            status["state"] = self._sim_state
            if self._sim_prompt is not None:
                status["prompt"] = self._sim_prompt
        return status

    # -- API -----------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """GET /status. In dry run an unreachable station yields a synthetic status flagged
        "_synthetic": True, and the state reflects the simulated POSTs."""
        try:
            status = self._request("GET", "/status")
        except StationError:
            if not self.dry_run:
                raise
            return self._overlay({"state": "running", "arms": {}, "cameras": {}, "_synthetic": True})
        self.last_status = status
        return self._overlay(dict(status)) if self.dry_run else status

    def start(self, prompt: str, checkpoint: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"prompt": prompt}
        if checkpoint:
            body["checkpoint"] = checkpoint
        return self._post("/start", body)

    def set_prompt(self, prompt: str) -> dict[str, Any]:
        return self._post("/prompt", {"prompt": prompt})

    def stop(self) -> dict[str, Any]:
        return self._post("/stop", {})

    def home(self) -> dict[str, Any]:
        return self._post("/home", {})

    def reset(self) -> dict[str, Any]:
        return self._post("/reset", {})

    def grippers(self) -> dict[str, Any]:
        """GET /arms (station_addons/sitecustomize.py): each gripper's measured opening (0 closed .. 1 open), the
        commanded opening and the effort. {} when the station runs without the add-on, so nothing depends on it."""
        try:
            return self._request("GET", "/arms")
        except Exception:  # noqa: BLE001 - optional route
            return {}

    def wait_for_state(self, states: set[str], timeout_s: float, poll_s: float = 0.5) -> dict[str, Any]:
        """Polls /status until its state is in `states`. Raises TimeoutError."""
        deadline = time.monotonic() + timeout_s
        last: dict[str, Any] = {}
        while True:
            last = self.status()
            if last.get("state") in states:
                return last
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"station state {last.get('state')!r} did not reach {sorted(states)} within {timeout_s:g} s"
                )
            time.sleep(poll_s)


def _error_message(raw: bytes, fallback: Any) -> str:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "error" in payload:
            return str(payload["error"])
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return str(fallback)
