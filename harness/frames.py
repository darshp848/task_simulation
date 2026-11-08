"""Reads the newest camera frames from the station's debug view (`--view-port 8093`)."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any


class FrameSource:
    def __init__(self, base_url: str = "http://127.0.0.1:8093", *, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.last_error: str | None = None
        self._cache_n: int | None = None
        self._cache: dict[str, bytes] = {}

    def _get_json(self, route: str) -> Any:
        with urllib.request.urlopen(self.base_url + route, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def latest_n(self) -> int | None:
        """Number of the newest round trip, or None."""
        try:
            trips = self._get_json("/trips")
            self.last_error = None
            if isinstance(trips, list) and trips:
                return int(trips[-1]["n"])
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return None

    def latest(self) -> dict[str, bytes]:
        """role -> JPEG bytes: the newest frame of each camera in the newest round trip. {} if none."""
        try:
            trips = self._get_json("/trips")
            if not isinstance(trips, list) or not trips:
                self.last_error = None
                return {}
            # Newest trip first; fall back to a couple of older ones if the newest has no frames.
            for entry in reversed(trips[-3:]):
                n = int(entry["n"])
                if n == self._cache_n and self._cache:
                    self.last_error = None
                    return dict(self._cache)
                try:
                    trip = self._get_json(f"/trip/{n}")
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        continue
                    raise
                out = _decode_frames(trip.get("frames") or {})
                if out:
                    self._cache_n, self._cache = n, out
                    self.last_error = None
                    return dict(out)
            self.last_error = None
            return {}
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {}


def _decode_frames(frames: dict[str, Any]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for key, images in frames.items():
        if not isinstance(images, list) or not images or not isinstance(images[-1], str):
            continue
        role = key.removeprefix("video.")
        try:
            out[role] = base64.b64decode(images[-1])
        except ValueError:
            continue
    return out
