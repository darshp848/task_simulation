"""Run log: runs/<UTC timestamp>/events.jsonl, saved JPEGs, summary.json."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLog:
    def __init__(self, dir: str | Path = "runs"):
        base = Path(dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = base / stamp
        n = 1
        while path.exists():
            n += 1
            path = base / f"{stamp}-{n}"
        path.mkdir(parents=True)
        self.path = path
        self._events = path / "events.jsonl"
        self._segments = path / "segments.jsonl"
        self._manifest: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._seq = 0

    def event(self, event: str, **fields: Any) -> None:
        record = {"t": round(time.time(), 3), "event": event, **fields}
        line = json.dumps(record, default=str)
        with self._lock:
            with self._events.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def segment(self, **fields: Any) -> None:
        """One line of segments.jsonl: a stage attempt or a recovery (see DESIGN_v2.md section 7)."""
        line = json.dumps(fields, default=str)
        with self._lock:
            with self._segments.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def update_manifest(self, **fields: Any) -> Path:
        """Merges fields into manifest.json (rewritten whole each time)."""
        with self._lock:
            self._manifest.update(fields)
            p = self.path / "manifest.json"
            p.write_text(json.dumps(self._manifest, indent=2, default=str), encoding="utf-8")
        return p

    def save_frames(self, tag: str, frames: dict[str, bytes]) -> list[Path]:
        with self._lock:
            self._seq += 1
            seq = self._seq
        paths = []
        for role, jpeg in frames.items():
            safe_role = "".join(c if c.isalnum() or c in "-_" else "_" for c in role)
            safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
            p = self.path / f"{seq:04d}_{safe_tag}_{safe_role}.jpg"
            p.write_bytes(jpeg)
            paths.append(p)
        return paths

    def write_summary(self, summary: dict[str, Any]) -> Path:
        p = self.path / "summary.json"
        p.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return p
