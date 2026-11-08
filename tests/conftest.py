"""Puts the harness root (the folder holding harness/ and fake_station.py) on sys.path; shared fixtures."""

import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fake_station  # noqa: E402


@pytest.fixture
def fake(request):
    """A started FakeStation; parametrize indirectly with a dict of FakeStation kwargs."""
    kwargs = getattr(request, "param", {}) or {}
    station = fake_station.FakeStation(**kwargs).start()
    yield station
    station.close()


def make_station(**kwargs):
    return fake_station.FakeStation(**kwargs).start()


@pytest.fixture
def write_plan(tmp_path):
    """write_plan(stages, **top) -> path of a YAML plan with fast defaults."""

    def _write(stages, **top):
        data = {"task": "test task", "time_budget_s": 30, "check_every_s": 0.05, "verifier": "manual"}
        data.update(top)
        data["stages"] = stages
        path = tmp_path / "plan.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    return _write


def stage(name, **kw):
    base = {"name": name, "prompt": f"do {name}", "success": f"is {name} done?",
            "timeout_s": 2.0, "max_attempts": 2, "min_run_s": 0.05, "on_fail": "retry_in_place"}
    base.update(kw)
    return base


def wait_until(predicate, timeout=5.0, step=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return False
