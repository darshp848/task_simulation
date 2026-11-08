"""The DimOS adapter, built by a real DimOS coordinator against fake_station. Skipped where dimos is not installed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dimos")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = '''
import os, sys
sys.path.insert(0, os.environ["HC"])
from fake_station import FakeStation
from integrations.dimos_skills import HillClimbSkills
from dimos.core.coordination.module_coordinator import ModuleCoordinator
if __name__ == "__main__":
    fake = FakeStation(); fake.start()
    mc = ModuleCoordinator.build(HillClimbSkills.blueprint(plan=os.environ["HC"] + "/plans/cables_only.yaml",
                                                         station_url=fake.api_url, view_url=fake.view_url))
    try:
        mod = mc.get_instance(HillClimbSkills)
        print("STATUS", mod.status())
        print("START", mod.start_stage("unplug_cable"))
        print("STOP", mod.stop_robot())
        print("POSTS", [r for _, r, _ in fake.posts()])
    finally:
        mc.stop(); fake.close()
'''


def test_skills_through_a_dimos_coordinator(tmp_path):
    script = tmp_path / "run_dimos.py"   # DimOS workers re-import the main module, so it must be a real file
    script.write_text(SCRIPT)
    out = subprocess.run([sys.executable, str(script)], env={**os.environ, "HC": str(ROOT)}, cwd=tmp_path,
                         capture_output=True, text=True, timeout=180).stdout
    assert '"state": "idle"' in out
    assert "START stage unplug_cable prompt sent" in out
    assert "STOP stopped." in out
    assert "POSTS ['/start', '/stop']" in out
