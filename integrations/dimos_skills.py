"""The harness as DimOS skills (Dimensional's robot OS, github.com/dimensionalOS/dimos).

A DimOS agent (McpClient), or any MCP client through McpServer, can then compose a Hill Climb run from the
harness's primitives: list_stages, start_stage, check_stage, recover, stop_robot, status, run_plan, plan_status
(`stop_robot`, not `stop`: Module.stop is the module's own lifecycle call).
They all go through Dream's station HTTP API (integrations.station_tools), so Dream-1 still drives the arms and
Dream's safety loop stays in charge. Nothing here talks to a motor, and DimOS's own arm drivers are not used.

Run (Linux/macOS or WSL, with dimos installed; ANTHROPIC_API_KEY for the monitor checker):

    python -m integrations.dimos_skills plans/hill_climb_v4.yaml [--var color=orange] [--mcp-port 9990]

then e.g. `dimos mcp list-tools` / `dimos mcp call start_stage --name unplug_cable`, or an McpClient agent.
"""

from __future__ import annotations

import argparse
import sys

from dimos.agents.annotation import skill
from dimos.core.module import Module, ModuleConfig

from integrations.station_tools import StationTools


class HillClimbSkillsConfig(ModuleConfig):
    plan: str = "plans/hill_climb_v4.yaml"
    station_url: str = "http://127.0.0.1:8091"
    view_url: str = "http://127.0.0.1:8093"
    variables: dict[str, str] = {}
    allow_free_prompts: bool = False
    runs_dir: str = "runs"


class HillClimbSkills(Module):
    """Hill Climb harness primitives over Dream's station agent."""

    config: HillClimbSkillsConfig

    @property
    def _hc(self) -> StationTools:
        """Built on first use, inside the worker process that hosts this module."""
        tools = self.__dict__.get("_hc_tools")
        if tools is None:
            c = self.config
            tools = StationTools(c.plan, station_url=c.station_url, view_url=c.view_url,
                                 variables=dict(c.variables), runs_dir=c.runs_dir,
                                 allow_free_prompts=c.allow_free_prompts)
            self.__dict__["_hc_tools"] = tools
        return tools

    @skill
    def status(self) -> str:
        """Dream station state: run state, current prompt, chunk/stall/clamp counts, arm health."""
        return self._hc.status()

    @skill
    def list_stages(self) -> str:
        """The plan's stages in order, each with its prompt, success question, timeout and recovery menu."""
        return self._hc.list_stages()

    @skill
    def start_stage(self, name: str) -> str:
        """Give Dream-1 the prompt of plan stage `name` (starts a run if none is live, else changes it live)."""
        return self._hc.start_stage(name)

    @skill
    def check_stage(self, name: str) -> str:
        """Judge stage `name` from the newest camera frames: done, progress, status, suggested recovery."""
        return self._hc.check_stage(name)

    @skill
    def recover(self, stage_name: str, recovery_name: str) -> str:
        """Send one recovery from that stage's menu (names come from list_stages)."""
        return self._hc.recover(stage_name, recovery_name)

    @skill
    def send_prompt(self, prompt: str) -> str:
        """Free-text prompt to Dream-1, only if the operator enabled it; prefer start_stage / recover."""
        return self._hc.send_prompt(prompt)

    @skill
    def stop_robot(self) -> str:
        """Stop Dream-1 now; the arms hold where they are."""
        return self._hc.stop()

    @skill
    def run_plan(self) -> str:
        """Run the whole plan with the harness's own checker and recoveries, in the background."""
        return self._hc.run_plan()

    @skill
    def plan_status(self) -> str:
        """Whether the background plan is running, or how it ended."""
        return self._hc.plan_status()


def main(argv: list[str] | None = None) -> int:
    from dimos.agents.mcp.mcp_server import McpServer
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.core.coordination.module_coordinator import ModuleCoordinator

    p = argparse.ArgumentParser(prog="python -m integrations.dimos_skills")
    p.add_argument("plan")
    p.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--station", default="http://127.0.0.1:8091")
    p.add_argument("--view", default="http://127.0.0.1:8093")
    p.add_argument("--allow-free-prompts", action="store_true")
    a = p.parse_args(argv)
    variables = dict(v.split("=", 1) for v in a.var)
    bp = autoconnect(
        HillClimbSkills.blueprint(plan=a.plan, station_url=a.station, view_url=a.view, variables=variables,
                                  allow_free_prompts=a.allow_free_prompts),
        McpServer.blueprint(),
    )
    print("Hill Climb skills up over MCP; Ctrl+C stops (and the skills never home or move a joint).", file=sys.stderr)
    ModuleCoordinator.build(bp).loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
