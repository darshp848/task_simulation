"""Offline stress test of the harness: whole matches on a virtual clock, with no station, no Dream and no API key.

The real Runner and the real MonitorVerifier run unchanged. Around them:
  - VClock      virtual time; Runner's sleeps advance it, so a 5-minute match takes about a second.
  - Table       a rule-based model of the relay table and of the policy's reaction to each prompt (T1-T6 texts).
  - SimStation  the station API (status/start/set_prompt/stop/...) with injectable faults taken from the NUC logs
                (DA-4000 refusals, ControlLoopError buffer starvation, LinAlgError, gateway DA-5000, arm loss).
  - SimFrames   camera frames that carry their capture time, so frozen or stale cameras can be modelled.
  - Scripted    the monitor's transport: answers each stage question from the table's state AT THE FRAME TIME,
                after a virtual latency, with optional false "done" answers, API outages and corrupt-image 400s.
  - referee()   scores the match by rubric v4 (DQ, black ball, points) from the table's event log.

    python experiments/stress_match.py                 # every scenario, one line each
    python experiments/stress_match.py -s no_release -v
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import tempfile
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.log import RunLog  # noqa: E402
from harness.plan import Plan, load_plan  # noqa: E402
from harness.runner import Runner  # noqa: E402
from harness.verifiers import MonitorVerifier  # noqa: E402

PLAN = Path(__file__).resolve().parents[1] / "plans" / "hill_climb_v4.yaml"
STEP = 0.25


# ---------------------------------------------------------------------------------------------------- clock
class VClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.lock = threading.RLock()
        self.tick_hooks: list[Any] = []
        self.monitor: MonitorVerifier | None = None
        self.transport: Scripted | None = None
        self.last_real_advance = time.monotonic()

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        with self.lock:
            end = self.now + dt
            while self.now < end - 1e-9:
                step = min(STEP, end - self.now)
                self.now += step
                for hook in self.tick_hooks:
                    hook(self.now, step)
            self.last_real_advance = time.monotonic()

    def sleep(self, dt: float) -> None:
        self.advance(dt)
        self.settle()

    def settle(self) -> None:
        """Let a monitor call that is due by now finish before the runner looks again (deterministic order)."""
        tr, mon = self.transport, self.monitor
        if tr is None or mon is None:
            return
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            due = tr.due
            if due is None or due > self.now + 1e-9:
                if not (due is None and mon._busy):
                    return
            time.sleep(0.0005)


# ---------------------------------------------------------------------------------------------------- table
@dataclass
class Behaviour:
    """How the policy does on this table. Times are virtual seconds per try."""
    unplug_s: float = 14.0
    unplug_p: float = 0.8
    putdown_s: float = 6.0
    releases: bool = True            # False: the policy carries the cable but never opens on it (sim finding)
    open_frees: bool = True          # False: even "open both grippers" leaves the plug stuck in the gripper
    plug_s: float = 16.0
    plug_p: float = 0.4
    block_s: float = 11.0
    block_p: float = 0.7
    knock_p: float = 0.05
    block_in: float = 1.5            # block height, inches
    ball_s: float = 8.0
    ball_p: float = 0.7
    black_p: float = 0.0             # chance a ball try picks a black 8-ball
    reach_across_p: float = 0.0      # chance an unplug try brushes a block (the hub-left DQ)
    unplug_never: bool = False


@dataclass
class TableState:
    hub: int = 4                 # cables that started in the hub and are still in
    out: int = 0                 # cables unplugged from the hub
    loose: int = 4               # loose cables lying on the table
    plugged: int = 0             # loose cables plugged into free ports
    replugged: int = 0           # unplugged cables pushed back in (not allowed)
    held: str | None = None      # "cable_out" (an unplugged one), "cable_loose", "block", "ball", "black"
    arms_up: bool = True
    tower: int = 1
    max_tower: int = 1
    balls: int = 0
    black: int = 0
    phase: float = 0.0           # how far into its current try the policy is (0-1), for the monitor's progress


class Table:
    def __init__(self, beh: Behaviour, seed: int) -> None:
        self.b, self.rng = beh, random.Random(seed)
        self.s = TableState()
        self.history: list[tuple[float, TableState]] = [(0.0, copy.copy(self.s))]
        self.events: list[tuple[float, str]] = []
        self.prompt: str | None = None
        self.running = False
        self.t_task = 0.0

    def log(self, t: float, what: str) -> None:
        self.events.append((round(t, 2), what))

    def snapshot_at(self, t: float) -> TableState:
        best = self.history[0][1]
        for ts, st in self.history:
            if ts <= t + 1e-9:
                best = st
            else:
                break
        return best

    def set_prompt(self, t: float, prompt: str | None, *, restart: bool = False) -> None:
        if prompt != self.prompt or restart:   # the same text again does not undo what the arm is doing
            self.t_task = 0.0
        self.prompt = prompt
        if prompt is not None:
            self.log(t, f"prompt: {prompt}")

    def passed_stage1(self) -> bool:
        return self.s.out >= 1

    def tower_passed(self) -> bool:
        return self.s.max_tower * self.b.block_in > 4.0

    def touch(self, t: float, obj: str) -> None:
        if obj == "block" and not self.passed_stage1():
            self.log(t, "DQ: touched a block before a cable was out")
        if obj == "ball" and not self.tower_passed():
            self.log(t, "DQ: touched a golf ball before the tower passed")

    def tick(self, t: float, dt: float) -> None:
        if self.running and self.prompt:
            before = self.t_task
            self.t_task += dt
            self._act(t, before, self.t_task)
            p, b = self.prompt, self.b
            span = (b.unplug_s if p.startswith("unplug") else b.block_s if p.startswith("put a block")
                    else b.ball_s if p.startswith("put a golf") else b.plug_s if p.startswith("plug") else 6.0)
            self.s.phase = min(0.95, (self.t_task % span) / span)
        self.history.append((t, copy.copy(self.s)))

    def _crossed(self, before: float, now: float, at: float) -> bool:
        return before < at <= now

    def _act(self, t: float, t0: float, t1: float) -> None:
        p, s, b, r = self.prompt, self.s, self.b, self.rng
        if p.startswith("unplug"):
            period = b.unplug_s
            k0, k1 = int(t0 // period), int(t1 // period)
            h0, h1, half = t0 % period, t1 % period, period * 0.5
            if ((k1 == k0 and h0 < half <= h1) or (k1 > k0 and (h0 < half or h1 >= half)))                     and b.reach_across_p and r.random() < b.reach_across_p:
                self.touch(t, "block")
            if k1 > k0:
                if s.held is None and s.hub > 0 and not b.unplug_never and r.random() < b.unplug_p:
                    s.hub -= 1; s.out += 1; s.held = "cable_out"; s.arms_up = False
                    self.log(t, f"cable unplugged ({s.out} out)")
        elif p.startswith("put the held cable down"):
            if s.held and s.held.startswith("cable") and t1 >= b.putdown_s > t0 and b.releases:
                s.held = None; self.log(t, "cable put down")
        elif p.startswith("open both grippers"):
            if t1 >= 2.0 > t0:
                if s.held and not (s.held.startswith("cable") and not b.open_frees):
                    if s.held == "black":
                        self.log(t, "black ball dropped on the table")
                    s.held = None
                s.arms_up = True
        elif p.startswith("plug the loose"):
            if t1 >= b.plug_s > t0:
                if s.held == "cable_out":
                    s.held = None; s.hub += 1; s.replugged += 1
                    self.log(t, "RULE: an unplugged cable was plugged back in")
                elif s.held is None and s.loose > 0 and r.random() < b.plug_p:
                    s.loose -= 1; s.plugged += 1; self.log(t, "loose cable plugged in")
                self.t_task = 0.0
        elif p.startswith("put a block"):
            if t1 >= 1.0 > t0:
                self.touch(t, "block")
            if t1 >= b.block_s > t0:
                if s.held is None and r.random() < b.block_p:
                    s.tower += 1; s.max_tower = max(s.max_tower, s.tower)
                    self.log(t, f"tower {s.tower} blocks")
                elif r.random() < b.knock_p:
                    s.tower = 1; self.log(t, "tower knocked down")
                self.t_task = 0.0
        elif p.startswith("put a golf ball"):
            if t1 >= 1.0 > t0:
                self.touch(t, "ball")
            if t1 >= b.ball_s * 0.5 > t0 and s.held is None:
                s.held = "black" if r.random() < b.black_p else "ball"
                if s.held == "black":
                    self.log(t, "black ball picked up")
            if t1 >= b.ball_s > t0:
                if s.held == "black":
                    s.black += 1; s.held = None; self.log(t, "BLACK BALL IN THE BIN")
                elif s.held == "ball":
                    s.held = None
                    if r.random() < b.ball_p:
                        s.balls += 1; self.log(t, f"ball in bin ({s.balls})")
                self.t_task = 0.0


# ---------------------------------------------------------------------------------------------------- station
@dataclass
class Fault:
    at: float                      # virtual time (s) the fault fires, while running
    kind: str                      # refusal | starve | linalg | gateway | arm
    once: bool = True
    prompt_has: str | None = None  # fire only while the prompt contains this


ERRORS = {
    "refusal": {"type": "DreamBadRequest", "error_code": "DA-4000", "message": "DreamAPI refused the inference "
                "request: DA-4000 BAD_REQUEST (malformed): unsafe instruction in this scene, refused: unplug cables"},
    "starve": {"type": "ControlLoopError", "error_code": None, "message": "action buffer empty for 241 consecutive "
               "ticks (12.05 s), more than max_stall_ticks=240"},
    "linalg": {"type": "LinAlgError", "error_code": None, "message": "SVD did not converge"},
    "gateway": {"type": "DreamServerError", "error_code": "DA-5000", "message": "DreamAPI gateway timeout"},
    "arm": {"type": "ArmError", "error_code": None, "message": "arm 'right' driver stopped answering"},
}


class SimStation:
    dry_run = False

    def __init__(self, clock: VClock, table: Table, faults: list[Fault], *, refuse: dict[str, int] | None = None,
                 start_delay_s: float = 2.0) -> None:
        self.clock, self.table, self.faults = clock, table, list(faults)
        self.refuse = dict(refuse or {})   # prompt substring -> how many starts get refused
        self.state, self.last_error, self.prompt = "idle", None, None
        self.start_delay_s = start_delay_s
        self._start_at: float | None = None
        self.posts: list[tuple[float, str, str | None]] = []
        self.arms = {"left": True, "right": True}
        self.chunks = 0
        clock.tick_hooks.insert(0, self._tick)

    def _tick(self, t: float, dt: float) -> None:
        if self.state == "starting" and self._start_at is not None and t >= self._start_at:
            for sub, n in list(self.refuse.items()):
                if n > 0 and sub in (self.prompt or ""):
                    self.refuse[sub] = n - 1
                    self._error("refusal"); return
            self.state = "running"; self.table.running = True
        if self.state == "running":
            self.chunks += 1 if int(t / 3.4) != int((t - dt) / 3.4) else 0
            for f in list(self.faults):
                if t >= f.at and (f.prompt_has is None or f.prompt_has in (self.prompt or "")):
                    if f.once:
                        self.faults.remove(f)
                    self._error(f.kind); return

    def _error(self, kind: str) -> None:
        self.state, self.last_error = "error", dict(ERRORS[kind])
        self.table.running = False
        self.table.log(self.clock.now, f"station error: {kind}")
        if kind == "arm":
            self.arms["right"] = False

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "last_error": self.last_error, "arms": dict(self.arms), "chunks": self.chunks,
                "cameras": {"scene": 0.03, "wrist_l": 0.03, "wrist_r": 0.03}, "prompt": self.prompt}

    def start(self, prompt: str, checkpoint: str | None = None) -> dict[str, Any]:
        self.posts.append((self.clock.now, "/start", prompt))
        if not all(self.arms.values()):
            from harness.station_client import StationError
            raise StationError(409, "cannot start: arm 'right' is not alive")
        self.state, self.last_error, self.prompt = "starting", None, prompt
        self._start_at = self.clock.now + self.start_delay_s
        self.table.running = False
        self.table.set_prompt(self.clock.now, prompt, restart=True)
        return self.status()

    def set_prompt(self, prompt: str) -> dict[str, Any]:
        self.posts.append((self.clock.now, "/prompt", prompt))
        self.prompt = prompt
        for sub, n in list(self.refuse.items()):   # Dream checks every new instruction, not only at /start
            if n > 0 and sub in prompt and self.state == "running":
                self.refuse[sub] = n - 1
                self.table.set_prompt(self.clock.now, prompt)
                self._error("refusal")
                return self.status()
        self.table.set_prompt(self.clock.now, prompt)
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.posts.append((self.clock.now, "/stop", None))
        self.state, self.table.running = "idle", False
        self.table.log(self.clock.now, "stop")
        return self.status()

    def home(self) -> dict[str, Any]:
        self.posts.append((self.clock.now, "/home", None))
        return self.status()

    def grippers(self) -> dict[str, Any]:
        return {}

    def wait_for_state(self, states: set[str], timeout_s: float, poll_s: float = 0.5) -> dict[str, Any]:
        end = self.clock.now + timeout_s
        while True:
            st = self.status()
            if st["state"] in states:
                return st
            if self.clock.now >= end:
                raise TimeoutError(f"station state {st['state']!r} did not reach {sorted(states)} within {timeout_s:g} s")
            self.clock.sleep(poll_s)


# ---------------------------------------------------------------------------------------------------- frames
GOOD, CORRUPT = b"\xff\xd8\xff\xe0", b"\x00\x00not-a-jpeg"


class SimFrames:
    """latest(): role -> fake JPEG whose bytes carry the capture time. Faults: frames freeze (the debug view keeps
    serving the last trip), go empty, or one camera serves a corrupt image."""

    def __init__(self, clock: VClock) -> None:
        self.clock = clock
        self.freeze: list[tuple[float, float]] = []   # (from, to): frames stop updating
        self.empty: list[tuple[float, float]] = []
        self.corrupt: list[tuple[float, float]] = []
        self._last: dict[str, bytes] = {}
        self.lag_s = 0.0

    @staticmethod
    def _in(t: float, spans: list[tuple[float, float]]) -> bool:
        return any(a <= t < b for a, b in spans)

    def latest(self) -> dict[str, bytes]:
        t = self.clock.now
        if self._in(t, self.empty):
            return {}
        if self._in(t, self.freeze) and self._last:
            return dict(self._last)
        stamp = f"t={max(0.0, t - self.lag_s):.2f}".encode()
        body = (CORRUPT if self._in(t, self.corrupt) else GOOD) + stamp + b"\xff\xd9"
        self._last = {"scene": body, "wrist_l": GOOD + stamp + b"\xff\xd9", "wrist_r": GOOD + stamp + b"\xff\xd9"}
        return dict(self._last)


def frame_time(jpeg: bytes) -> float | None:
    i = jpeg.find(b"t=")
    try:
        return float(jpeg[i + 2:jpeg.find(b"\xff\xd9", i)].decode()) if i >= 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------------- monitor
class Scripted:
    """The monitor's transport. Reads the stage from the prompt text and the table state at the newest frame time."""

    def __init__(self, clock: VClock, table: Table, plan: Plan, *, latency_s: float = 4.0, seed: int = 0) -> None:
        self.clock, self.table, self.plan = clock, table, plan
        self.latency_s = latency_s
        self.rng = random.Random(seed)
        self.due: float | None = None
        self.outages: list[tuple[float, float]] = []
        self.false_done: dict[str, float] = {}   # stage name -> chance of a false stage_done
        self.guard_misses = 0.0                  # chance the guard answer is wrong when a black ball is held
        self.calls = 0
        self.base: dict[str, TableState] = {}
        self.stage_seen: str | None = None

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        t_req = self.clock.now
        self.calls += 1
        content = payload["messages"][0]["content"]
        text = content[-1]["text"]
        stage = next(line.split(".", 1)[1].split(":", 1)[0].strip() for line in text.splitlines()
                     if "<-- CURRENT STAGE" in line)
        images = [c["source"]["data"] for c in content if c.get("type") == "image"]
        import base64
        raw = [base64.b64decode(d) for d in images]
        newest = max((frame_time(j) or 0.0) for j in raw) if raw else t_req
        start = min((frame_time(j) or 0.0) for j in raw) if raw else t_req
        due = t_req + self.latency_s
        self.due = due
        try:
            if threading.current_thread() is threading.main_thread():
                self.clock.advance(self.latency_s)    # a synchronous call: the runner waits for it
            else:
                while self.clock.now < due - 1e-9:
                    time.sleep(0.0005)
                    if time.monotonic() - self.clock.last_real_advance > 0.2:   # runner blocked waiting on us
                        self.clock.advance(min(STEP, due - self.clock.now))
            if any(a <= t_req < b for a, b in self.outages):
                raise urllib.error.URLError("network unreachable (scripted outage)")
            if any(not j.startswith(GOOD) for j in raw):
                raise urllib.error.HTTPError("x", 400, "Could not process image", {}, None)  # type: ignore[arg-type]
            return {"content": [{"type": "text", "text": self._answer(stage, start, newest, "guard" in text)}]}
        finally:
            self.due = None

    def _answer(self, stage: str, t_start: float, t_new: float, guarded: bool) -> str:
        import json
        s = self.table.snapshot_at(t_new)
        s0 = self.table.snapshot_at(t_start)
        b = self.table.b
        tower_ok = s.tower * b.block_in > 4.0
        if stage.startswith(("unplug_cable",)):
            done = s.out >= 1
        elif stage == "unplug_more":
            done = s.out > s0.out
        elif stage in ("put_cable_down", "put_more_down"):
            done = s.held is None
        elif stage.startswith("plug_"):
            done = s.plugged > s0.plugged
        elif stage == "arms_clear":
            done = s.held is None and s.arms_up
        elif stage == "stack_2":
            done = s.tower >= 2
        elif stage == "stack_3":
            done = s.tower >= 3
        elif stage == "tower_pass":
            done = tower_ok
        elif stage.startswith("higher_"):
            done = s.tower > s0.tower
        elif stage.startswith("ball_"):
            done = s.balls >= int(stage.split("_")[1]) and s.black == 0
        else:
            done = False
        if not done and self.rng.random() < self.false_done.get(stage, 0.0):
            done = True
        held = {"left": "nothing", "right": {"cable_out": "the unplugged USB cable plug", "cable_loose": "a cable",
                                              "block": "a block", "ball": "a golf ball", "black": "a black 8-ball",
                                              None: "nothing"}[s.held]}
        obj: dict[str, Any] = {"stage_done": done, "progress": 1.0 if done else round(0.1 + 0.8 * s.phase, 2), "status": "ok", "holding": held,
                               "recovery": None, "scene_ok": s.held is None, "confidence": 0.9, "reason": "scripted"}
        if guarded:
            g = s.held == "black"
            if g and self.rng.random() < self.guard_misses:
                g = False
            obj["guard"] = g
        return "<json>" + json.dumps(obj) + "</json>"


# ---------------------------------------------------------------------------------------------------- referee
def referee(table: Table) -> dict[str, Any]:
    s, ev = table.s, [e for _, e in table.events]
    dq = [e for e in ev if e.startswith("DQ")]
    tower_in = s.tower * table.b.block_in
    passed2 = s.max_tower * table.b.block_in > 4.0
    pts = 0
    if s.out >= 1:
        pts += 10 + 5 * (s.out - 1)
    pts += 20 * min(s.plugged, 4)
    if passed2:
        pts += 20 + 3 * int(tower_in)
    pts += 5 * s.balls
    lost = bool(dq) or s.black > 0
    return {"points": 0 if dq else pts, "dq": dq[:1], "black_in_bin": s.black, "lost": lost,
            "replugged": s.replugged, "out": s.out, "plugged": s.plugged, "tower": s.tower, "balls": s.balls}


# ---------------------------------------------------------------------------------------------------- one match
@dataclass
class Scenario:
    name: str
    beh: Behaviour = field(default_factory=Behaviour)
    faults: list[Fault] = field(default_factory=list)
    refuse: dict[str, int] = field(default_factory=dict)
    outages: list[tuple[float, float]] = field(default_factory=list)
    freeze: list[tuple[float, float]] = field(default_factory=list)
    empty: list[tuple[float, float]] = field(default_factory=list)
    corrupt: list[tuple[float, float]] = field(default_factory=list)
    false_done: dict[str, float] = field(default_factory=dict)
    latency_s: float = 4.0
    fast_guard: bool = False        # a local black-ball detector on the wrist frames (harness.detect), every poll
    frame_lag_s: float = 0.0        # frames show the table this long ago (trip cadence ~3.4 s on the station)
    plan: Path = PLAN
    note: str = ""


def run_match(sc: Scenario, seed: int = 1, out_dir: Path | None = None) -> dict[str, Any]:
    plan = load_plan(sc.plan)
    clock = VClock()
    table = Table(sc.beh, seed)
    station = SimStation(clock, table, sc.faults, refuse=sc.refuse)
    clock.tick_hooks.append(table.tick)
    frames = SimFrames(clock)
    frames.freeze, frames.empty, frames.corrupt = sc.freeze, sc.empty, sc.corrupt
    frames.lag_s = sc.frame_lag_s
    tr = Scripted(clock, table, plan, latency_s=sc.latency_s, seed=seed)
    tr.outages, tr.false_done = sc.outages, sc.false_done
    mon = MonitorVerifier(plan, transport=tr)
    clock.monitor, clock.transport = mon, tr
    tmp = out_dir or Path(tempfile.mkdtemp(prefix="stress_"))
    log = RunLog(tmp)
    runner = Runner(plan, station, frames, {"monitor": mon}, log, clock=clock, sleep=clock.sleep)
    if sc.fast_guard:
        def local(fr: dict[str, bytes]) -> tuple[bool | None, str]:
            if "wrist_r" not in fr:
                return None, "no wrist view"
            return table.snapshot_at(frame_time(fr["wrist_r"]) or 0.0).held == "black", "sim detector"
        runner.fast_guard = local
    result = runner.run()
    ref = referee(table)
    return {"scenario": sc.name, "seed": seed, "harness_ok": result.ok, "reason": result.reason,
            "t_end": round(clock.now, 1), "stages": [(r.name, r.ok, r.attempts) for r in result.stages],
            "monitor_calls": tr.calls, **ref, "events": table.events, "log": str(log.path), "posts": station.posts}


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("happy"),
    Scenario("unplug_never", Behaviour(unplug_never=True), note="stage 1 never passes: blocks must stay untouched"),
    Scenario("slow_unplug", Behaviour(unplug_s=55.0, unplug_p=0.5), note="unplug needs > one 60 s timeout"),
    Scenario("no_release", Behaviour(releases=False), note="policy keeps holding the unplugged cable"),
    Scenario("stuck_cable", Behaviour(releases=False, open_frees=False), note="the plug will not leave the gripper"),
    Scenario("false_done_unplug", false_done={"unplug_cable": 0.35}, note="monitor calls unplug done too early"),
    Scenario("refusal_at_start", refuse={"unplug": 1}, note="DA-4000 on the first start (seen 4x on the NUC)"),
    Scenario("refusal_twice", refuse={"unplug": 2}, note="DA-4000 on two starts in a row"),
    Scenario("starve_mid", faults=[Fault(70, "starve")], note="action buffer empty 12 s (seen 2x)"),
    Scenario("linalg_mid", faults=[Fault(40, "linalg")], note="SVD did not converge (seen 3x)"),
    Scenario("gateway_x3", faults=[Fault(30, "gateway"), Fault(90, "gateway"), Fault(150, "gateway")]),
    Scenario("refusal_put_down", refuse={"put the held cable down": 3}, note="the put-down prompt is refused every time"),
    Scenario("arm_loss", faults=[Fault(100, "arm")], note="must stop, must not restart"),
    Scenario("api_outage_30s", outages=[(20, 50)], note="Claude API unreachable for 30 s during unplug"),
    Scenario("api_outage_balls", Behaviour(black_p=0.25), outages=[(0, 1e9)],
             note="monitor dead for the whole match; black balls on the table"),
    Scenario("black_ball", Behaviour(black_p=0.3, block_in=2.0), note="the guard must stop before a black ball lands"),
    Scenario("black_ball_slow_monitor", Behaviour(black_p=0.3, block_in=2.0), latency_s=8.0,
             note="monitor p99-ish latency"),
    Scenario("black_ball_lagged", Behaviour(black_p=0.3, block_in=2.0), frame_lag_s=2.0,
             note="as black_ball, frames 2 s old (trip cadence)"),
    Scenario("black_ball_local_guard", Behaviour(black_p=0.3, block_in=2.0), frame_lag_s=2.0, fast_guard=True,
             note="local detector on the wrist frames every poll, frames 2 s old"),
    Scenario("black_ball_local_guard_api_down", Behaviour(black_p=0.3, block_in=2.0), frame_lag_s=2.0,
             fast_guard=True, outages=[(150, 1e9)], note="local guard while the Claude API is down"),
    Scenario("frozen_cam", freeze=[(30, 90)], note="debug view serves the same trip for 60 s"),
    Scenario("empty_frames", empty=[(10, 60)], note="no frames for 50 s"),
    Scenario("corrupt_first_frame", corrupt=[(0, 12)], note="the stage-start image is not a JPEG"),
    Scenario("reach_across", Behaviour(reach_across_p=0.3), note="unplug tries brush the blocks (the hub-left DQ)"),
]}


def _one(job: tuple[str, int]) -> dict[str, Any]:
    name, seed = job
    r = run_match(SCENARIOS[name], seed)
    r.pop("posts", None)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--scenario", action="append", help="scenario name (repeatable); default all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=1, help="first seed (to split a sweep across machines)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="matches in parallel (processes)")
    ap.add_argument("--out", help="append one JSON line per match to this file")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.list:
        for n, sc in SCENARIOS.items():
            print(f"{n:32s} {sc.note}")
        return 0
    names = args.scenario or list(SCENARIOS)
    jobs = [(n, s) for n in names for s in range(args.seed0, args.seed0 + args.seeds)]
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            rows_all = pool.map(_one, jobs)
    else:
        rows_all = [_one(j) for j in jobs]
    if args.out:
        import json
        with open(args.out, "a", encoding="utf-8") as fh:
            for r in rows_all:
                fh.write(json.dumps(r, default=str) + "\n")
    for name in names:
        rows = [r for r in rows_all if r["scenario"] == name]
        pts = [r["points"] for r in rows]
        lost = sum(r["lost"] for r in rows)
        rep = sum(r["replugged"] > 0 for r in rows)
        early = sum(r["t_end"] < 280 for r in rows)
        print(f"{name:24s} pts={pts} lost={lost}/{len(rows)} replug={rep} ended_early={early} "
              f"| {rows[0]['reason'][:90]}")
        if args.verbose:
            for r in rows:
                print("   seed", r["seed"], "t_end", r["t_end"], r["stages"])
                for t, e in r["events"]:
                    if not e.startswith("prompt"):
                        print("     ", t, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
