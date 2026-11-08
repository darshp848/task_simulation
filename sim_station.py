"""Stand-in for Dream's station agent that drives Dream's own MuJoCo YAM simulator.

Serves the station API (default port 8091) and the debug view (default 8093) described in DESIGN.md, so the
harness can be tested end to end in simulation:

    python sim_station.py --policy dream            # the real Dream-1 model (needs the dreamlabs library + key)
    python sim_station.py --policy scripted         # offline: deterministic sinusoidal chunks, no key, no network
    python -m harness plans/sim_fruit_bowl.yaml --live --station http://127.0.0.1:8091 --view http://127.0.0.1:8093

The control loop is Dream's own (`dream_examples/yam_mujoco/dream_control.main`): the camera windows are filled by
holding the pose for FRAMES x FRAME_EVERY_STEPS steps, the session is opened with `open_policy`, the whole windows
are fed once, then each `get_action` is followed by `validate_chunk` and the rows are stepped at 20 Hz, feeding
the newest frame every FRAME_EVERY_STEPS steps. The simulator and policy code are imported, not copied: from the
installed `dream_examples.yam_mujoco`, else from `--example-dir` (or $DREAM_YAM_EXAMPLE_DIR).

Never touches hardware. In scripted mode it never touches the network either.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

IDLE, STARTING, RUNNING, STOPPING, HOMING, RESETTING, ERROR = (
    "idle", "starting", "running", "stopping", "homing", "resetting", "error")
MAX_BODY_BYTES = 64 * 1024
API_TIMEOUT_S = 60.0
ARMS = ("left", "right")
STATIC_BODIES = ("bowl", "plate", "hub", "bin")  # fixed scene parts worth reporting beside the free objects
SCENES = ("fruit", "hill_climb")


class _Refused(Exception):
    """A refused transition: HTTP 409."""


class _Timeout(Exception):
    """The station did not finish in time: HTTP 504."""


class _Abort(Exception):
    """The run was told to stop (internal)."""


# ---------------------------------------------------------------------------------------------------------- imports
def load_example(example_dir: str | None):
    """Import (yam_env, dream_control) from the example dir, the installed package, or `example_dir`."""
    candidates: list[Path] = []
    if example_dir:
        candidates.append(Path(example_dir))
    if os.environ.get("DREAM_YAM_EXAMPLE_DIR"):
        candidates.append(Path(os.environ["DREAM_YAM_EXAMPLE_DIR"]))
    if not example_dir:
        try:
            spec = importlib.util.find_spec("dream_examples.yam_mujoco")
        except Exception:
            spec = None
        if spec and spec.submodule_search_locations:
            candidates.append(Path(list(spec.submodule_search_locations)[0]))
    for path in candidates:
        if (path / "yam_env.py").is_file() and (path / "dream_control.py").is_file():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            import yam_env  # type: ignore
            import dream_control  # type: ignore

            return yam_env, dream_control
    raise SystemExit(
        "Cannot find Dream's yam_mujoco example. Install the dreamlabs library (pip install dreamlabs) so "
        "dream_examples/yam_mujoco is under site-packages, or pass --example-dir PATH.")


# ---------------------------------------------------------------------------------------------------------- scripted policy
class ScriptedPolicy:
    """Deterministic stand-in for the dream client: small sinusoidal joint chunks of the right shape.

    Same interface as the dream client's policy: set_observation(frames, state, timestamp=), get_action(), close().
    Joint targets oscillate around the pose the run started from; the gripper opens and closes slowly.
    """

    def __init__(self, anchor: dict[str, Any], latency_s: float = 0.05) -> None:
        import numpy as np

        self.np = np
        self.anchor = {side: np.array(anchor[side], dtype=float) for side in ARMS}
        self.latency_s = latency_s
        self.phase = 0  # rows handed out so far, so consecutive chunks join up
        self.observations = 0
        self.last_prompt: str | None = None
        self.closed = False

    def set_observation(self, frames: dict, state: dict, timestamp: float = 0.0) -> None:
        self.observations += 1
        self.last_prompt = state.get("prompt")

    def get_action(self) -> dict:
        np = self.np
        time.sleep(self.latency_s)
        rows = np.arange(self.phase, self.phase + 48)
        self.phase += 48
        t = rows / 20.0
        out = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            chunk = np.tile(self.anchor[side], (48, 1))
            for joint in range(6):
                amplitude = 0.08 + 0.01 * joint
                chunk[:, joint] += sign * amplitude * np.sin(2 * np.pi * (0.15 + 0.03 * joint) * t)
            chunk[:, 6] = 0.5 + 0.4 * np.sin(2 * np.pi * 0.1 * t + (0.0 if side == "left" else 1.0))
            out[f"q_{side[0]}_chunk"] = chunk.tolist()
        return out

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------------------------------------- station
class SimStation:
    """The station agent's state machine and HTTP servers around one YamEnv owned by one sim thread."""

    def __init__(self, *, policy: str = "scripted", checkpoint: str | None = None, api_port: int = 8091,
                 view_port: int = 8093, realtime: bool | None = None, example_dir: str | None = None,
                 actions: int = 48, host: str = "127.0.0.1", scripted_latency_s: float = 0.05,
                 max_trips: int = 50, scene: str = "fruit", viewer: bool = False) -> None:
        if policy not in ("dream", "scripted"):
            raise ValueError("policy must be 'dream' or 'scripted'")
        if scene not in SCENES:
            raise ValueError(f"scene must be one of {SCENES}")
        self.scene = scene
        self.policy_kind = policy
        self.realtime = (policy == "dream") if realtime is None else realtime
        self.actions = actions
        self.scripted_latency_s = scripted_latency_s
        self._api_port, self._view_port, self._host = api_port, view_port, host
        self.yam_env, self.dc = load_example(example_dir)
        self._referee_cls = None
        if scene == "hill_climb":
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from sim_scenes import hill_climb_table

            hill_climb_table.patch(self.yam_env)
            self._referee_cls = hill_climb_table.HillClimbReferee
        self._referee = None
        import numpy as np

        self.np = np
        self.default_checkpoint = (self.dc.resolve_model(checkpoint) if checkpoint else
                                   (self.dc.BETA_MODEL if policy == "dream" else "scripted"))
        self.key = ""
        self.gateway = "https://api.dreamlabs.ai"
        if policy == "dream":
            config = self.dc.settings()
            self.key = config.get("DREAM_API_KEY", "")
            self.gateway = config.get("DREAM_GATEWAY_URL", self.gateway)
            url = urlsplit(self.gateway)
            if url.scheme != "https" or not url.hostname:
                raise SystemExit("DREAM_GATEWAY_URL must be an HTTPS origin.")
            if not self.key:
                raise SystemExit("No Dream API key: save it in ~/.dreamlabs/credentials, set DREAM_API_KEY, "
                                 "or use --policy scripted.")

        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self.state = IDLE
        self.prompt: str | None = None
        self.checkpoint: str | None = self.default_checkpoint
        self.run_number = 0
        self.chunks = 0
        self.stalls = 0
        self.clamps = 0
        self.last_error: dict | None = None
        self._run_started: float | None = None
        self._last_render = 0.0
        self._start_req: dict | None = None
        self._stop_req = False
        self._cmd: tuple[str, threading.Event, dict] | None = None  # ("home"|"reset", done, result)
        self._trips: deque[dict] = deque(maxlen=max_trips)
        self._trip_n = 0
        self.calls: list[tuple[str, str, Any]] = []
        self.env: Any = None
        self.hold: dict[str, Any] | None = None  # the targets held while idle
        self._poses: dict[str, Any] = {}  # sim-only ground truth, refreshed after every physics step
        self._servers: list[ThreadingHTTPServer] = []
        self._threads: list[threading.Thread] = []
        self.api_url = ""
        self.view_url = ""
        # live visualizers: the latest camera renders (GET /live on the view port) and the native MuJoCo viewer
        self._images: dict[str, Any] = {}
        self._images_n = 0
        self._jpeg_cache: dict[str, tuple[int, bytes]] = {}
        self.viewer_wanted = viewer
        self._viewer: Any = None
        self._viewer_panel: Any = None
        self._viewer_sync = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "SimStation":
        ready = threading.Event()
        sim = threading.Thread(target=self._sim_main, args=(ready,), daemon=True, name="sim")
        sim.start()
        self._threads.append(sim)
        if not ready.wait(300):
            raise RuntimeError("the simulator did not come up")
        if self.env is None:
            raise RuntimeError("the simulator failed to start (see the error above)")
        api = ThreadingHTTPServer((self._host, self._api_port), self._api_handler())
        view = ThreadingHTTPServer((self._host, self._view_port), self._view_handler())
        for server in (api, view):
            server.daemon_threads = True
            server.allow_reuse_address = True
            self._servers.append(server)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            thread.start()
            self._threads.append(thread)
        self.api_url = f"http://{self._host}:{api.server_address[1]}"
        self.view_url = f"http://{self._host}:{view.server_address[1]}"
        return self

    def close(self) -> None:
        self._closed.set()
        self._wake.set()
        for server in self._servers:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(10.0)
        self._servers, self._threads = [], []

    def __enter__(self) -> "SimStation":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ status
    def _status(self) -> dict[str, Any]:
        with self._lock:
            age = None if not self._last_render else round(time.monotonic() - self._last_render, 2)
            return {
                "state": self.state, "prompt": self.prompt, "checkpoint": self.checkpoint,
                "action_space": "joint", "steps": 1, "run": self.run_number, "chunks": self.chunks,
                "stalls": self.stalls, "clamps": self.clamps, "arms": {name: True for name in ARMS},
                "cameras": {role: age for role in self.dc.VIDEO_ROLES.values()},
                "last_error": self.last_error, "started_at": None,
                "run_elapsed_s": None if self._run_started is None else round(time.monotonic() - self._run_started, 1),
                "sim": True, "policy": self.policy_kind,
            }

    @staticmethod
    def _checked_prompt(prompt: Any) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return prompt.strip()

    # ------------------------------------------------------------------ API operations (HTTP threads)
    def _do_start(self, body: dict) -> dict:
        prompt, checkpoint = body.get("prompt"), body.get("checkpoint")
        with self._lock:
            if self.state not in (IDLE, ERROR):
                raise _Refused(f"start refused: station is {self.state}")
            if prompt is not None:
                self.prompt = self._checked_prompt(prompt)
            if checkpoint is not None:
                if not isinstance(checkpoint, str) or not checkpoint.strip():
                    raise ValueError("checkpoint must be a non-empty string")
                self.checkpoint = self.dc.resolve_model(checkpoint) if self.policy_kind == "dream" else checkpoint
            if not self.prompt:
                raise _Refused("start refused: no task prompt; send the task with the start")
            self.last_error = None
            self.run_number += 1
            self.chunks = 0
            self._run_started = time.monotonic()
            self._stop_req = False
            self.state = STARTING
            self._start_req = {"prompt": self.prompt, "model": self.checkpoint}
            result = self._status()
        self._wake.set()
        return result

    def _do_stop(self) -> dict:
        with self._lock:
            if self.state == HOMING:
                raise _Refused("stop refused: a home move is in progress; it ends within seconds")
            if self.state == RESETTING:
                raise _Refused("stop refused: a reset is in progress; it ends within seconds")
            if self.state == STOPPING:
                raise _Refused("stop refused: already stopping")
            if self.state not in (STARTING, RUNNING):
                return self._status()
            self._stop_req = True
            self.state = STOPPING
        self._wake.set()
        deadline = time.monotonic() + API_TIMEOUT_S
        while time.monotonic() < deadline:
            with self._lock:
                if self.state != STOPPING:
                    return self._status()
            time.sleep(0.02)
        raise _Timeout("stop timed out")

    def _queue_cmd(self, kind: str) -> dict:
        done = threading.Event()
        result: dict = {}
        with self._lock:
            if self.state not in (IDLE, ERROR):
                raise _Refused(f"{kind} refused: station is {self.state}; stop the run first")
            self.last_error = None
            self.state = HOMING if kind == "home" else RESETTING
            self._cmd = (kind, done, result)
        self._wake.set()
        if not done.wait(API_TIMEOUT_S):
            raise _Timeout(f"{kind} timed out")
        return self._status()

    def _do_prompt(self, body: dict) -> dict:
        if "prompt" not in body:
            raise ValueError("body must be a JSON object with a 'prompt' key")
        with self._lock:
            self.prompt = self._checked_prompt(body["prompt"])
            return self._status()

    # ------------------------------------------------------------------ the sim thread
    def _sim_main(self, ready: threading.Event) -> None:
        try:
            self.env = self.yam_env.YamEnv()
            self._track_poses(self.env)
            self.hold = self._current_targets()
            self._render()
            if self.viewer_wanted:
                self._open_viewer()
        except Exception as exc:  # pragma: no cover
            print(f"sim_station: could not build the scene: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self.env = None
            ready.set()
            return
        ready.set()
        try:
            while not self._closed.is_set():
                with self._lock:
                    start_req, cmd = self._start_req, self._cmd
                    self._start_req = None
                if start_req is not None:
                    self._run(start_req)
                    continue
                if cmd is not None:
                    self._handle_cmd(cmd)
                    continue
                self._idle_tick()
        finally:
            if self._viewer is not None:
                try:
                    self._viewer.close()
                except Exception:
                    pass
            try:
                self.env.close()
            except Exception:
                pass

    def _track_poses(self, env: Any) -> None:
        """Wrap env.step/env.reset so every physics update refreshes the ground-truth snapshot behind /objects."""
        step, reset, images = env.step, env.reset, env.images

        model = env.model
        self._body_names = [model.body(i).name for i in range(model.nbody)
                            if (model.body_jntnum[i] and model.jnt_type[model.body_jntadr[i]] == 0)  # free joint
                            or model.body(i).name in STATIC_BODIES]

        def stepped(*args: Any, **kwargs: Any) -> Any:
            out = step(*args, **kwargs)
            if self._referee is not None:
                self._referee.update(env.model, env.data)
            self._snap_poses()
            self._sync_viewer()
            return out

        def was_reset(*args: Any, **kwargs: Any) -> Any:
            out = reset(*args, **kwargs)
            self._new_referee()
            self._snap_poses()
            self._sync_viewer(force=True)
            return out

        def rendered(*args: Any, **kwargs: Any) -> Any:
            out = images(*args, **kwargs)
            with self._lock:
                self._images = out
                self._images_n += 1
            return out

        env.step, env.reset, env.images = stepped, was_reset, rendered
        self._new_referee()
        self._snap_poses()

    # ------------------------------------------------------------------ live visualizers
    def _open_viewer(self) -> None:
        """The native MuJoCo window (passive: this thread steps the physics, the viewer only draws), with Dream's
        three camera tiles and a status overlay. On macOS it needs `mjpython` instead of `python`."""
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data,
                                                        show_left_ui=False, show_right_ui=False)
        except Exception as exc:
            hint = (" On macOS run it with mjpython (in the venv: .venv/bin/mjpython sim_station.py ... --viewer)."
                    if sys.platform == "darwin" else "")
            print(f"sim_station: no MuJoCo viewer ({type(exc).__name__}: {exc}).{hint} "
                  f"The web view at /live on the view port still works.", file=sys.stderr, flush=True)
            self._viewer = None
            return
        try:
            self.yam_env.aim_viewer(self._viewer)
            from camera_panel import CameraPanel  # type: ignore  # Dream's example, on sys.path via load_example

            self._viewer_panel = CameraPanel(self.env, self._viewer)
        except Exception:
            self._viewer_panel = None
        self._sync_viewer(force=True)

    def _sync_viewer(self, force: bool = False) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        now = time.monotonic()
        if not force and now - self._viewer_sync < 1 / 30:
            return
        self._viewer_sync = now
        try:
            if not viewer.is_running():
                self._viewer = None
                return
            import mujoco

            with self._lock:
                state, prompt, chunks, run = self.state, self.prompt, self.chunks, self.run_number
                referee = (self._poses or {}).get("referee")
            body = f"{state}  run {run}  chunks {chunks}\n{(prompt or '-')[:90]}"
            if referee:
                body += (f"\npoints {referee.get('points')}  unplugged {referee.get('unplugged')}  "
                         f"tower {referee.get('tower_in')} in  golf {referee.get('golf_in_bin')}"
                         + (f"  DQ {referee['dq']}" if referee.get("dq") else ""))
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "SIM STATION", body))
            viewer.sync()
            if self._viewer_panel is not None and self._images:
                self._viewer_panel.update(self._images)
        except Exception as exc:  # a viewer problem must never stop the station
            print(f"sim_station: viewer closed ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
            self._viewer = None

    def _camera_jpeg(self, name: str) -> bytes | None:
        """The newest render of one camera as JPEG, encoded at most once per render."""
        with self._lock:
            rgb, n = self._images.get(name), self._images_n
            cached = self._jpeg_cache.get(name)
        if rgb is None:
            return None
        if cached and cached[0] == n:
            return cached[1]
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
        data = buf.getvalue()
        with self._lock:
            self._jpeg_cache[name] = (n, data)
        return data

    def _live_state(self) -> dict[str, Any]:
        with self._lock:
            poses = dict(self._poses)
        trips = self._trip_list()[-12:]
        return {"status": self._status(), "objects": poses, "trips": trips, "scene": self.scene,
                "cameras": sorted(self._images), "frame": self._images_n}

    def _new_referee(self) -> None:
        """A /reset starts a new match for the referee (scene=hill_climb only)."""
        self._referee = self._referee_cls(self.env.model) if self._referee_cls else None

    def _snap_poses(self) -> None:
        data = self.env.data
        bodies = {name: [round(float(v), 4) for v in data.body(name).xpos] for name in self._body_names}
        tcp = {side: [round(float(v), 4) for v in data.site(f"{side}_tcp").xpos] for side in ARMS}
        state = self.env.state()
        joints = {side: [round(float(v), 4) for v in state[side]] for side in ARMS}
        snap = {"t": round(float(data.time), 3), "bodies": bodies, "tcp": tcp, "joints": joints}
        if self._referee is not None:
            snap["referee"] = self._referee.report(data)
        with self._lock:
            self._poses = snap

    def _current_targets(self) -> dict[str, Any]:
        state = self.env.state()
        return {side: state[side].copy() for side in ARMS}

    def _pace(self, tick: float, seconds: float = 0.05) -> None:
        if self.realtime:
            time.sleep(max(0.0, seconds - (time.monotonic() - tick)))

    def _render(self) -> dict:
        images = self.env.images()
        self._last_render = time.monotonic()
        return images

    def _idle_tick(self) -> None:
        """Hold the pose at 20 Hz and keep the cameras fresh (the harness refuses stale cameras)."""
        tick = time.monotonic()
        self.env.step(self.hold["left"], self.hold["right"])
        if time.monotonic() - self._last_render > 0.5:
            self._render()
        self._wake.wait(max(0.0, 0.05 - (time.monotonic() - tick)))
        self._wake.clear()

    def _handle_cmd(self, cmd: tuple[str, threading.Event, dict]) -> None:
        kind, done, result = cmd
        try:
            if kind == "home":
                for left, right in self.yam_env.home_path(self.env.state()):
                    tick = time.monotonic()
                    self.env.step(left, right)
                    if time.monotonic() - self._last_render > 0.25:
                        self._render()
                    self._pace(tick)
                self.hold = {"left": self.yam_env.HOME.copy(), "right": self.yam_env.HOME.copy()}
            else:
                self.env.reset()
                self.hold = self._current_targets()
            self._render()
        except Exception as exc:  # pragma: no cover
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._cmd = None
                self.state = IDLE
            done.set()

    # ------------------------------------------------------------------ one run (dream_control.main's loop)
    def _record_trip(self, windows: dict, obs: dict, started: float, reply: Any, error: str | None) -> None:
        names = {self.dc.frame_key(n, False): n for n in windows}
        state = {k: v for k, v in obs.items() if not k.startswith(("video.", "image_window."))}
        if reply is not None:
            action = {k: [list(map(float, row)) for row in v[:4]] for k, v in reply.items()
                      if k in ("q_l_chunk", "q_r_chunk")}
        else:
            action = None
        trip = {
            "n": 0, "sent_s": round(started - (self._run_started or started), 3),
            "frames": {key: list(windows[name]) for key, name in names.items()},
            "observation": state, "action": action,
            "timing": {"rtt_ms": round((time.monotonic() - started) * 1000, 1)},
            "error": error, "trained": [],
        }
        with self._lock:
            self._trip_n += 1
            trip["n"] = self._trip_n
            self._trips.append(trip)

    def _check_stop(self) -> None:
        if self._stop_req or self._closed.is_set():
            raise _Abort()

    def _open(self, model: str, poses: bool) -> Any:
        if self.policy_kind == "scripted":
            return ScriptedPolicy(self.hold, self.scripted_latency_s)
        return self.dc.open_policy(self.dc.policy_config(model, self.gateway, poses), self.key)

    @staticmethod
    def _retry_wait(error: Exception) -> float | None:
        """Seconds to wait if `error` is a retryable refusal (e.g. pool_warming), else None."""
        if getattr(error, "retryable", False) or getattr(error, "reason", None) == "pool_warming":
            return float(getattr(error, "retry_after_s", None) or 2.0)
        return None

    def _sleep_checking(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check_stop()
            time.sleep(0.05)

    def _run(self, req: dict) -> None:
        dc, env = self.dc, self.env
        model = req["model"]
        policy = None
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            from collections import deque as _dq

            windows = {name: _dq(maxlen=dc.FRAMES) for name in dc.VIDEO_ROLES}
            steps = 0
            targets = self._current_targets()
            while any(len(w) < dc.FRAMES for w in windows.values()):
                self._check_stop()
                tick = time.monotonic()
                state = env.state()
                env.step(state["left"], state["right"])
                if steps % dc.FRAME_EVERY_STEPS == 0:  # only these frames are kept: skip the other renders
                    dc.capture(env, windows, keep=True)
                    self._last_render = time.monotonic()
                steps += 1
                self._pace(tick)
            poses = model in dc.POSE_MODELS if self.policy_kind == "dream" else False
            self.hold = self._current_targets()
            policy = self._open_with_retry(model, poses)
            dc.feed(policy, env, windows, self.prompt, model, steps, every=0, station=poses)
            restarts = 0
            while True:
                self._check_stop()
                task = self.prompt  # a /prompt change applies from the next observation on
                started = time.monotonic()
                future = pool.submit(policy.get_action)
                while not future.done():
                    if self._stop_req or self._closed.is_set():
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise _Abort()
                    if time.monotonic() - self._last_render > 1.0:
                        self._render()
                    time.sleep(0.02)
                try:
                    reply = future.result()
                except Exception as error:
                    obs = dc.observation(env, windows, task, model, poses)
                    self._record_trip(windows, obs, started, None, f"{type(error).__name__}: {error}")
                    wait = self._retry_wait(error)
                    if type(error).__name__ == "DreamSessionExpired" and restarts < 3:
                        restarts += 1
                        continue
                    if wait is not None and self._first_reply_pending():
                        self._sleep_checking(wait)  # pool_warming: stay in `starting` and ask again
                        continue
                    if not poses and self.policy_kind == "dream" and dc.needs_poses(error):
                        policy.close()
                        poses = True
                        policy = self._open_with_retry(model, poses)
                        dc.feed(policy, env, windows, task, model, steps, every=0, station=poses)
                        continue
                    raise
                restarts = 0
                obs = dc.observation(env, windows, task, model, poses)
                self._record_trip(windows, obs, started, reply, None)
                adjustments: list[str] = []
                left, right = dc.validate_chunk(env, reply, self.actions, verbose=False, adjustments=adjustments)
                with self._lock:
                    self.chunks += 1
                    self.clamps += len(adjustments)
                    if self.state == STARTING:
                        self.state = RUNNING
                for l, r in zip(left[: self.actions], right[: self.actions]):
                    self._check_stop()
                    tick = time.monotonic()
                    env.step(l, r)
                    targets = {"left": l.copy(), "right": r.copy()}
                    kept = steps % dc.FRAME_EVERY_STEPS == 0
                    if kept:
                        dc.capture(env, windows, keep=True)
                        self._last_render = time.monotonic()
                        dc.feed(policy, env, windows, self.prompt, model, steps, station=poses)
                    elif time.monotonic() - self._last_render > 1.0:
                        self._render()
                    steps += 1
                    self._pace(tick)
                self.hold = targets
        except _Abort:
            pass
        except Exception as exc:
            with self._lock:
                self.last_error = {
                    "message": str(exc) or type(exc).__name__, "type": type(exc).__name__,
                    "error_code": getattr(exc, "code", None) or getattr(exc, "reason", None)
                    or type(exc).__name__}
            print(f"sim_station: run {self.run_number} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._finish_run(policy, pool, ERROR)
            return
        self._finish_run(policy, pool, IDLE)

    def _first_reply_pending(self) -> bool:
        with self._lock:
            return self.state == STARTING

    def _open_with_retry(self, model: str, poses: bool) -> Any:
        while True:
            self._check_stop()
            try:
                return self._open(model, poses)
            except Exception as error:
                wait = self._retry_wait(error)
                if wait is None:
                    raise
                self._sleep_checking(wait)

    def _finish_run(self, policy: Any, pool: ThreadPoolExecutor, final: str) -> None:
        """Hold the arms where they are, release the session, and settle in `final`."""
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            self.hold = self._current_targets() if self.hold is None else self.hold
            self._render()
        except Exception:
            pass
        if policy is not None:
            threading.Thread(target=self._close_policy, args=(policy,), daemon=True).start()
        with self._lock:
            self._stop_req = False
            self.state = final

    @staticmethod
    def _close_policy(policy: Any) -> None:
        try:
            policy.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ HTTP
    def _trip_list(self) -> list[dict]:
        with self._lock:
            return [{"n": t["n"], "sent_s": t["sent_s"], "error": t["error"], "rtt_ms": t["timing"]["rtt_ms"]}
                    for t in self._trips]

    def _trip(self, n: int) -> dict | None:
        with self._lock:
            return next((t for t in self._trips if t["n"] == n), None)

    def _api_handler(self) -> type[BaseHTTPRequestHandler]:
        sim = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                route = self.path.split("?", 1)[0]
                sim.calls.append(("GET", route, None))
                if route == "/status":
                    self._reply(200, sim._status())
                elif route == "/objects":  # sim only: ground-truth poses for offline benchmarks, never the harness
                    with sim._lock:
                        self._reply(200, dict(sim._poses))
                else:
                    self._reply(404, {"error": f"no such route: GET {self.path}"})

            def do_POST(self) -> None:
                route = self.path.split("?", 1)[0]
                try:
                    body = self._read_json()
                except ValueError as exc:
                    sim.calls.append(("POST", route, None))
                    self._reply(400, {"error": str(exc)})
                    return
                sim.calls.append(("POST", route, body))
                try:
                    if route == "/start":
                        result = sim._do_start(body)
                    elif route == "/stop":
                        result = sim._do_stop()
                    elif route == "/home":
                        result = sim._queue_cmd("home")
                    elif route == "/reset":
                        result = sim._queue_cmd("reset")
                    elif route == "/prompt":
                        result = sim._do_prompt(body)
                    else:
                        self._reply(404, {"error": f"no such route: POST {route}"})
                        return
                except _Refused as exc:
                    self._reply(409, {"error": str(exc)})
                except ValueError as exc:
                    self._reply(400, {"error": str(exc)})
                except _Timeout as exc:
                    self._reply(504, {"error": str(exc)})
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
        sim = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                number = path.removeprefix("/trip/")
                cam = path.removeprefix("/live/cam/").removesuffix(".jpg")
                if path in ("/", "/live"):
                    self._send(200, LIVE_PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/live/state":
                    self._send(200, json.dumps(sim._live_state(), default=str).encode(), "application/json")
                elif path.startswith("/live/cam/") and (jpeg := sim._camera_jpeg(cam)) is not None:
                    self._send(200, jpeg, "image/jpeg")
                elif path == "/trips":
                    self._send(200, json.dumps(sim._trip_list()).encode(), "application/json")
                elif path.startswith("/trip/") and number.isdigit() and (trip := sim._trip(int(number))):
                    self._send(200, json.dumps(trip).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def _send(self, code: int, body: bytes, kind: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        return Handler


LIVE_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sim station · live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b35;--ink:#e7e9ee;--dim:#8a93a3;--ok:#3ecf8e;--warn:#f5b84a;--bad:#ef5a5a;--info:#5aa7ef}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 ui-sans-serif,-apple-system,"Segoe UI",sans-serif}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}h1 span{color:var(--dim);font-weight:400}.dim{color:var(--dim)}
main{display:grid;grid-template-columns:minmax(560px,2fr) minmax(300px,1fr);gap:12px;padding:12px 16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:0 0 8px;font-weight:600}
.cams{display:grid;grid-template-columns:2fr 1fr;gap:8px}.cams .side{display:grid;gap:8px}
.cam{position:relative;background:#000;border-radius:8px;overflow:hidden}.cam img{display:block;width:100%}
.cam span{position:absolute;left:8px;top:6px;font-size:11px;background:#0009;padding:1px 6px;border-radius:4px}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;border:1px solid var(--line)}
.pill.running{color:var(--ok);border-color:var(--ok)}.pill.starting,.pill.stopping,.pill.homing,.pill.resetting{color:var(--warn);border-color:var(--warn)}
.pill.error{color:var(--bad);border-color:var(--bad)}.pill.idle{color:var(--dim)}
table{width:100%;border-collapse:collapse;font:12px/1.5 ui-monospace,Menlo,monospace}td{padding:2px 4px;border-bottom:1px solid #1c2029}
td:first-child{color:var(--dim)}.bad{color:var(--bad)}.ok{color:var(--ok)}
#prompt{font-size:14px;margin-top:6px}
</style></head><body>
<header><h1>Sim station <span>· live view</span></h1><span class="pill" id="state">–</span>
<span class="dim">run</span><b id="run">–</b><span class="dim">chunks</span><b id="chunks">0</b>
<span class="dim">elapsed</span><b id="elapsed">–</b><span class="dim">policy</span><b id="policy">–</b>
<span class="dim">fps</span><b id="fps">–</b></header>
<main>
<section style="display:grid;gap:12px">
  <div class="card"><div class="cams">
    <div class="cam"><img id="cam_top" alt=""><span>overhead (top)</span></div>
    <div class="side"><div class="cam"><img id="cam_left" alt=""><span>left wrist</span></div>
      <div class="cam"><img id="cam_right" alt=""><span>right wrist</span></div></div></div>
    <div id="prompt"><span class="dim">prompt</span> <span id="ptext">–</span></div>
    <div class="bad" id="err"></div></div>
  <div class="card"><h2>Top-down map (ground truth)</h2><svg id="map" viewBox="0 0 600 300" width="100%"></svg></div>
</section>
<section style="display:grid;gap:12px;align-content:start">
  <div class="card" id="refcard"><h2>Referee</h2><table id="ref"></table></div>
  <div class="card"><h2>Station</h2><table id="st"></table></div>
  <div class="card"><h2>Policy trips (newest last)</h2><table id="trips"></table></div>
</section></main>
<script>
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let frames=[];
for(const n of ['top','left','right']){const img=$('cam_'+n);
  const load=()=>{img.src='/live/cam/'+n+'.jpg?t='+Date.now()};
  img.onload=()=>{if(n=='top'){frames.push(Date.now());frames=frames.filter(t=>Date.now()-t<3000)}setTimeout(load,60)};
  img.onerror=()=>setTimeout(load,1000);load()}
const B={x0:1e9,x1:-1e9,y0:1e9,y1:-1e9};
function col(n){if(/^eight_/.test(n))return['#050505','#ef5a5a'];if(/^golf_/.test(n))return['#f2f2f2','#999'];
  if(/^cube_|^cylinder_|block/.test(n))return['#f0b04a','#7a5a20'];if(/plug|cable|usb/.test(n))return['#5aa7ef','#2a5a8a'];
  if(/hub/.test(n))return['#3a4256','#8a93a3'];if(/bin|bowl|plate/.test(n))return['#2a2f3a','#cfd5e2'];return['#a58cf5','#55488a']}
function map(o){if(!o||!o.bodies)return'';const pts=Object.entries(o.bodies).concat(Object.entries(o.tcp||{}).map(([s,p])=>['tcp_'+s,p]));
  for(const[,p]of pts){B.x0=Math.min(B.x0,p[0]);B.x1=Math.max(B.x1,p[0]);B.y0=Math.min(B.y0,p[1]);B.y1=Math.max(B.y1,p[1])}
  const m=.08,sx=(B.x1-B.x0+2*m)||1,sy=(B.y1-B.y0+2*m)||1,k=Math.min(560/sx,260/sy);
  // as the overhead camera sees it: the arms at the bottom, the left arm on the left (world -x to the right,
  // world +y down the page)
  const X=p=>20+(B.x1+m-p[0])*k,Y=p=>20+(p[1]-B.y0+m)*k;let g='';
  for(const[n,p]of Object.entries(o.bodies)){const[f,s]=col(n),r=/hub|bin|bowl|plate/.test(n)?14:/^eight_/.test(n)?7:6;
    const z=p[2].toFixed(2);g+=`<g><title>${esc(n)} z=${z}</title>`+(/hub|bin|bowl|plate/.test(n)?
      `<rect x="${X(p)-r}" y="${Y(p)-r}" width="${2*r}" height="${2*r}" rx="3" fill="${f}" stroke="${s}"/>`:
      `<circle cx="${X(p)}" cy="${Y(p)}" r="${r}" fill="${f}" stroke="${s}"/>`)+
      (/hub|bin|bowl|plate/.test(n)?`<text x="${X(p)}" y="${Y(p)+r+12}" fill="#8a93a3" font-size="10" text-anchor="middle">${esc(n)}</text>`:'')+`</g>`}
  for(const[s,p]of Object.entries(o.tcp||{})){const x=X(p),y=Y(p),c=s=='left'?'#3ecf8e':'#ef5a5a';
    g+=`<path d="M${x-9} ${y}H${x+9}M${x} ${y-9}V${y+9}" stroke="${c}" stroke-width="2"/><circle cx="${x}" cy="${y}" r="${4+8*Math.max(0,Math.min(1,(p[2]-0.75)/0.3))}" fill="none" stroke="${c}"/>
      <text x="${x+11}" y="${y-6}" fill="${c}" font-size="10">${s} tcp z=${p[2].toFixed(2)}</text>`}
  return g}
function rows(obj){return Object.entries(obj||{}).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="${/dq|eight|error/.test(k)&&v&&v!=0&&!(Array.isArray(v)&&!v.length)?'bad':''}">${esc(typeof v=='object'?JSON.stringify(v):v)}</td></tr>`).join('')}
async function tick(){try{const s=await (await fetch('/live/state')).json(),st=s.status;
  $('state').textContent=st.state;$('state').className='pill '+st.state;$('run').textContent=st.run;$('chunks').textContent=st.chunks;
  $('elapsed').textContent=st.run_elapsed_s==null?'–':st.run_elapsed_s+' s';$('policy').textContent=st.policy+' · '+st.checkpoint;
  $('fps').textContent=(frames.length/3).toFixed(1);$('ptext').textContent=st.prompt||'–';
  $('err').textContent=st.last_error?('last error: '+st.last_error.type+': '+st.last_error.message):'';
  const ref=s.objects&&s.objects.referee;$('refcard').style.display=ref?'':'none';if(ref)$('ref').innerHTML=rows(ref);
  $('st').innerHTML=rows({stalls:st.stalls,clamps:st.clamps,arms:st.arms,'camera age s':st.cameras,'sim t':s.objects&&s.objects.t,scene:s.scene});
  $('trips').innerHTML=s.trips.map(t=>`<tr><td>#${t.n} @${t.sent_s}s</td><td class="${t.error?'bad':''}">${t.error?esc(t.error):t.rtt_ms+' ms'}</td></tr>`).join('')||'<tr><td>none yet</td></tr>';
  $('map').innerHTML=map(s.objects)}catch(e){$('state').textContent='offline';$('state').className='pill error'}}
setInterval(tick,250);tick();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulated Dream station agent (MuJoCo YAM) for the harness.")
    parser.add_argument("--policy", choices=("dream", "scripted"), default="dream")
    parser.add_argument("--checkpoint", default=None, help="default model: beta, or any model id (dream mode)")
    parser.add_argument("--api-port", type=int, default=8091)
    parser.add_argument("--view-port", type=int, default=8093)
    rt = parser.add_mutually_exclusive_group()
    rt.add_argument("--realtime", dest="realtime", action="store_true", default=None,
                    help="sim time = wall time (default for dream)")
    rt.add_argument("--no-realtime", dest="realtime", action="store_false", help="as fast as practical")
    parser.add_argument("--example-dir", default=None, help="dream_examples/yam_mujoco dir if not installed")
    parser.add_argument("--actions", type=int, default=48, help="rows executed per chunk, 1-48")
    parser.add_argument("--scene", choices=SCENES, default="fruit",
                        help="fruit: Dream's example table; hill_climb: the relay table with a rubric referee")
    parser.add_argument("--viewer", action="store_true",
                        help="open the native MuJoCo 3D viewer (macOS: run with mjpython instead of python)")
    parser.add_argument("--open", action="store_true", help="open the live web view (/live) in the browser")
    args = parser.parse_args(argv)
    if not 1 <= args.actions <= 48:
        parser.error("--actions must be 1-48")
    if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
        # Headless offscreen rendering: EGL needs a GPU render node, else software OSMesa (libosmesa6).
        os.environ["MUJOCO_GL"] = "egl" if Path("/dev/dri").exists() else "osmesa"
    sim = SimStation(policy=args.policy, checkpoint=args.checkpoint, api_port=args.api_port,
                     view_port=args.view_port, realtime=args.realtime, example_dir=args.example_dir,
                     actions=args.actions, scene=args.scene, viewer=args.viewer).start()
    print(f"sim station ({args.policy}, {'realtime' if sim.realtime else 'fast'}): api {sim.api_url}  "
          f"view {sim.view_url}  live {sim.view_url}/live  (Ctrl+C to quit)", flush=True)
    if args.open:
        import webbrowser

        webbrowser.open(f"{sim.view_url}/live")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sim.close()


if __name__ == "__main__":
    main()
