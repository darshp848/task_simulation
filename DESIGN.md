# Hill Climb harness: design (the contract every file follows)

## What it is

A supervisor that runs Dream-1 through a long-horizon task as a sequence of stages. For each stage it gives
the model a prompt, watches the cameras, decides whether the stage worked, and recovers when it did not.

It never touches a motor. It drives Dream's own station agent (`dreamcli robot up`) through that agent's
HTTP API, so every speed cap, clamp, gripper force cap and home move in Dream's control loop stays in force.

## The two endpoints it uses (both on the station, both 127.0.0.1)

Station agent, started by the person with `dreamcli robot up [joint] --view-port 8093` (default API port 8091).
Routes (from `robots/station_api.py`, dreamlabs 1.0.37):

| Route | Body | Reply |
| --- | --- | --- |
| `GET /status` | | 200, status dict |
| `POST /start` | `{"prompt": str?, "checkpoint": str?}` | 200 status; 409 `{"error"}` when a run exists |
| `POST /stop` | `{}` | 200 status once the arms hold; 409 during a home move |
| `POST /home` | `{}` | 200 once the arms are home; 409 while a run exists |
| `POST /reset` | `{}` | 200 status (reopens a dead arm driver) |
| `POST /prompt` | `{"prompt": str}` | 200 status; 400 without a non-empty prompt. Changes the running task live |

Errors: 400 bad body, 404 route, 409 refused transition, 504 timeout (60 s), 500 other; body `{"error": msg}`.

Status dict keys used: `state` (one of `idle`, `starting`, `running`, `stopping`, `homing`, `resetting`,
`error`), `prompt`, `checkpoint`, `run` (int, run number), `chunks`, `stalls`, `clamps`, `arms`
(`{name: bool alive}`), `cameras` (`{name: age_s or None}`), `last_error` (str or None), `run_elapsed_s`.

Debug view (`--view-port 8093`), from `dream/debug_view.py`:

- `GET /trips` -> JSON list, oldest first, of `{"n": int, "sent_s": float, "error": str|None, "rtt_ms": ...}`.
- `GET /trip/<n>` -> JSON `{"n", "sent_s", "frames": {"video.<role>": [base64 JPEG, ... oldest first]},
  "observation", "action", "timing", "error", "trained"}`; 404 when no longer kept.

The harness reads the newest trip's newest frame per camera as "what the robot sees now".
Camera role keys look like `video.scene`, `video.wrist_l`, `video.wrist_r` (exact names vary by station:
match by substring, config `cameras_for_check` lists which roles a verifier gets, default all).

## Package layout (`harness/`, Python 3.12+, stdlib + PyYAML only; no numpy, no SDKs)

```
harness/
  __init__.py
  station_client.py   StationClient
  frames.py           FrameSource
  plan.py             Plan, Stage, load_plan()
  verifiers.py        Verifier protocol + ManualVerifier, ClaudeVerifier, TimeVerifier, make_verifier()
  runner.py           Runner, RunResult
  log.py              RunLog (JSONL + saved JPEGs)
  cli.py              main()  (python -m harness ...)
plans/example_hill_climb.yaml
fake_station.py       (offline stand-in for both endpoints; see below)
tests/                pytest, against fake_station
README.md
```

### station_client.py

```python
class StationError(Exception): ...          # carries .status_code and .message
class StationClient:
    def __init__(self, base_url="http://127.0.0.1:8091", *, timeout_s=70.0, dry_run=False, log=None): ...
    def status(self) -> dict                  # GET /status (allowed in dry run)
    def start(self, prompt: str, checkpoint: str | None = None) -> dict
    def set_prompt(self, prompt: str) -> dict
    def stop(self) -> dict
    def home(self) -> dict
    def reset(self) -> dict
    def wait_for_state(self, states: set[str], timeout_s: float, poll_s=0.5) -> dict  # raises TimeoutError
```

`dry_run=True`: every POST is NOT sent; it logs `"DRY RUN: would POST /start {...}"` and returns the last
real status (or a synthetic `{"state": "running", ...}` when the station is unreachable). GETs are sent.
Uses `urllib.request` only. Non-2xx -> `StationError`.

### frames.py

```python
class FrameSource:
    def __init__(self, base_url="http://127.0.0.1:8093", *, timeout_s=5.0): ...
    def latest(self) -> dict[str, bytes]      # role -> JPEG bytes, newest trip's newest frame per key; {} if none
    def latest_n(self) -> int | None
```
Strips a `video.` prefix from role names. Never raises on network trouble: returns `{}` and remembers the error.

### plan.py (YAML schema)

```yaml
task: "clear the table into the bin"         # overall, for logs
checkpoint: null                              # dream-1-ft:... or null for base Dream-1
time_budget_s: 900                            # whole run
check_every_s: 3.0                            # how often a verifier is asked while a stage runs
verifier: manual                              # default for stages: manual | claude | time
claude: {model: claude-haiku-5-5, max_tokens: 200}
stages:
  - name: pick_red
    prompt: "pick up the red cube"
    success: "Is the red cube held in a gripper, off the table?"   # the yes/no question a verifier answers
    timeout_s: 60
    max_attempts: 3
    min_run_s: 4                                # don't ask a verifier before this
    verifier: manual                            # optional override
    recovery:                                   # tried in order on attempt 2, 3, ... (else the stage prompt again)
      - "open both grippers and move the arms up"
    on_fail: home_and_retry                     # retry_in_place | home_and_retry | skip | abort
```

`load_plan(path) -> Plan` validates: non-empty prompts (<=512 chars, single line), positive times,
`max_attempts>=1`, known enums; raises `ValueError` naming the stage and field.

### verifiers.py

```python
@dataclass
class Verdict: done: bool | None; confidence: float; reason: str   # None = can't tell yet
class Verifier(Protocol):
    def check(self, stage: Stage, frames: dict[str, bytes], elapsed_s: float) -> Verdict: ...
```

- `ManualVerifier(input_stream=sys.stdin)`: a background thread reads lines; `y`=done, `n`=failed now,
  anything else ignored; `check()` returns the latest unread key press as a verdict, else `Verdict(None,...)`.
  Prints the stage question once per stage.
- `ClaudeVerifier(model, api_key from ANTHROPIC_API_KEY, max_tokens)`: POST to
  `https://api.anthropic.com/v1/messages` with headers `x-api-key`, `anthropic-version: 2023-06-01`,
  `content-type: application/json`; content = each camera JPEG as an `image` block (base64, media_type
  image/jpeg) labelled by role, then the text: the stage question + "Answer with JSON only:
  {\"done\": true|false, \"confidence\": 0..1, \"reason\": \"...\"}". Parses the first JSON object in the reply.
  Any error -> `Verdict(None, 0, "verifier error: ...")`. Requires done with confidence >= 0.7 to count.
  Never sends the key anywhere else; never logs it.
- `TimeVerifier`: done once `elapsed_s >= stage.timeout_s * 0.9` (for stages with no check), confidence 0.5.
- `make_verifier(name, plan) -> Verifier`.

### runner.py: the state machine

```python
@dataclass
class StageResult: name: str; ok: bool; attempts: int; seconds: float; reason: str
@dataclass
class RunResult: ok: bool; stages: list[StageResult]; seconds: float; reason: str
class Runner:
    def __init__(self, plan, station: StationClient, frames: FrameSource, verifiers: dict[str, Verifier],
                 log: RunLog, *, clock=time.monotonic, sleep=time.sleep): ...
    def run(self) -> RunResult
    def request_stop(self) -> None            # thread-safe; makes run() stop the station and return
```

Behaviour:
1. Preflight: `status()`. Refuse (RunResult ok=False) if state not in {idle, error}... except state
   `running` is refused too (someone else's run). Refuse if any `arms` value is False or any camera age is
   None or > 2 s. In dry run, report these as warnings and continue.
2. First stage: `start(prompt, plan.checkpoint)`, then `wait_for_state({"running"}, 120)` (a new model can be
   slow; on `error` state stop and fail with `last_error`).
3. Each later stage, if the run is still `running`: `set_prompt(next prompt)` (no stop between stages).
   If the run ended (state idle) start it again with the new prompt.
4. While a stage runs: every `check_every_s`, poll status; if `state == "error"` or an arm is dead -> `stop()`
   and fail the whole run (`abort`). After `min_run_s`, grab `frames.latest()` and ask the stage's verifier.
   `done True` -> stage ok, next stage. `done False` (manual `n`) -> attempt failed.
   `elapsed >= timeout_s` -> attempt failed.
5. Attempt failed: if attempts < max_attempts: by `on_fail`:
   - `retry_in_place`: `set_prompt(recovery[i] if any else prompt)`; for a recovery prompt, run it for
     `min(timeout_s/2, 20)` s, then `set_prompt(prompt)` again and restart the stage clock.
   - `home_and_retry`: `stop()`, wait idle, `home()`, wait idle, `start(prompt)`.
   - `skip`: mark failed, go to next stage. `abort`: stop and fail the run.
   Out of attempts: `skip` behaviour if on_fail is `skip`, else fail the run.
6. Whole-run `time_budget_s` exceeded -> stop, fail. `request_stop()` or KeyboardInterrupt -> `stop()` then
   return. After the last stage: `stop()` (the arms hold; the person homes them).
7. Every decision goes to the log.

### log.py

`RunLog(dir)`: creates `runs/<UTC timestamp>/`, writes `events.jsonl` (`{"t", "event", ...}` per line),
`save_frames(tag, frames)` writes `<seq>_<tag>_<role>.jpg`, and `summary.json` at the end.

### cli.py (`python -m harness`)

```
python -m harness plans/x.yaml --dry-run          # reads status/frames, sends nothing that moves an arm
python -m harness plans/x.yaml --live              # real; asks the person to type "yes" first
python -m harness plans/x.yaml --mock              # starts fake_station in-process, no station needed
  --station http://127.0.0.1:8091 --view http://127.0.0.1:8093 --runs-dir runs
```
Exactly one of `--dry-run`, `--live`, `--mock` is required. `--live` prints the safety lines (hand on the
e-stop, people clear) and requires typing `yes`. Ctrl+C at any time -> `POST /stop` then exit.
`s` + Enter in the terminal while running also stops (shares the stdin reader with ManualVerifier).

## fake_station.py (offline stand-in)

`FakeStation(api_port=0, view_port=0, *, scenario="happy")`: two `ThreadingHTTPServer`s on 127.0.0.1
implementing exactly the routes above, with the real state machine (idle -> starting -> running; stop ->
stopping -> idle; home -> homing -> idle; 409s for refused transitions), a growing `chunks` count, trip list
with synthetic JPEG frames (a tiny valid JPEG constant; roles `video.scene`, `video.wrist_l`, `video.wrist_r`)
whose observation carries the current prompt. `.api_url`, `.view_url`, `.start()`, `.close()`, `.calls`
(list of `(method, route, body)`). Scenarios: `happy`; `arm_dies` (an arm goes False 3 s into the run);
`start_error` (start -> error state with last_error); `slow_start` (5 s in starting); `camera_stale`.
Also `fake_station.FakeVerifier(script: dict[stage_name, list[Verdict|None]])` returning scripted verdicts.
Runnable alone: `python fake_station.py` serves on 8091/8093.
