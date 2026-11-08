# Hill Climb harness

A supervisor that runs Dream-1 through a long-horizon task as a sequence of stages. For each stage it
gives the model a prompt, watches the cameras, decides whether the stage worked, and recovers (re-prompt,
home and retry, skip, or abort) when it did not. Every decision goes to `runs/<timestamp>/events.jsonl`
with the camera frames it looked at.

Python 3.12+, stdlib + PyYAML only. The design contract is `DESIGN.md`.

## Safety model

- **It never touches a motor.** It has no robot driver, no joint commands, no camera drivers. It only
  calls Dream's own station agent over HTTP on 127.0.0.1 (`/status`, `/start`, `/stop`, `/home`, `/reset`,
  `/prompt`) and reads frames from the debug view. Every speed cap, clamp, gripper force cap and home move
  stays in Dream's control loop.
- **Preflight refuses** to start if the station is not idle, an arm is dead, or a camera is stale (> 2 s).
- **Fails safe**: an arm dying, the station entering `error`, the time budget, Ctrl+C, or `s` + Enter all
  send `POST /stop` (the arms hold where they are). It never homes at the end; you do that.
- `--dry-run` sends no POST at all; `--live` makes you type `yes` after the safety lines.
- The physical e-stop is still the real safety. Keep your hand on it.

## Run it (on the station)

```bash
source ~/dream-env/bin/activate
dreamcli robot up --view-port 8093            # Metal fine-tune too; a YAM fine-tune: dreamcli robot up joint ...

python -m harness plans/example_hill_climb.yaml --dry-run    # reads status/frames, prints what it would POST
python -m harness plans/example_hill_climb.yaml --live       # hand on the e-stop, people clear; type yes

dreamcli robot down                                          # at the end
```

Keep the `dreamcli robot up` terminal running; run the harness in a second terminal (also with the
`dream-env` activated). Options: `--station URL` (default `http://127.0.0.1:8091`), `--view URL`
(default `http://127.0.0.1:8093`), `--runs-dir DIR`, `--verifier manual|claude|time` (override every stage).

While running, type `y` / `n` + Enter to answer a manual verifier, `s` + Enter to stop.

Offline practice, no station needed: `python -m harness plans/example_hill_climb.yaml --mock
[--scenario happy|arm_dies|start_error|slow_start|camera_stale]` (uses `fake_station.py`). Manual stages
wait for your keypresses; add `--verifier time` to let it run unattended.

## Test in simulation

`sim_station.py` is a stand-in for the station agent that drives Dream's own MuJoCo YAM simulator
(`dream_examples/yam_mujoco`), with the same HTTP routes on 8091 and the same debug view on 8093. The harness
cannot tell the difference, so you can test a plan end to end, with the real Dream-1 model, before the robot.
It uses Dream's own control loop (camera windows, `get_action`, `validate_chunk`, 20 Hz stepping), so nothing
physical is involved and `--live` is safe to confirm.

On a laptop with the `dreamlabs` library installed and your Dream API key saved (`~/.dreamlabs/credentials`,
from the Getting started page):

```bash
# terminal 1: the simulated station (real Dream-1, real time)
python sim_station.py --policy dream [--checkpoint dream-1-ft:...]

# terminal 2: the harness against it
python -m harness plans/sim_fruit_bowl.yaml --live --station http://127.0.0.1:8091 --view http://127.0.0.1:8093
```

`plans/sim_fruit_bowl.yaml` has three stages for objects in the sim scene (apple, orange, banana, a bowl and a
blue plate). The verifier is `manual`: watch the saved frames in `runs/<timestamp>/` or the `/trip/<n>` frames
on 8093, or run with `--verifier time` to let it run unattended. If Dream's example is not under
site-packages, add `--example-dir PATH/TO/dream_examples/yam_mujoco`. `--no-realtime` runs the sim as fast as it
renders.

Offline, no key and no network: `python sim_station.py --policy scripted` returns small sinusoidal joint chunks
of the right shape, so the harness, the state machine and the frames can be tested anywhere that has
`pip install mujoco pillow numpy`. On a headless Linux box set `MUJOCO_GL=egl` (needs a GPU) or
`MUJOCO_GL=osmesa` (software; `apt install libosmesa6`), which is the automatic default there. Software
rendering is slow (about 5 s per chunk), so use long stage timeouts or `--verifier time`.
`pytest tests/test_sim_station.py` covers the scripted mode and skips itself without mujoco.

## Match plan and the fine-tune

`plans/hill_climb_v4.yaml` is the match plan for rubric v4. Its prompts are the six task texts in
`TELEOP_PLAN.md`, the texts the fine-tune is recorded with. Check that a plan's prompts all have demos:

```bash
python -m harness.tasks plans/hill_climb_v4.yaml ~/hackathon_data/unplug-v1 ~/hackathon_data/relay-v1
```

It also has a black-ball `guard:`. On every ball check the monitor answers "is a gripper holding a black 8-ball?",
and a yes stops the run at once, with the ball still held (HARNESS_GUIDE.md section 6d).

## Writing a plan

```yaml
task: "clear the table into the bin"      # for logs
checkpoint: null                           # dream-1-ft:... or null for base Dream-1
time_budget_s: 900                         # whole run
check_every_s: 3.0                         # how often a verifier is asked
verifier: manual                           # default: manual | claude | time
cameras_for_check: [scene]                 # optional: roles a verifier sees (substring match), default all
stages:
  - name: pick_red                         # unique
    prompt: "pick up the red cube"         # one line, <= 512 chars; sent to the model as-is
    success: "Is the red cube held in a gripper, off the table?"   # yes/no question for the verifier
    timeout_s: 60                          # attempt fails after this
    max_attempts: 3
    min_run_s: 4                           # no verifier call before this
    verifier: manual                       # optional per-stage override
    recovery:                              # prompts tried in order on attempt 2, 3, ...
      - "open both grippers and move the arms up"
    on_fail: home_and_retry                # retry_in_place (default) | home_and_retry | skip | abort
```

Stages run back to back; between stages the harness changes the prompt live (`/prompt`) with no stop.
`retry_in_place` runs the recovery prompt for `min(timeout_s/2, 20)` s and then the stage prompt again;
`home_and_retry` does stop, home, start. Out of attempts: `skip` moves on, anything else fails the run.
Bad plans are rejected with the stage and field named.

## Verifiers

- `manual` (default): you watch the robot; `y` = stage done, `n` = attempt failed now.
- `time`: for stages with no check; done after 90% of `timeout_s`.
- `claude` (optional): sends the camera frames and the stage question to the Anthropic API
  (`claude-haiku-5-5` by default) and needs `ANTHROPIC_API_KEY` in the environment. The key is only sent to
  api.anthropic.com and is never logged or printed. A stage counts as done only for `done: true` with
  confidence >= 0.7. **Using an external model as a judge may not be allowed by the event rules. Check with
  the organizers before using it.** Use `manual` if unsure.

- `claude+manual`: Claude judges every check, and your `y` / `n` overrules it. Run with
  `--verifier claude+manual`. Good for practice runs: you watch Claude's calls in `events.jsonl` and correct them.

## Logs

`runs/<UTC timestamp>/` holds `events.jsonl` (one JSON event per line), `<seq>_<tag>_<role>.jpg` frames the
verifier saw, and `summary.json`.
