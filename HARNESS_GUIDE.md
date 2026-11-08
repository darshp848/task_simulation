# Hill Climb harness: how to use it

The harness runs Dream-1 through a long task as a list of stages. For each stage it:

1. sends a prompt,
2. watches the cameras,
3. decides whether the stage worked,
4. recovers when it didn't.

Version 0.2.0. The design contracts are `DESIGN.md` (v1) and `DESIGN_v2.md` (v2). The short reference is `README.md`.

## 1. What it is, and what it never does

- **It never touches a motor.** It only talks to Dream's own station agent (`dreamcli robot up`) over HTTP on 127.0.0.1:
  - `/status`, `/start`, `/stop`, `/home`, `/reset`, `/prompt` on port 8091
  - the frames from the debug view on port 8093 (`/trips`, `/trip/<n>`)

  Every speed cap, clamp, gripper force cap and home move stays inside Dream's control loop.
- **Preflight refuses to start** if:
  - the station is not idle (or is in error),
  - an arm is dead,
  - a camera is older than 2 s.
- **It fails safe.** Each of these sends `POST /stop`, and the arms hold where they are:
  - an arm dies
  - the station goes into `error`
  - the time budget runs out
  - Ctrl+C
  - `s` + Enter
- It never homes the arms at the end. You do that yourself.
- The physical e-stop is still the real safety. Keep a hand on it during `--live` runs.

## 2. Install

Python 3.12 or newer. The harness needs only PyYAML. On the station, the `dream-env` already has it.

```bash
unzip hill_climb_harness.zip && cd hill_climb_harness
pip install pyyaml pytest             # pytest only for the tests
python -m pytest -q                   # offline, against fake_station.py
```

## 3. Where to run it

**Option A: on the station.** This needs two terminals, both with `source ~/dream-env/bin/activate`.

```bash
# terminal 1: Dream's station agent (leave it running)
dreamcli robot up --view-port 8093          # Metal fine-tune too; only a YAM fine-tune needs: dreamcli robot up joint

# terminal 2: the harness
python -m harness plans/<plan>.yaml --dry-run
python -m harness plans/<plan>.yaml --live
```

**Option B: on your laptop, through an SSH tunnel to the station or NUC.** The station agent still runs on the station. The laptop only sends HTTP through the tunnel.

```bash
ssh -L 8091:localhost:8091 -L 8093:localhost:8093 <user>@<station-ip>
# in another laptop terminal:
python -m harness plans/<plan>.yaml --dry-run      # then --live
```

Both endpoints listen on 127.0.0.1 on the station, so the tunnel is the only way to reach them from outside. If you also want the dashboard, add `-L 8080:localhost:8080`.

## 4. The three modes (exactly one is required)

| Mode | What it does |
| --- | --- |
| `--dry-run` | Reads status and frames, and prints every POST it *would* send. It sends nothing that moves an arm. Always do this first on a new station. |
| `--live` | The real run. It prints the safety lines (hand on the e-stop, people clear) and waits until you type `yes`. |
| `--mock` | Starts `fake_station.py` in-process, so no station is needed. Add `--scenario happy\|arm_dies\|start_error\|slow_start\|camera_stale` to practise failure handling. |

Other options:

- `--station URL` (default `http://127.0.0.1:8091`)
- `--view URL` (default `http://127.0.0.1:8093`)
- `--runs-dir DIR` (default `runs`)
- `--verifier NAME` overrides every stage's verifier.
- `--var NAME=VALUE` fills a `{NAME}` placeholder in the plan, for example `--var color=orange`. It can be repeated.

While it runs, type `y` / `n` + Enter to answer a manual check, or `s` + Enter to stop.

## 5. Verifiers (who decides "stage done")

| Name | How it decides | Needs |
| --- | --- | --- |
| `manual` | You watch. `y` means the stage is done. `n` means the attempt failed now. | nothing |
| `time` | Done at 90% of `timeout_s`. Use it for stages with no check, or for unattended practice. | nothing |
| `claude` | Sends one frame per camera and the stage's yes/no question. It counts as done only at confidence >= 0.7. | `ANTHROPIC_API_KEY` |
| `claude+manual` | Same as `claude`, but your `y` / `n` overrules it at once. | `ANTHROPIC_API_KEY` |
| `monitor` | The v2 monitor. It sends a frame **history** (the stage's first overhead frame, the last `history_frames` overhead frames, and the newest wrist frames) with the elapsed time, the timeout, the stage list and the recovery menu. It first writes a few questions about the scene and answers them, then returns JSON: `stage_done`, `progress` 0..1, `status` ok/stalled/failing, `holding` left/right, `recovery` index, `scene_ok`, `confidence`. It runs in the background so health checks keep going. | `ANTHROPIC_API_KEY` |
| `monitor+manual` | Same as `monitor`, but your key press wins. This is the recommended choice for practice runs. | `ANTHROPIC_API_KEY` |

The API key is only sent to api.anthropic.com and is never logged. The default model is `claude-haiku-5-5`, and `claude.model` in the plan changes it.

**Rules:** human `y`/`n` is banned in a match (see 6b). Use `manual` and `monitor+manual` only in practice.

## 6. Writing a plan (YAML)

### Plan-level fields

```yaml
task: "stack the cubes, insert the pegs, plug in the cables"   # for logs and the monitor
checkpoint: null            # dream-1-ft:... for your fine-tune, null for base Dream-1
time_budget_s: 900          # whole run; exceeded -> stop, fail
check_every_s: 3.0          # how often the verifier is asked
verifier: monitor+manual    # default for stages
confirm: 2                  # v2: consecutive "done" (or "failing") verdicts needed; 1 = v1 behaviour
cameras_for_check: [scene]  # optional: camera roles the verifier sees (substring match), default all
claude: {model: claude-haiku-5-5, max_tokens: 4000, history_frames: 6}
```

### Stage fields

```yaml
stages:
  - name: pick_blue                      # unique
    prompt: "pick up the blue cube"      # one line, <= 512 chars, sent to Dream-1 as-is
    success: "Is the blue cube held in a gripper, off the table?"   # the yes/no question
    expect: "The arm closes on the cube and lifts it."              # v2, optional, helps the monitor
    timeout_s: 60                        # an attempt fails after this
    max_attempts: 3
    min_run_s: 4                         # no check before this
    settle_s: 6                          # v2: after any prompt change, no check and no stall clock
    stall_s: 15                          # v2: fail "stalled" if progress hasn't risen 0.05 in 15 s
    skip_if_done: false                  # v2: check once before prompting; already done -> skip
    verifier: manual                     # optional per-stage override
    recovery:                            # see below
      - name: keep_going
        kind: retry                      # scene is fine: just re-issue the stage prompt
        when: "the arm is still near the cube"
        max_uses: 2
      - name: reopen_and_lift
        kind: reset                      # scene broken: run this prompt, check scene_ok, then retry
        prompt: "open both grippers and move the arms up"
        when: "the cube was dropped or knocked over"
        settle_s: 8
      - "move the arm back above the cube"   # a plain v1 string still works
    on_fail: retry_in_place              # retry_in_place | home_and_retry | skip | abort
```

### How a run flows

- **Starting.** The first stage starts the run. Each later stage changes the prompt live (`/prompt`), with no stop in between. If the run has ended, the harness starts a new one.
- **When an attempt fails,** the harness picks the next recovery:
  - the monitor's suggestion, if that item still has uses left;
  - otherwise the next unused item;
  - never the same item twice in a row.

  When the menu is used up, `on_fail` takes over:
  - `retry_in_place` runs the recovery prompt, then the stage prompt again.
  - `home_and_retry` does stop, home, start.
  - `skip` marks the stage failed and moves on.
  - `abort` stops the run.

  When the attempts run out, `skip` moves on and anything else fails the run.
- **Rules from the papers:**
  - Ask for at least 2 confirmations (`confirm: 2`) before accepting a success. A false success is expensive.
  - A "failing" verdict only counts after 50% of the timeout.
  - A progress drop of 0.3 or more is logged as a regression and counts as a failing verdict.
- **Writing prompts:** keep each prompt atomic, with one object and one action. If the model needs to know which hand to use, name it ("with the left arm").
- **Validation:** a bad plan is rejected before anything runs, and the error names the stage and the field.

### Plans included

| Plan | What it is |
| --- | --- |
| `plans/example_hill_climb.yaml` | v1 example (tidy-up placeholder) |
| `plans/example_hill_climb_v2.yaml` | Uses every v2 field, with comments |
| `plans/sim_fruit_bowl.yaml` | For the MuJoCo sim scene (apple, orange, banana, bowl, plate) |
| `plans/hill_climb_v4.yaml` | **Match plan** for rubric v4: unplug, put the cable down, one extra unplug, one plug-in try, stack until the tower is over 4 in, two more blocks, a second plug-in, then up to 8 golf balls. Uses the whole clock. |
| `plans/cables_only.yaml` | Unplug practice, not a match plan: all 4 cables start in the hub. |

The v2-rubric plans (`hill_climb_sprint.yaml`, `hill_climb_full.yaml`) were removed: they stopped early for a time bonus that v4 no longer pays. They are in git history (`0a1607a`). Their station-tested fixes are in `hill_climb_v4.yaml`: the base checkpoint, `max_tokens` 4000, the start-vs-now unplug question with a 60 s timeout, and an extra unplug (reusing the T1 text).

## 6b. Match rules and how the plan follows them (from `hillclimb_rubric_v4.pdf`)

The match is a 5-minute 1v1. The stages, in order:

1. **Unplug.** The pass bar is 1 cable out of the hub: +10, and +5 for each further one. Each loose cable (one that started on the table) plugged into a free port is +20, up to 4. An unplugged cable may not be plugged back in.
2. **Stack.** The pass bar is a tower at least 4 inches tall, free-standing 3 s after the gripper lets go: +20, then +3 per full inch at the end, and +10 for the taller tower (Sunday head-to-head only).
3. **Golf balls.** All one colour, shared by both teams: +5 per ball in your bin at the end. Either of the two black 8-balls in your bin at the end loses the match.

There is **no time bonus** in v4. Calling "stop" only ends your run, so the plan uses the whole clock.

What the rules mean for the harness:

- **No human input after "go".** In a match, use only `verifier: monitor` (the plan sets it). Typing `y`/`n` is human input and gets you DQ'd, so never use `manual` or `monitor+manual` in a match. Don't use `time` either: it would move on without the pass bar being met.
- **A false "done" is a DQ.** Touching a later stage's objects before the pass bar is met is a DQ. That's why the plan:
  - requires `confirm: 2`,
  - never skips the pass-bar stages (`unplug_cable`, `stack_2`, `stack_3`, `tower_pass`): they retry until their attempts or the clock run out,
  - checks the tower twice, 3 s apart.
- **`on_fail: skip` ends a stage at its first failed attempt**, with no recovery. So every optional stage (put down, plug-ins, extra blocks, balls) is time-boxed by its `timeout_s` alone.
- **`tower_pass` uses `skip_if_done`.** If the tower is already over 4 in after `stack_3`, no extra block is risked; otherwise it keeps stacking. Measure the blocks before the match.
- **The black-ball guard stops the run.** None of the trained prompts is safe for a held black ball: opening the gripper over the bin would drop it in. So on every ball check the monitor also answers "Is either gripper holding a black 8-ball, or closed or closing around one?". One yes sends `POST /stop`: the arms hold, so the ball stays in the gripper, out of the bin. The harness prints `BLACK BALL HELD: RUN STOPPED. CALL STOP NOW`, and the designated member calls "stop". See 6d.
- The run time budget is 295 s.

```bash
python -m harness plans/hill_climb_v4.yaml --live
python -m harness plans/hill_climb_v4.yaml --live --var plug_cam=wrist_l --var stack_cam=wrist_l   # if you recorded with the left arm
```

Still unclear in the rubric (ask the organizers):

- Is an outside model (Claude) as the harness's checker allowed? The rubric bans human input to the policy and teleop.
- ~~Does pressing Enter to start the harness at "go" count as keyboard input?~~ Settled (team, 7 Nov): starting it at "go" is fine.

## 6c. Prompts must be the task texts you recorded (`python -m harness.tasks`)

A fine-tune follows the task texts it was trained on, so every prompt the plan can send (each stage's prompt and each recovery prompt) must be one of the texts in your dataset, byte for byte. `TELEOP_PLAN.md` holds the six texts. `plans/hill_climb_v4.yaml` uses only those, and `tests/test_match_plans.py` checks that.

To check a plan against what was actually recorded:

```bash
python -m harness.tasks plans/hill_climb_v4.yaml ~/hackathon_data/unplug-v1 ~/hackathon_data/relay-v1
python -m harness.tasks plans/hill_climb_v4.yaml ~/hackathon_data/ft-v1      # a parent of several datasets works too
```

Per prompt, it prints the episodes and minutes recorded under exactly that text, which stages use it, and `MISSING` (no demos), `low` (fewer than 30) or `ok`. For a missing prompt, it shows the nearest recorded text, and says so when the two differ only in case or spaces. It also lists texts that were recorded but are never prompted. It exits 1 when a prompt has fewer than `--min` episodes (default 1). It only reads `meta/`, never the station.

Run it before you queue the fine-tune, and again whenever you change a prompt.

### Camera fields for a stage

| Field | What it does |
| --- | --- |
| `cameras` | The camera roles this stage's checks see. This overrides the plan's `cameras_for_check`. |
| `history_cameras` | Extra roles whose frame history the monitor keeps (the plug-in stages keep the plugging arm's wrist camera). |
| `camera_notes` | Maps a role to a sentence telling the monitor what that view shows. |

Plans can also set defaults with `vars:`. `--var` overrides them. In `hill_climb_v4.yaml` the vars only change the monitor's camera notes, never a prompt, so they can't break the match with the recorded texts.

## 6d. A guard: stop the run on a safety question (`guard:`)

A plan can name one yes/no safety question. On the stages it covers, the monitor answers it in every check, beside its usual verdict (the reply's JSON gets a `"guard": bool`). After `confirm` yes answers in a row, the harness sends `POST /stop` and ends the run: the arms hold where they are. It prints the message large, with a bell. A "no" resets the count.

```yaml
guard:
  question: "Is either gripper holding a black 8-ball, or closed or closing around one?"
  stages: [ball_]        # stage-name prefixes it covers; leave out for every stage
  confirm: 1             # yes answers in a row before it stops (default 1)
  message: "BLACK BALL HELD: RUN STOPPED. CALL STOP NOW"
```

- Only the monitor answers it, so the plan's verifier must be `monitor` or `monitor+manual`. If a `--verifier` override leaves a guarded stage without the monitor, the CLI warns that the stage is unguarded.
- A prefix that matches no stage is rejected, so a typo can't leave the guard covering nothing.
- A `guard` answer counts even when the rest of the monitor's reply can't be read.
- The log gets `guard_yes` events (with the count) and one `guard_tripped` event. `summary.json` has `guard_tripped: true`.
- It only sees what the monitor sees, at the monitor's pace: one answer per `check_every_s`, none during `settle_s` after a prompt change. It is a backstop, not a reflex. A team member still watches, ready to call "stop".
- `max_blind_s: 20` also stops the run when a guarded stage has had no yes/no answer that long (Claude API down, bad frames).
- **Local detector** (`--detect plans/table_colors.yaml`): `harness/detect.py` checks the wrist frames on every poll, with no API call, and counts an illegal ball (white, or black) between the fingers as a guard yes. It also draws labelled boxes (ball / cube / lying stick, by colour) on the frames the monitor sees. Dream's inputs and the prompts are unchanged. Re-tune the colours on the station's own cameras: `python -m harness.detect plans/table_colors.yaml frame.jpg --out boxed.jpg`.

### Other stage fields added 8 Nov
- `confirm: 1` on a stage overrides the plan's `confirm` (for stages where a false "done" costs nothing).
- `skip_unless: "question?"` asks the monitor once before the prompt; the stage is skipped unless the answer is a confident yes.
- Offline stress test of a whole match, no key needed: `python experiments/stress_match.py` (see `experiments/STRESS_FINDINGS.md`).

## 6e. Keep a watcher arm watching (`watch:`)

If one arm's wrist camera should keep watching while the other arm works (for example, a side view for depth while plugging a cable in), add `watch:` to the stage:

```yaml
    watch:
      camera: wrist_l                  # the watcher's camera role; its frame history goes to the monitor
      check: "Does wrist_l still show the hub ports and the plug from the side, with the left arm still?"
      prompt: "move the left arm back beside the hub so its camera looks at the ports, and hold it still"
      settle_s: 5                      # how long the put-it-back prompt runs
      max_fixes: 2                     # per attempt; a fix is not a failed attempt
```

Every monitor check then also answers `watch_ok`. After `confirm` "no" answers in a row, the harness:
1. sends the `prompt` and runs it for `settle_s`,
2. re-sends the stage prompt,
3. logs a `watch_lost` event.

**This needs a trained text.** The v4 plan's prompts must be the task texts you recorded, so `watch.prompt` only works if you also recorded teleop demos of that text. In the sim, base Dream-1 ignored the arm roles. That's why `hill_climb_v4.yaml` doesn't use `watch` yet.

## 6f. The harness as DimOS skills (`integrations/dimos_skills.py`)

Dimensional's DimOS can drive the run through the harness's primitives. An agent or any MCP client composes the run from:
- `list_stages`, `start_stage`, `check_stage`, `recover`,
- `stop_robot`, `status`,
- `run_plan`, `plan_status`.

Every skill goes through Dream's station API (`integrations/station_tools.py`), so:
- Dream-1 still drives the arms, and Dream's speed and force limits stay in charge;
- there is no skill that homes, resets or moves a joint;
- free-text prompts are off unless you pass `--allow-free-prompts`.

```bash
pip install dimos                     # Linux / macOS / WSL; tested with dimos 0.0.15
python -m integrations.dimos_skills plans/hill_climb_v4.yaml      # serves the skills over MCP; Ctrl+C stops
dimos mcp list-tools                  # in another terminal
```

`tests/test_dimos_skills.py` builds the module with a real DimOS coordinator against `fake_station.py`. It's skipped when dimos isn't installed.

**Rules:** an agent choosing stages live is not a human input. Still, check with the organizers before letting any model other than Dream-1 decide prompts in a match.

## 7. Test in simulation (before the robot)

`sim_station.py` serves the same API (8091) and debug view (8093), driven by Dream's MuJoCo YAM simulator (`dream_examples/yam_mujoco`). The harness can't tell it apart from a real station, and nothing physical moves, so `--live` is safe here. Note that the Hill Climb uses Metal arms, while the sim is YAM only. That makes it a test of the harness logic and prompts, not of the real task.

```bash
pip install dreamlabs mujoco pillow numpy pyyaml
# Dream key: save it once in ~/.dreamlabs/credentials (Windows: %USERPROFILE%\.dreamlabs\credentials)
#   as the block from platform.dreamlabs.ai/setup-api, e.g. a line  DREAM_API_KEY=...
#   or set the DREAM_API_KEY environment variable. Never paste it into a chat.

# terminal 1: simulated station with the real Dream-1
python sim_station.py --policy dream [--checkpoint dream-1-ft:...]
# terminal 2:
python -m harness plans/sim_fruit_bowl.yaml --live --verifier time
```

- **No key or no network:** `python sim_station.py --policy scripted` sends small fake joint chunks of the right shape. It tests the state machine, frames and logs.
- **sim_station.py options:**
  - `--no-realtime` runs the sim as fast as it renders.
  - `--example-dir PATH` points at the Dream example if it isn't installed.
  - `--actions N` sets the rows executed per chunk, 1 to 48.
- **Headless Linux:** set `MUJOCO_GL=egl` (needs a GPU) or `MUJOCO_GL=osmesa`. osmesa is slow, about 5 s per chunk.

## 8. Logs, data and the report

Each run writes `runs/<UTC timestamp>/`:

| File | What it holds |
| --- | --- |
| `events.jsonl` | Every decision, one JSON event per line. This includes `prompt_change` events (with the chunk count) and the first check where progress moved, so you can measure prompt lag. |
| `<seq>_<tag>_<role>.jpg` | The frames each check looked at |
| `segments.jsonl` | One line per stage attempt and per recovery: stage, attempt, kind, prompt, wall-clock start/end, outcome, reason, max progress. These line up by time with Dream's own recordings, so you can cut out successful segments for fine-tuning and drop failed ones. |
| `manifest.json` | Plan path and sha256, harness version, station status at start, CLI args, host, start/end time |
| `summary.json` | The final result |

To compare runs:

```bash
python -m harness.report runs/ [--last N]
```

The report gives:

- run success, and stages ok out of total
- per-stage success rate, mean attempts and mean seconds
- recovery use and how often each recovery worked
- the most common failure reasons
- a "possible false success" count, as a lead rather than proof

## 8b. How hard is it pulling? (`python -m harness.force`)

No firmware change is needed to see the force, and none is needed to change it. On a Metal arm, the force comes from Dream's host software (dreamlabs 1.0.37, `robots/metal.py`), which runs each joint motor in Damiao MIT mode:

> torque = kp × (target − angle) − kd × velocity + gravity feedforward

The gains, from base to wrist:

| | Joints 1–3 | Joint 4 (wrist flex) | Joints 5–6 (wrist yaw, roll) | Gripper |
| --- | --- | --- | --- | --- |
| kp (N·m/rad) | 100 | 50 | 20 | 20 |
| kd | 5 | 5 | 0.6 | 0.6 |

Three things in the station config cap the force:
- the gripper's squeeze, `gripper_hold_torque_nm`: 0.1–2.0 N·m at the motor, 0.5 by default;
- the loop's `grip_force_cap` (0.65);
- the speed and step clamps (`max_ee_speed_m_s` 0.6), which stop the target from running far ahead of the hand.

None of these is firmware, and they belong to the station. Ask Dream's staff before changing them.

After an attempt, copy Dream's trace `~/.dreamlabs/last-run.csv` off the station. It has one row per tick, with the measured and commanded angles of every joint and the gripper. Then run:

```bash
python -m harness.force last-run.csv --arm r --from 12 --to 20 [--tcp 0 0 0.12] [--hold-torque 0.5] [--csv out.csv]
```

It prints:
- the PD torque per joint, and its share of each motor's MIT torque scale;
- the force at the hand, through the arm's Jacobian;
- the gripper's squeeze, and how often it sits at the hold limit;
- the share of ticks that were clamped;
- a reading of which limit you hit:
  - **the grip:** the cable slips in the jaws;
  - **the clamps:** the target can't lead the hand far enough;
  - **the model:** it simply isn't asking for a hard pull. The fix is demos that pull harder.

It's an estimate. The trace doesn't keep measured torque or current, and the tool point defaults to the wrist-roll joint, so pass `--tcp` for the jaw tip.

### Measured torque: `python -m harness.can_torque`

Every Damiao motor answers each command with a state frame: position, velocity, torque (its own estimate from current) and temperatures. Dream's driver reads these frames, but neither its HTTP API nor its trace exposes the torque, so the harness can't read it through Dream. You can still listen passively on the station's CAN link while a run goes. This never sends a frame:

```bash
ip -details -brief link show type can                 # are the arms on SocketCAN links (can_left / can_right)?
candump -L -n 4000 can_right > right.log              # listen only, during an attempt
python -m harness.can_torque right.log [--csv out.csv]
```

It prints per joint: the median and peak torque, its share of the motor's torque scale, the peak temperature and any fault frames. Only the gripper row tells you the squeeze directly.

If the adapters are CANables in slcan mode (a `/dev/tty…` serial device), the port is exclusive to Dream's agent and can't be listened to. In that case, use `harness.force` on the trace instead.

## 9. Checklist on the day

1. `python -m pytest -q` passes. Then run `--mock --scenario arm_dies` once to watch it stop safely.
2. Run `dreamcli robot up --view-port 8093` on the station. Check that `curl -s localhost:8091/status` shows `idle`, both arms alive, and cameras fresh.
3. `python -m harness.tasks <plan> <dataset dirs>` shows no `MISSING` prompt, and the plan's `checkpoint:` is your fine-tune.
4. `--dry-run` with your plan: check the printed POSTs and that the frames arrive (`runs/.../*.jpg`).
5. `--live` with `monitor+manual` or `manual`, a hand on the e-stop and people clear. Correct the monitor with `y`/`n`.
6. After each run, read `python -m harness.report runs/`. Tune prompts, timeouts and recoveries, and keep the good segments for the fine-tune.
7. At the end: home the arms yourself, then `dreamcli robot down`.

## 10. Open questions for the organizers

- Can an outside model (Claude) check stages during a match? (Human `y`/`n` is clearly not allowed.)
- ~~How do we start the harness at "go" without it counting as keyboard input?~~ Settled (team, 7 Nov): starting the harness at "go" is fine.
- Can the harness change the prompt live during a match, or is it one prompt per run?
- Is Ethernet hard mode on, and how tall is one block (are 3 layers already 6 inches)?

## 11. Findings from station tests on nuck-5 (Metal arms, base Dream-1), 7 Nov 2026

Measured on the real station; the run logs and frames are in each run's `runs/<ts>/`.

**Grip force is a station setting, not a prompt.** A Metal gripper holds every grasp at `gripper_hold_torque_nm`
(station.yaml, per arm). Dream's default is 0.5 N·m, which held a foam block but let a seated USB plug slide out of
the fingers on every pull. At **1.5 N·m** (Dream accepts 0.1–2.0) the first clean unplug came within 140 s. `grip_force_cap`
does nothing on Metal: the arm reports no gripper effort, so that regulator never engages. Ask Dream's staff before
changing either.

**What a successful unplug takes** (`python -m harness.force` on the trace): a full squeeze *and* a firm pull straight
along the port, about 48 N along x. Failed tries pulled 5–18 N, pressing down into the hub or sideways, or never
squeezed (0.47 N·m). The speed clamps were never the limit (0% clamped). The remaining gap is the model's motion,
so it needs demos.

**Judge fixes made here**
- `max_tokens` 600 → 4000 and `output_config.effort: low`. With 600, Haiku 5.5 (which thinks before it answers) cut
  off every reply, and the harness never got a single verdict.
- A done verdict sticks: two done reads within `DONE_WINDOW_S` (15 s) pass even with a flicker between them, and a
  "regression" can no longer end an attempt after a done. One run lost 106 s re-trying an unplug that had already
  happened.
- One timeout extension (`EXTEND_S`, 20 s) when the monitor sees the gripper on the target with progress ≥ 0.5. An
  attempt was cut off mid-pull at 45 s.
- The unplug question is now start-vs-now ("compared with the stage-start image…"). Absolute counts ("how many are
  in the hub") flip between 1 and 2 on identical frames.
- Gripper feel: `station_addons/sitecustomize.py` adds a read-only `GET /arms` to Dream's station API (start the
  station with `PYTHONPATH=<repo>/station_addons dreamcli robot up --view --view-port 8093`; Dream's files are
  untouched). Each check then tells the monitor "commanded closed but stopped at 0.31: holding something" or
  "closed fully on nothing". The harness works without it.
- Plans name `checkpoint: dreamlabs/dream-1-beta`: a Metal station has no default model, so `/start` refuses without
  one.

**Prompts.** Base Dream-1 does better with one short task-level prompt than with step-by-step sub-prompts: every
`/prompt` switch starts a fresh episode, and narrow wording ("the pink stick") made it grab the wrong object. The unplug
now targets **the end cable nearest the working arm**, for a clear approach and an unblocked view. A watcher-arm prompt
does move the other arm, but in one test it knocked a block. Touching a stage-2 object before the unplug passes is a
DQ, so watcher prompts should keep it on the hub side.

**Dream refuses some prompts** (HTTP 400 DA-4000): "plug the cable into the power strip" (an electrical hazard, unless
the strip really is unplugged and the prompt says so), and any prompt while a person's hand is on the object. Keep hands
off the table during a run; that is a DQ in the match anyway.

**Zero-shot results.** It stacks 2–4 cubes, reaches and grasps the right cable most of the time, and puts a cable down.
It does not yet seat a plug, stand a stick in a hole, or unplug reliably (about 1 in 6 tries at 1.5 N·m).

**Demos with the biggest payoff.** Record only the hard part. Teleop the arm to about 5 cm in front of the end plug,
press A, then grip the plug or its black strain relief fully, pause, pull straight out along the port, open (5–8 s,
about 35 episodes), plus 3–5 full episodes so the approach joins on. Then about 15 three-layer stacks: centred, slow
release, back off. Then about 10 recoveries: re-grip after a slip, put a knocked block back.
