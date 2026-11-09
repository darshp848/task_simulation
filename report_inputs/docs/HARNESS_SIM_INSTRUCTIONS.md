# Running the Hill Climb harness in simulation (Windows laptop)

This runs in simulation only, using Dream's MuJoCo YAM sim (`dream_examples/yam_mujoco`) with no real robot. Don't point
`--station` or `--view` at a real station.

Folder: `C:\Users\darsh\TAMU\EMPE_Lab\Actuator-controls\hill_climb_harness_test\hill_climb_harness`
(unzipped from `hill_climb_harness.zip`; the package's own docs are `README.md` and `DESIGN.md`).

## 1. Environment (done once)

```powershell
cd C:\Users\darsh\TAMU\EMPE_Lab\Actuator-controls\hill_climb_harness_test\hill_climb_harness
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe dreamlabs mujoco pillow numpy pyyaml pytest
.venv\Scripts\Activate.ps1
```

Installed: Python 3.12.13, dreamlabs 1.0.37, dreamlabs-client 1.1.3.

## 2. Offline tests

```powershell
python -m pytest -q
```

Result on 2026-11-07: **127 passed, 1 failed.** The failure is
`tests/test_v2.py::test_segment_written_when_run_aborts`, a timing race in the test:

- The test's fake arm dies after 0.3 s, but `wait_for_state` polls every 0.5 s.
- So the harness sees the error while it is still waiting for the run to start, before `_attempt()` writes a
  segment.
- The harness still stops the station and fails the run correctly.

It doesn't affect the sim.

## 3. Dream API key

The key goes in `%USERPROFILE%\.dreamlabs\credentials` as one line, `DREAM_API_KEY=<key>`.

- To save it without the key showing on screen, run the helper in the parent folder:

  ```powershell
  .venv\Scripts\python.exe -I ..\save_dream_key.py
  ```

  It asks for the key with hidden input, writes the file and prints only the key length.
- Never paste the key into chat or into a command line.
- `sim_station.py` also accepts `DREAM_API_KEY` from the environment, or the yam_mujoco `/login` file
  `~/.config/dream-yam-sim/credentials.json`.

## 4. Windows caveat: dreamcli does not run

`dreamcli` crashes on import with `ModuleNotFoundError: No module named 'fcntl'` (`robots/arms_lock.py`, which is
Unix-only). So on this laptop:

- `dreamcli example yam-mujoco --self-test` can't be run. Skip it.
- The harness's closing message "Home them with: dreamcli robot down" doesn't apply here. It's for the real station.

`sim_station.py` loads Dream's example code directly and doesn't need `dreamcli`.

## 5. Offline sim check (no key needed)

```powershell
# terminal 1
python sim_station.py --policy scripted --api-port 8191 --view-port 8193
# terminal 2
python -m harness plans/sim_fruit_bowl.yaml --live --verifier time --station http://127.0.0.1:8191 --view http://127.0.0.1:8193 --runs-dir runs_scripted
```

Type `yes` at the prompt. It's a simulator, so nothing physical moves.

Result on 2026-11-07: `SUCCESS: completed (168.2s)`.

| Stage | Result |
| --- | --- |
| apple_in_bowl | ok, 57.4 s |
| orange_on_plate | ok, 55.1 s |
| banana_in_bowl | ok, 55.2 s |

- 153 frames were saved in `runs_scripted\20261107T201447Z`.
- Final station status: 1 run, 52 chunks, 0 stalls, no errors.
- The prompt changed live between stages without a restart.

## 6. Real Dream-1 run (needs the key)

```powershell
# terminal 1: simulated station with the real Dream-1 model, real time; keep the log
python sim_station.py --policy dream *> sim_dream.log
#   optional: --checkpoint dream-1-ft:...   --example-dir PATH\to\dream_examples\yam_mujoco   --no-realtime

# wait until this answers:
Invoke-RestMethod http://127.0.0.1:8091/status

# terminal 2: the harness
python -m harness plans/sim_fruit_bowl.yaml --live --verifier time --station http://127.0.0.1:8091 --view http://127.0.0.1:8093
```

- Type `yes` at the confirmation.
- The first model call can wait several minutes while the model warms up (`pool_warming`). That's expected.
- The run is capped by `time_budget_s: 600` in the plan.
- **Stop:** `s` + Enter or Ctrl+C in the harness terminal. This sends `/stop` and the arms hold.
- **When done:** Ctrl+C in the `sim_station` terminal so nothing is left running.

Verifier options (`--verifier`):

| Option | What it does |
| --- | --- |
| `manual` | The default. You answer each stage with `y` / `n` + Enter. |
| `time` | Runs unattended. A stage counts as done at 90% of its `timeout_s`. |
| `claude` | Needs `ANTHROPIC_API_KEY`. Check the event rules before using it. |
| `claude+manual` | Claude judges each check; your `y` / `n` overrules it. |

### Result on 2026-11-07 (run `runs\20261107T203122Z`, model `dreamlabs/dream-1-beta`)

**Harness result:** `SUCCESS: completed (172.1s)`.

| Stage | Result |
| --- | --- |
| apple_in_bowl | ok, 62.1 s |
| orange_on_plate | ok, 54.8 s |
| banana_in_bowl | ok, 54.9 s |

All three "ok"s come from the `time` verifier, not from a check of the scene.

**Dream-1 output:**
- It returned action chunks: 33 in 181 s, about one every 5 s, with 0 stalls and 32 clamps.
- The first chunk arrived within about 10 s of `/start`. There was no `pool_warming` wait.
- The only sim_station message was a warning that `python -m dream.check` (the camera-input check) hasn't been run.

**Prompt changes:** they went through live with `/prompt`, at chunk 12 and chunk 22, in one run with no restart.

**What the arms actually did** (from the overhead frames):

| Stage | Arms | Task done? |
| --- | --- | --- |
| apple_in_bowl | The left arm reached to the apple and lifted around it near the bottle and left bowl. The right arm stayed parked. | No. The apple was still on the table at the end. |
| orange_on_plate | The right arm came alive and swept toward the orange and banana. | No. The orange didn't reach the plate. |
| banana_in_bowl | The right arm kept working at the banana/orange area. | No. The banana stayed on the table. |

The prompts clearly changed which arm moved, but none of the three placements finished in 60 s. Both plates render
white, not blue.

**Takeaway:** use `--verifier manual` (or `claude`) to judge real success. `time` only shows that the run kept going.

### Watcher-arm test, 2026-11-07 (`plans/sim_watcher_test.yaml`, run `runs\20261107T204020Z`)

**Idea:** one arm parks its wrist camera to give a side view (depth the overhead camera lacks) while the other arm
does the task. The plan sets `cameras_for_check: [overhead, wrist_l]`, so the verifier sees only those two cameras.

**Harness side: works.** Only overhead and wrist_l frames were saved, and wrist_r was left out. The run had
24 chunks, no errors, and the harness reported `SUCCESS: completed (126.3s)` from the `time` verifier.

**Dream-1 side: didn't follow the arm roles.**

| Stage | Prompt asked for | What the arm did |
| --- | --- | --- |
| watcher_pose | The left arm looks at the apple from the side and holds still | It reached down to the apple as if to grasp it. Its wrist camera ended up looking down at the apple from close range, not from the side. |
| right_arm_task | The left arm stays still while the right arm puts the apple in the bowl | The left arm lifted and moved toward the bottle and bowl, and the apple left its view. The right arm went to the orange and banana, not the apple. The apple wasn't moved to the bowl. |

The left wrist camera's view in the home pose (frame 0001) is a usable low side-on view of the table.

**Takeaway:** the harness can choose which cameras judge a stage. It can't make Dream-1 hold an arm still: the
harness never commands joints, so arm roles exist only as words in the prompt.

## 7. What to look at afterwards

`runs\<UTC timestamp>\` holds:

| File | Contents |
| --- | --- |
| `events.jsonl` | Every decision, one JSON event per line. Look for `prompt_change`, `verdict` and `stop`. |
| `segments.jsonl` | One line per stage attempt or recovery. |
| `<seq>_<stage>_a<attempt>_<camera>.jpg` | The frames the verifier saw. Cameras are `overhead`, `wrist_l`, `wrist_r`. |
| `summary.json` | Overall ok, the per-stage results and the time taken. |

Also check `sim_dream.log`, which has sim_station's chunk count and any model errors. `GET /status` on 8091 shows
`chunks`, `stalls`, `clamps` and `last_error`. The debug view on 8093 has `/trips` and `/trip/<n>`.
