# task_simulation: testing the Hill Climb harness before the robot

This repo is everything needed to test and improve the Hill Climb relay harness (Physical AI Hacks, 7-8 Nov 2026)
without the real station: the harness itself, two simulators, the scenarios, the cluster scripts, the station
data they are based on, and the results so far.

The harness (`harness/`) supervises Dream-1 on the MakerMods Metal station: it sends one recorded task prompt per
stage, asks a Claude vision monitor whether the stage is done, retries, recovers, and stops on an illegal ball. It is
a copy of [hill-climb-harness](https://github.com/darshp848/hill-climb-harness) `main` at `d938644`; port changes
back there (or the other way) by hand, the two repos don't sync themselves.

```
harness/            the harness (runner, plan parser, Claude monitor, station client, local detector)
plans/              hill_climb_v4.yaml (the match plan), table_colors.yaml (detector colours), other plans
experiments/        stress_match.py (offline sim), sim_match.py (MuJoCo matches), dual_arm_bench.py, STRESS_FINDINGS.md
sim_station.py      MuJoCo station with real Dream-1;  sim_scenes/hill_climb_table.py  the relay table + referee
fake_station.py     tiny HTTP stand-in for the station (unit tests, --mock)
cluster/            Dockerfile, run_matches.sh (MuJoCo), run_stress.sh (offline sweep)
tests/              189 tests; test_stress.py = whole-match scenarios, test_detect.py = detector on a real photo
data/               what the station logged on 7 Nov (inference log, last run trace, config), teleop dataset metadata
results/            stress sim tables (before / after the fixes) and the cluster's MuJoCo match logs (no images)
HARNESS_GUIDE.md    full harness manual;  HARNESS_README.md, DESIGN*.md, TELEOP_PLAN.md  harness design notes
```

## Setup

Python 3.12 or 3.13.

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pillow pyyaml scipy pytest
python -m pytest -q                                  # ~3 min; 4 tests skip without the dreamlabs package
```

The MuJoCo sim also needs `pip install "dreamlabs==1.0.37" mujoco` and a Dream key in `~/.dreamlabs/credentials`
(a line `DREAM_API_KEY=...`). The real Claude monitor needs `ANTHROPIC_API_KEY` in the environment. Neither key is
needed for the offline sim or the tests. **Never commit a key**: `.gitignore` covers `credentials` and `.env`, and
this repo is public.

## The two simulators

| | Offline stress sim | MuJoCo sim |
|---|---|---|
| Run | `python experiments/stress_match.py` | `python sim_station.py --policy dream --scene hill_climb` + `python experiments/sim_match.py` |
| Policy | rule-based table model (times and probabilities) | real Dream-1 over the Dream API |
| Monitor | scripted: answers from the table state at the frame's time | oracle: reads object positions |
| Clock | virtual, a 5-minute match in ~5 s | real time, ~5 min per match |
| Keys | none | Dream key |
| Tests | the harness's decisions under faults | policy + harness on a physical table |
| Blind spot | no physics, monitor never misjudges unless told to | base Dream-1 never gets past the first unplug |

Neither one tests what the real Claude monitor sees in camera frames. See "Next steps".

### Offline stress sim (`experiments/stress_match.py`)

The real `Runner` and the real `MonitorVerifier` run unchanged; the code around them is scripted:

- **VClock**: the harness's sleeps advance a virtual clock. A monitor call returns only after its virtual latency
  (default 4 s), so timing races between checks, verdicts and the policy are real.
- **Table + Behaviour**: cables in and out of the hub, what each gripper holds, the tower, balls in the bin. Each
  task prompt (T1-T6 of `TELEOP_PLAN.md`) has a try time and a success chance (`Behaviour` dataclass). Switches:
  `releases` / `open_frees` (the policy won't let go of the plug), `black_p` (picks an illegal ball),
  `reach_across_p` (brushes a block during unplug = DQ), `knock_p`, `unplug_never`.
- **SimStation**: the station HTTP API in-process, with faults (`Fault(at, kind)`, `refuse={prompt substring: n}`)
  that use the exact error types and messages from `data/station_nuc_20261107/robots-inference.jsonl`: DA-4000
  refusals, ControlLoopError (empty action buffer), LinAlgError (SVD), gateway DA-5000, a dead arm.
- **SimFrames**: frames carry their capture time; freeze, empty, corrupt (non-JPEG) and lag (`frame_lag_s`).
- **Scripted**: replaces only the HTTP call to Claude. It finds the stage in the real prompt the harness built and
  answers from the table **as it was at the newest frame's time**. Options: `false_done`, API `outages`, 400 on a
  corrupt image, `fast_guard` (stands in for the local detector).
- **referee()**: rubric v4 points, DQ (a block touched before a cable is out, a ball before the tower passes), an
  illegal ball in the bin = loss.

```bash
python experiments/stress_match.py --list                     # scenarios
python experiments/stress_match.py --seeds 4 -j 4              # all scenarios, 4 seeds, 4 processes
python experiments/stress_match.py -s no_release --seeds 1 -v  # one match, event by event
python experiments/stress_match.py --seeds 20 -j 7 --out runs.jsonl   # one JSON line per match
```

**Add a scenario**: append a `Scenario(...)` to `SCENARIOS` at the bottom of `stress_match.py` (behaviour, faults,
outages, camera spans, latency, or another `plan=`). If it guards a fix, add an assertion to `tests/test_stress.py`.

### MuJoCo sim (`sim_station.py`, `experiments/sim_match.py`)

`sim_station.py --scene hill_climb` serves the station API on Dream's YAM MuJoCo simulator with the relay table
from `sim_scenes/hill_climb_table.py` (hub in front of the right arm, as the plan's prompts assume).
`sim_match.py` runs the unchanged harness and plan against it; its verifier is an oracle over `/objects`, and the
referee scores each match. Layout coordinates are estimates from photos.

## On the cluster

The home cluster's head node (Inspiron: 8 threads, Intel iGPU, no CUDA) holds the image and the shared folder
`/cluster/shared/dev/darsh/projects/hill_climb_sim` (repo checkout in `repo/`, results in `runs/`).

```bash
docker build -t hill-climb-sim -f cluster/Dockerfile cluster          # once per node
PLAN=plans/hill_climb_v4.yaml cluster/run_matches.sh 3 2 my_label      # MuJoCo: 3 workers x 2 matches, real Dream-1
cluster/run_stress.sh 50 7 sweep1                                     # offline: all scenarios x 50 seeds, 7 cores
SEED0=51 cluster/run_stress.sh 50 2 sweep1 happy black_ball           # another node: seeds 51-100
```

`run_matches.sh` gives each worker its own container (own ports), mounts the Dream key read-only from
`~/.dreamlabs/credentials`, and staggers start-up by 20 s. About 3 MuJoCo workers fit on the Inspiron. The offline
sweep is pure CPU (~5 s per match), so the small worker nodes can take seed ranges too.

## Results so far

Offline sim, 4 seeds each, points (full tables in `results/stress/`, write-up in `experiments/STRESS_FINDINGS.md`):

| Scenario | main before fixes (3055bf3) | main now (d938644) |
|---|---|---|
| happy path | 87 74 90 116 | 83 94 95 95 |
| Dream refusal on the first start | 0 0 0 0, match ended | 83 94 95 95 |
| empty action buffer mid-run | 10 10 15 15, ended | 79 94 95 100 |
| SVD did not converge | 10 0 10 10, ended | 83 84 95 95 |
| policy keeps the unplugged plug | plugged it back in 4/4 | never |
| 12 s of corrupt frames | 20 15 20 20 | 84-90 |
| arm dies | ended, then sent /start | ended, no /start |
| illegal ball picked 30% of tries (6 seeds, frames 2 s old) | lost 4/6 (monitor guard) | lost 2/6 with the local detector guard |
| unplug brushes a block | DQ 4/4 | DQ 4/4: only the policy or the layout can fix this |

MuJoCo, base Dream-1, 25 matches on the cluster (`results/cluster_mujoco/`): a cable out in 14 of 25 (time to
first unplug median 57 s, 28-269 s); 3 DQs, all with the hub on the left; never past the unplug, because the policy
keeps holding the plug.

Real station, 7 Nov (`data/station_nuc_20261107/`): 42 policy runs, 10 errors (DA-4000 refusals x4, SVD x3, empty
buffer x2, session expired x1); run outcomes were not logged. Human teleop: unplug median 8.5 s (15 demos), put-down
4.8 s (9 demos).

## Classifiers

- **Stage monitor and guard**: Claude `claude-haiku-5-5` (set on the `claude:` line of `plans/hill_climb_v4.yaml`),
  `max_tokens` 4000. It gets the stage-start frame, the last 6 overhead frames and the newest wrist frames, and
  answers JSON (done, progress, status, recovery, holding, guard).
- **Local detector** (`--detect plans/table_colors.yaml`, `harness/detect.py`): no ML. HSV thresholds, blobs,
  and a shape test (ball / cube / lying stick). It answers the illegal-ball guard on every poll from the wrist
  cameras and draws labelled boxes on the monitor's frames. On our table the **white** golf ball is illegal and the
  coloured ones are legal. Colours were tuned on a phone photo; re-check them on the station cameras:
  `python -m harness.detect plans/table_colors.yaml frame.jpg --out boxed.jpg`.
- Dream-1 drives the arms; neither classifier changes its inputs or its prompts (prompts must match the recorded
  task texts word for word, see `TELEOP_PLAN.md`).

## Next steps

1. **Real monitor in the loop**: with `ANTHROPIC_API_KEY` on the cluster, run `sim_match.py` with `MonitorVerifier`
   instead of the oracle and compare its verdicts with the referee's ground truth (false and late "done" rates).
2. **Seed sweeps**: `run_stress.sh` with 50+ seeds per scenario, then sweep plan knobs (`timeout_s`, `confirm`,
   `check_every_s`, `max_blind_s`, number of `higher_*` blocks) against points and loss rate.
3. **Real numbers into `Behaviour`**: run station tests through the harness (its logs record success per stage) and
   replace the assumed try times and success chances.
4. **Detector on station frames**: tune `table_colors.yaml` on overhead and wrist frames, with the gripper open and
   closed around a white ball.
5. Port anything that changes `harness/` or `plans/` back to hill-climb-harness.
