# Fine-tune diagnosis: `dream-1-ft:bb4b9e5f-relay-final` (8 Nov 2026)

Job `ftj_fdff469db8e2b643`, dataset `ds_5ae1c3b4e8739af1` (`muhammad.mukthar11/relay-final`), 170 episodes, 48,413 frames.
Hyperparameters from the watcher log: 1000 steps, lr 1e-5, batch 32, `use_eef: true`, constant schedule. Final loss 0.0034.

## Bottom line

The fine-tune is confused mainly because its training data holds two incompatible mappings from state to action.
The 73 Quest episodes were recorded with the arm sagging under gravity: their action sits about 5.5 cm above the
measured tool. The 97 leader episodes and every live run have gravity feed-forward, so command and arm agree.

On the table this looks like the arm stopping high. In the fine-tune's unplug runs the tool never gets below 8-13 cm
while reaching forward, and the gripper closes at 12-16 cm. The demos grasp at 4-6 cm.

A corrected fine-tune will not be ready by 15:00 PT. For the demo, use base Dream-1. In the offline replay no
env-var-only fix gets the fine-tune to grasp height (see "Sim comparison").

## Ranked causes

| # | Cause | Evidence strength | Fix without retraining? |
|---|---|---|---|
| 1 | Quest actions carry a gravity-sag offset (about 5.5 cm, 10 deg) that the leader data and the live station do not | Strong: measured directly; heights match | Partly: fix 2 |
| 2 | Almost no demos start from home: 9 of 69 unplugs start within 0.3 rad of it | Solid as a fact, effect on behaviour unproven | No |
| 3 | Stack and balls are thin and left-armed; prompts outside the trained set | Solid for prompts and counts | Yes: base Dream-1 for those stages |
| 4 | Left arm parked somewhere else in 17 Quest episodes | Moderate, likely minor | No |
| 5 | 20 Hz playback and extra tracking lag (station-tuning lead) | Real, but too small to explain the height error | Yes, optional A/B |
| 6 | One bad action jump (Quest ep 36) | Minor | Drop it on any retrain |

Checked and found clean: camera keys, swaps, rotation, colour, resolution, camera position, joint order and units,
gripper scale, TCP, schema across the 10 folders. See "Ruled out".

## Cause 1: gravity sag in the Quest actions

Mean action minus state, right arm, on slow frames only (joint speed under 0.15 rad/s), so it is a steady-state offset
and not lag:

| Folder | Source | Frames | Joint 2 | Joint 3 | Joint 4 | Tool offset x / y / z (mm) | Median norm (mm) |
|---|---|---|---|---|---|---|---|
| quest-split-v4-final-eef | Quest | 2137 | +0.078 | +0.047 | +0.043 | +18 / +3 / +54 | 55.9 |
| unplug-split-v1-final-eef | Quest | 96 | +0.093 | +0.030 | +0.004 | +9 / 0 / +37 | 37.2 |
| unplug-v1-split-eef | Quest | 152 | +0.086 | +0.039 | +0.033 | +16 / +3 / +48 | 50.9 |
| leader-split-v1-frozen-eef | Leader | 1121 | +0.007 | +0.008 | +0.002 | +3 / +1 / +6 | 8.0 |
| leader-split-v2-frozen-eef | Leader | 467 | +0.010 | +0.010 | +0.013 | +4 / 0 / +7 | 6.1 |
| leader-split-v4-final-frozen-eef | Leader | 4015 | -0.002 | +0.005 | +0.001 | +1 / 0 / +1 | 5.5 |
| leader-split-v6-frozen-eef | Leader | 2310 | -0.006 | +0.003 | +0.012 | +1 / +1 / +1 | 7.8 |
| plugin-split-v1-final-frozen-eef | Leader | 2987 | -0.014 | +0.003 | -0.002 | -2 / +3 / -5 | 7.6 |

The 4-6 degree shoulder figure matches the note at the top of `follower_code/metal/leader_collect.py`: the MetalFollower
sends zero torque, so each joint sits below its target by gravity over kp. The leader path adds `FollowerGravity`.
Dream's station driver also sends gravity torque. Live command minus measured on slow frames is at most 0.016 rad.

Quest datasets also show a median rotation error of 9-10 deg between action and state. Leader datasets show 2-3 deg.

Mix of the two conventions in the data:

| Task | Quest episodes | Leader episodes |
|---|---|---|
| Unplug | 36 | 33 |
| Put down | 37 | 18 |
| Plug in, stack, balls, cylinder | 0 | 46 |

The mechanism: in a Quest-style episode the model learns that holding the arm at height z takes a command at
z + 5.5 cm. Live, that command is reached, so the arm rises and is asked to rise again. Leader episodes say the
opposite for the same scene and state.

### What the live runs do

Tool position by FK from measured joints (FK matches the dataset's EEF columns to under 0.001 mm).

Where demos grasp (first frame the right gripper drops under 0.35):

| Source | n | State xyz (m) | Action xyz (m) |
|---|---|---|---|
| Leader | 28 | 0.381, 0.162, 0.043 | 0.382, 0.163, 0.049 |
| Quest | 32 | 0.369, 0.135, 0.059 | 0.387, 0.137, 0.110 |

Live unplug prompt, tool height while forward (x > 0.25 m):

| Run (rollout) | Model | Min z (m) | Median z (m) | Right gripper min |
|---|---|---|---|---|
| 095731 | base | 0.060 | 0.064 | 0.03 |
| 095854 | fine-tune | 0.083 | 0.157 | 0.03 |
| 100247 | fine-tune | 0.099 | 0.130 | 0.15 |
| 100709 | fine-tune | 0.121 | 0.162 | 0.33 |
| 101053 | fine-tune | 0.126 | 0.161 | 0.37 |
| 101617 | fine-tune | 0.128 | 0.152 | 0.95 |
| 101716 | fine-tune | 0.132 | 0.166 | 0.41 |
| 101840 | fine-tune | 0.115 | 0.150 | 0.22 |
| 102040 (long no-power prompt, not a trained text) | fine-tune | 0.102 | 0.137 | 0.18 |

The fine-tune closes the gripper at 11-16 cm in the runs where it closes at all (`last-run.csv`). That is above the
Quest action height (11 cm) and well above the demo grasp height (4-6 cm). Its y is on target (0.13-0.22), so the
approach direction is right and the height is wrong.

Base Dream-1 on 7 Nov, same station, same 20 Hz, same driver, reached 1-6 cm on most unplug attempts, with some at
10 cm. Today's three base runs reached 6.0, 4.0 and 0 cm. (An earlier version of this report listed run 102040 as base.
The station log shows that run started on the fine-tune at 10:18:59.) See "Base vs fine-tune" below.

Effect not measured: I did not run the policy against held-out frames, so the link from offset to behaviour is an
inference from the numbers above. Fix 2 tests it directly.

## Cause 2: start poses

Live runs always start from home. Most demos do not. Starting joint distance from the live home (L2, rad):

| Task | Arm | n | Median | Within 0.3 rad of home |
|---|---|---|---|---|
| Unplug | Right | 69 | 2.14 | 9 (7 Quest, 2 leader) |
| Put down | Right | 55 | 2.23 | 0 |
| Plug in | Right | 19 | 2.14 | 0 |
| Block | Left | 4 | 1.92 | 0 |
| Ball | Left 11, right 1 | 12 | 1.85 | 2 |

The plan asked for about 1 in 5 from home. The split tools cut at the grasp, so the median unplug starts near the
plug with the arm already forward. Dream's data in the batch (`base_share` default 0.5) should cover general approach,
and the runs do reach the right x and y, so this is less damaging than cause 1.

## Cause 3: task mix and prompts

| Prompt in tasks.jsonl (exact, across all 10 folders) | Episodes |
|---|---|
| unplug the USB cable at the end of the hub nearest the right arm | 69 |
| put the held cable down on the table away from the hub | 55 |
| plug the loose USB cable on the table into a free port on the hub | 19 |
| put a golf ball in the bin | 12 |
| stand the cylinder upright in the hole of the top block | 10 |
| put a block on top of the stack | 4 |
| put a cylinder upright on top of the stack | 1 |

No near-duplicates apart from the two cylinder wordings. The unplug, put-down, block and ball prompts in the plan match word for word.

Two things sent live or planned are not in the trained set:

- `open both grippers and move the arms up`: the recovery prompt, used in runs 3 and 5. TELEOP_PLAN expects this and
  leaves it to base skill, which a fine-tune checkpoint does not switch to.
- `tower_prompts` in `plans/hill_climb_v4.yaml` on main: `stand a stick upright on top of the stack`. The plan's own
  comment says to delete this block for a fine-tune. Main (b1f6e4b) now defaults to the fine-tune and still has it.

All four block demos are left-arm. In run 5's block phase the right arm moved (travel 10.8 vs 1.6) and stayed at 25 cm.
In left-arm demos the right gripper is held closed (0.004-0.04) and the right wrist video is a frozen tilted still,
neither of which the live station shows.

## Cause 4: left arm in other poses

Live home for the left arm is `[0.144, 0.002, 0.009, -0.026, 0.215, -0.030]`. Left arm start distance from it:
quest-split-v4-final episodes 4-11 at 4.4 rad, 12-17 at 2.1 rad, and unplug-v1-split 0-2 at 2.5-2.7 rad.
That is 17 of 73 Quest episodes. The arm does not move within those episodes. The left wrist camera shows a very
different view in them (see the frames).

## Cause 5: control rate and lag (the station-tuning lead)

- **Control rate confirmed.** `robots.control` logs "20 Hz ticks, 250 Hz commands, 48-row chunks", the configured
  default for Metal is 20, and live tick spacing is a median of exactly 50 ms. Training is at 30 fps. Dream's config
  comment calls this deliberate: "20 Hz plays every plan at two thirds speed".
- **Lag confirmed but moderate.** Cross-correlating joint velocity gives a live command-to-measured lag of 160-200 ms
  and about 83-117 ms between action and state in the recordings. The live extra is about 80 ms.
- **Filter.** `FF_ALPHA = 0.04` in `robots/damiao.py`. I did not determine the feed-forward setting on the recording
  path (the kit's MetalFollower), so I can only report the measured lag difference.
- **Why it ranks below cause 1.** At the arm's typical speeds under 0.1 m/s, 100 ms is about 1 cm, and geometry does
  not change with playback rate. Observation history is timed by wall clock (`dream/_window.py`, 5 fps), so 20 Hz does
  not stretch it. Base Dream-1 runs under the same settings and has gone lower.
- Parquet timestamps are exactly 1/30 s because the kit writes frame_index/fps, so they cannot show the real loop rate.

`control_hz: 30` and `PAH_FF_TAU_S=0.15` are reasonable to A/B later. Do not change them together with fix 2, or the
result will not say which one helped.

## Cause 6: demo quality

- One real jump: quest-split-v4-final episode 36 (unplug), frame 245 of 297, action jumps 257 mm in one frame. Drop it.
- About 20 other Quest episodes have 30-52 mm action steps (up to 1.5 m/s) with the joint speed capped at 0.05 rad per
  frame. Worth a look, not urgent.
- Idle segments are short. No episode has more than 4.1 s still in a row.
- Duplicate consecutive state frames are 5-23% of leader frames and 1-3% of Quest frames. I did not separate idle ones.
- All 170 episodes are between 1.8 and 46 s. None are empty.

## Ruled out

| Check | Result |
|---|---|
| Camera keys | Dataset `scene`, `left_wrist`, `right_wrist`, 640x360, 30 fps. Live frames map correctly (see images). |
| Camera swap | Scene views and wrist views match the right sides in every sampled session. |
| Rotation and colour | `rotate: 0` live. Frames match by eye; mean colour within normal lighting variation. |
| Resolution | Live sends 640x360 from 1280x720, same aspect as training. |
| Overhead camera moved | Under 2 px shift between any session and today (phase correlation). |
| Schema | One feature signature across all 10 folders. |
| Joint order, sign | Same names and matching ranges. |
| Units | Radians. Gripper 0-100 in state, 0-1 in `grip`, on both sides. |
| TCP | `add_eef` FK matches the URDF with a 0.135 m tool, under 0.001 mm. |
| Action space | Absolute tool pose of the commanded joints. Same as the live `eef` head. |

Not fully ruled out: table clutter. Today's scenes have a headset, jar and cards near the hub in some runs, which the
demos do not have.

## What to do

### Before the demo (no retraining)

1. **Default the demo to base Dream-1** with `--var model=dreamlabs/dream-1-beta`. Main currently defaults to the fine-tune.
2. **Only if there is a slot, one supervised A/B of a joint-offset patch** (see "Sim comparison"). Gravity scale alone
   does not reach the bar in the replay, so `PAH_GRAVITY_SCALE` by itself is not worth the slot. A person must be at
   the e-stop. This is untested on the arm.
3. If the fine-tune is used at all, use it for the cable stages only and keep base Dream-1 for stack, balls and the
   `open both grippers` recovery. Delete `tower_prompts` if the fine-tune stacks.

### After the demo, or if there is time: a corrected fine-tune

Timing from the watcher log: queued 23:00, training 23:20 to 05:56 (0.4 min per step), saving until 07:01, warm at 07:14.
500 steps (the minimum) comes to about 5 h from queueing, roughly $100 going by the plan's $200 for 1000 steps.
**Queued now it lands after 15:00 PT.**
`resume_from` also has a 500-step minimum and starts from the biased weights, so it does not save time.

Recipe:

1. For the three Quest folders, recompute the arm action as the measured joints 3 frames later (keep the commanded gripper), then rerun `tools/add_eef.py --replace`.
2. Drop quest-split-v4-final episode 36. Consider dropping episodes 4-17 and unplug-v1-split 0-2.
3. Add unplug demos from the home pose.
4. Consider `lr_schedule: cosine`, which the SDK docs recommend for specialisation.

## Sim comparison: fixes for the fine-tune aiming high (offline, 8 Nov)

Offline only: a MuJoCo model of one Metal arm (Dream's link table, Dream's kp and kd, a 30 Hz bus, gravity and velocity
feed-forward as in `robots/damiao.py`), replaying the saved `commanded_r` rows from the 7 fine-tune unplug runs.

**Calibration.** On Quest slow frames, measured sag over URDF gravity gives an effective load of 1.18 / 1.30 / 1.48
times URDF on joints 2 / 3 / 4 with kp 100 / 100 / 50. The live with-feed-forward residuals independently suggest
about 1.07 / 1.40 / 1.46. The model uses the Quest values. This is why a sag of 5.5 cm comes out here, against the
smaller numbers from an unfitted model (no extra load, measured at the flange).

**Validation.** Replaying Quest actions with no feed-forward reproduces the recorded tool height with a median RMS of
10 mm and a mean offset of +1.7 mm (9 episodes). Replaying live commands with feed-forward matches the measured live
height with 5-6 mm RMS and a mean offset of -2 to 0 mm (4 runs).

Fixes are applied to the commands. Corrections fade in over tool x 0.20-0.25 m so the home pose is unchanged.
Below, "grasp z" is the median height at the 17 gripper-close events during the unplug prompt, and "reach" is while x > 0.25 m.
The table is about z = 0 (lowest tool z in any demo: 0.03 cm).

| Fix | Needs | Grasp z (cm) | Reach median z (cm) | Reach min z, worst run (cm) | Passes 6 cm grasp bar | Hits table? |
|---|---|---|---|---|---|---|
| Baseline | | 13.1 | 15.2 | 8.6 | no | no |
| (a) gravity 0.5 | env var | 10.8 | 13.3 | 6.4 | no | no |
| (a) gravity 0 | env var | 8.5 | 11.5 | 4.3 | no | no |
| (b) joints x0.43 (mix) | code | 10.9 | 13.0 | 6.5 | no | no |
| (b) joints x1 (full Quest offset) | code | 7.9 | 10.2 | 3.9 | no | no |
| (b) joints x1.5 (live-implied) | code | 5.5 | 7.9 | 1.8 | **yes** | no, 1.8 cm margin |
| (c) z -2.4 cm | code | 10.8 | 12.9 | 6.3 | no | no |
| (c) z -5.5 cm | code | 7.8 | 9.3 | -3.1 | no | yes |
| (c) z -8 cm (live-implied) | code | 5.2 | 4.8 | -7.3 | yes | yes |
| (d) gravity 0.5 + joints x0.43 | env var + code | 8.6 | 11.3 | 4.5 | no | no |
| (d) gravity 0.5 + z -2.4 | env var + code | 8.5 | 11.1 | 4.1 | no | no |
| (d) gravity 0 + z -2.4 | env var + code | 6.4 | 9.1 | 2.1 | no (close) | no |
| (d) gravity 0.5 + z -5.5 | env var + code | 5.6 | 7.5 | -4.1 | yes | yes |

Joint limits: joint-offset fixes clip 27-191 of 13,595 ticks. Cartesian z-offsets clip 971-4,138 ticks, because the
Cartesian shift pushes the arm into its limits.
Other arm: a gravity scale is global, so the parked left arm also droops, statically by about 2.1 cm at 0.5 and
3.5 cm at 0, up from 0.65 cm. The joint and z fixes can be right-arm only.

**Recommendation.** No env-var-only fix passes. The only fix that passes without going through the table is a joint
offset of about 1.5 times the Quest sag, on the right arm, faded in by reach. It needs a code change: about 20 lines
in `station_addons/pah_tuning.py`, a goal offset in `DamiaoChain.send_action`, opt-in by env var, plus a test. Not built.

Worth one supervised try only if there is a free slot and base Dream-1 stays the demo default. Start at x1.0, then
x1.5, right arm only, unplug only, e-stop manned.

Fidelity limits, in order of weight:

- **Open loop.** It replays recorded commands. The live policy would react to the lower arm, and its images and state would change.
- **One offset for two kinds of output.** Leader-style rows get corrected as well, though they need no correction, so
  those rows would go about 8 cm too low. The worst-case margin above may be optimistic.
- **No table, plug or cable contact in the sim.** "Hits table" is only a z check.
- The grasp bar is a height check, not an unplug. It says nothing about whether the plug comes out.

## Base vs fine-tune

From the saved station logs (no Dream calls). Unplug or cable-prompt runs only. Run outcomes are not logged, so
the success rows use earlier estimates or simulation.

| | Fine-tune today (trained text) | Fine-tune today (long text) | Base today | Base 7 Nov |
|---|---|---|---|---|
| Runs | 7 | 1 | 3 | 5 |
| Tool min z on cable prompt (cm), median (range) | 11.7 (8.3-12.8) | 10.0 | 4.0 (0-6.0) | 2.1 (0.8-5.6) |
| First gripper close while forward: z (cm), median | 15.8 (n=5) | 12.8 | 28.6 (n=3, early closes) | 7.8 (n=5) |
| Time to first close (s), median | 41 | 17 | 6 | 13 |
| Dream round trip (ms), p50 / p90 | 700 / 857 (n=441) | 700 / 917 (n=58) | 750 / 1603 (n=337) | 890 / 2135 (n=551) |
| Stalls in these runs | 1 | 0 | 5 | 12 |
| Dream refusals (DA-4000) in these runs | 0 | 0 | 0 | 0 |

Success, where anything is known:

| Task | Fine-tune | Base Dream-1 |
|---|---|---|
| Unplug, station | No run reached grasp height. Outcomes not logged. | About 1 in 6 (TELEOP_PLAN estimate) |
| Unplug, MuJoCo YAM sim | Not run | 14 of 25 (first unplug median 57 s) |
| Put-down, stack, balls | Not tested beyond a few seconds | MuJoCo: put-down 0 of 54 tries, stack 0 of 25 |

Base reaches table height and the fine-tune does not. The first-close row is noisy for base because it sometimes closes
early, up high. The whole log also holds 18 DA-4000 refusals, none in these archived runs.

**A fair sim A/B cannot run today with the sim we have.** `sim_station.py` drives Dream's YAM simulator, and the
fine-tune is a Metal embodiment. It would need a Metal scene with cameras (the arm dynamics model above has neither scene nor cameras).
It would also need Muhammad's key on the machine running it, typed in by him. Nothing read or copied that key.

**Cost if built, or as a station A/B instead:** from the logs, about 0.55-0.6 Dream calls per second of run. Ten matches
per model with a 120 s unplug cap is about 1,400 calls in total (about 3,500 at a full 295 s). The docs and logs give
training prices only, so the price per inference call is unknown. No Dream calls have been made for this.

## Files

- `FT_DIAG_frames_live_vs_train_1.png`, `FT_DIAG_frames_train_2.png`: frames by camera, live against sessions.
- `ft_diag/`: copied parquet and meta, 30 sample videos, live frames, rollout logs, finetune watcher log.

## How this was done

Everything on the NUC was read over SSH: dataset files, logs, `tmux capture-pane` on the finetune session, and copies
to this laptop. No robot commands, no station API calls that act, no Dream API calls.

One deviation: a shell command wrote a small file list to `/tmp/ftlist.txt` on the NUC before I switched to piping the
list over stdin. It is about 3 KB, in `/tmp`, and I left it rather than run a delete there.
