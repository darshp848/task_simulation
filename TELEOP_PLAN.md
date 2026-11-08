# Teleop recording plan: Sat 7 Nov, until the fine-tune is queued

## Decision (17:50): one fine-tune on UNPLUG and PUT-DOWN only

- **Unplug is everything.** It is Stage 1's pass bar: without it the run scores 0, and base Dream-1 unplugs about
  1 try in 6. Every demo minute goes here.
- **No plug-in.** Base Dream-1 never seated a plug, and no teleop take completed one (17:09-17:41: 50-140 s of
  fumbling, never released). The +20 per cable is not reachable this weekend. It is also out of the match plan now.
- **Stacking and balls stay on base Dream-1's own skill** (it stacks 2-4 cubes zero-shot). The fine-tune mixes in
  Dream's data by default (`base_share` 0.5), which keeps general skill. Sunday morning, test the fine-tune on a
  stack; if it got worse than base, run those stages on base Dream-1 (the harness can switch checkpoint per run).
- **One job, queued when recording stops** (not at 20:00: takes are still coming in). 1000 steps (the default,
  ~$200), constant learning rate. Training takes 7-8 h, plus any queue, plus up to 1 h of warm-up on the first call:
  queue no later than about 10 h before the first match.

## The two task texts (set one in the dashboard before EVERY take)

| # | Task text | Take | Have | Want by 19:45 |
| --- | --- | --- | --- | --- |
| T1 | `unplug the USB cable at the end of the hub nearest the right arm` | Enter with the gripper open ~5 cm from the end plug. Squeeze fully, pause, pull straight up until clear. Save while holding it up. 5-10 s | 56 (19:00) | done; more is better |
| T2 | `put the held cable down on the table away from the hub` | Starts holding the cable. Carry it to the same spot each time, open, lift. Save. 3-6 s | 45 (19:00) | done |

Chain them: T1, save, switch the text to T2, Enter, put down, save. Then plug the cable back in by hand
(arms still, between takes) and repeat. About 1 in 3 T1 takes with the end port already empty (the harness's extra
unplug), and about 1 in 5 starting from home instead of 5 cm away.

Other texts the harness sends (`put a block on top of the stack`, `put a golf ball in the bin`,
`open both grippers and move the arms up`) are left to base skill: `harness.tasks` will list them as MISSING, which
is expected.

## Rules for every take

- **Press Enter before touching anything,** gripper open, so the take starts before the grasp.
- **Under 30 s, one sub-task, exact text.** Discard (`d` / B) any take that stalls, fails or fumbles.
- **Full squeeze, straight pull.** The good takes pull the trigger fully for most of the grasp.
- **No hands in the overhead camera,** nobody else at the table during a take, left arm (leader) held low.
- **Leader/follower recording:** record into its own folder, then freeze the left wrist camera (below).

## Splitting a long take into task episodes

A take that does several sub-tasks in a row can be cut into one episode per sub-task, each under its exact text,
into a NEW dataset (the source is never touched). Find the cut times from the video, write a spec like
`tools/splits/plug-unplug-v1_ep3.json`, then run it on the laptop (it needs pyarrow, numpy and ffmpeg; the station
has no ffmpeg):

```bash
rsync -a pah@10.103.0.143:hackathon_data/<source> ~/hackathon_data/          # the spec's paths start at ~/hackathon_data
python tools/split_episodes.py tools/splits/<spec>.json
rsync -a ~/hackathon_data/<out> pah@10.103.0.143:hackathon_data/
# on the station: ~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/<out> --arm metal   (runs Dream's checks)
```

**Automatic cut times:** `tools/auto_split.py` proposes the spec from the right gripper's grasps (a grasp at the
hub that lifts and releases away from it = unplug + put down; a grasp away from it that releases at the hub =
plug-in), and writes one review image per episode (right wrist at each segment's start and end, overhead at its
end). Look at every row, delete bad segments from the spec, then run `split_episodes.py`:

```bash
python tools/auto_split.py ~/hackathon_data/<dataset> --out ~/hackathon_data/<dataset>-split --spec spec.json --sheets sheets/
```

Done so far (18:30), each with `-eef` built and Dream's checks passing; specs in `tools/splits/`:

| Folder | From | T1 | T2 |
| --- | --- | --- | --- |
| `unplug-split-v1` | the 60 s Quest take (16:44) | 3 | 3 |
| `leader-split-v1-frozen` | leader takes 17:09-17:16, left wrist frozen | 8 | 6 |
| `unplug-v1-split` | the reviewed good parts of `unplug-v1` (13:20-13:47) | 2 | 1 |
| `leader-split-v2-frozen` | leader takes 6 and 9 (17:30-17:53), left wrist frozen | 3 | 0 |
| `quest-split-v3` | Quest takes 12-22 (18:12-18:31); every unplug ends with the plug in the jaws | 40 | 35 |
| **total** | | **56** | **45** |

Optional, stacking (`stack-split-v1-frozen`, from leader-right takes 10-11, right wrist frozen, the operator's
arm is in the overhead view): 2 × `put a block on top of the stack` and 2 × T7
`stand the cylinder upright in the hole of the top block`. Zero-shot stacks cubes but not cylinders, so T7 is the
stacking skill worth demos: 20+ short takes under that exact text would make it count. The match plan does not send
T7 yet; add a cylinder stage once there are demos.

Not used: raw `unplug-v1` (its other cuts end with an empty gripper); `plug-unplug-v1` 3-5, 7, 8 (start already
holding, plug-in fumbling), 23 (59 s, nothing moves), and the unlabelled parts of 10-11. The raw `plug-unplug-v1`
(placeholder text) never goes up as it is. `~/hackathon_backup/plug-unplug-v1-1714` is a 17:14 copy of its first
5 takes, already covered.
Better still, record each sub-task as its own episode from the start.

## Leader/follower episodes: freeze the left wrist camera

With `follower_code/metal/leader_collect.py` the left arm is the leader, moved by hand, so its wrist camera swings
around and films the operator (measured on the 17:09 takes: the view wanders 59-63 on a 0-255 scale, against 6 for
a parked camera). In a match the left arm is parked and that camera is still. Before `add_eef`, write a copy where
that camera shows the parked view (laptop; the station has no ffmpeg):

```bash
rsync -a pah@10.103.0.143:hackathon_data/<leader-dataset> ~/hackathon_data/
python tools/freeze_camera.py ~/hackathon_data/<leader-dataset> --ref ~/hackathon_data/<quest-dataset>:EP:0 --list
python tools/freeze_camera.py ~/hackathon_data/<leader-dataset> ~/hackathon_data/<leader-dataset>-frozen \
    --episodes all --ref ~/hackathon_data/<quest-dataset>:EP:0
rsync -a ~/hackathon_data/<leader-dataset>-frozen pah@10.103.0.143:hackathon_data/
# station: ~/dream-env/bin/python tools/add_eef.py ~/hackathon_data/<leader-dataset>-frozen --arm metal
```

The reference must be a parked-left-arm frame from the CURRENT table layout. Keep leader and Quest episodes in
separate folders, so `--episodes all` is right. Dream's checks pass on a frozen copy (tested 17:15).

## Clean-up done

17:00: 8 empty or no-grasp episodes dropped (unplug-v1 1, 2, 5, 7, 8, 9; plug-unplug-v1 1, 2). `unplug-v1` keeps 4
(30, 15, 46 and 96 s). Long multi-task takes stay in `plug-unplug-v1` and never go up as they are: only their
reviewed splits do.

## Queue the fine-tune (when recording stops, arms not needed)

What goes in: the four reviewed folders above, and whatever is recorded from now on (leader takes: split or
recorded short with the exact text, then left wrist frozen). Each as its `-eef` copy:

```bash
cd ~/physical-ai-hacks
# add_eef on each new folder (the frozen copy for leader takes); the four reviewed ones already have -eef
mkdir -p ~/hackathon_data/ft-v1 && cp -r ~/hackathon_data/{unplug-split-v1,leader-split-v1-frozen,unplug-v1-split,leader-split-v2-frozen,quest-split-v3}-eef ~/hackathon_data/ft-v1/
# optional: cp -r ~/hackathon_data/stack-split-v1-frozen-eef ~/hackathon_data/ft-v1/
(cd ~/hill-climb-harness && python3 -m harness.tasks plans/hill_climb_v4.yaml ~/hackathon_data/ft-v1)   # T1, T2 counts
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/ft-v1 --arm metal --name unplug-v1 --suffix unplug-v1 --check
# PAID, about $200 (1000 steps): only with the team's yes
tmux new -A -s finetune
~/dream-env/bin/python tools/finetune.py ~/hackathon_data/ft-v1 --arm metal --name unplug-v1 --suffix unplug-v1
```

The platform reads each subfolder of `ft-v1` as one session. Keep the default constant learning rate (not cosine),
which keeps `resume_from` possible. Training takes 7-8 h, so the model is ready around 04:00, and its first call can
take up to 1 h to warm up.

## Sunday morning

1. Warm the model up early: `dreamcli robot run --checkpoint dream-1-ft:<account>-unplug-v1 --task "unplug the USB
   cable at the end of the hub nearest the right arm" --dry-run` (the first call may wait up to an hour).
2. Start the agent with `PYTHONPATH=~/hill-climb-harness/station_addons dreamcli robot up --view --view-port 8093`:
   plain `up`, not `up joint` (a Metal fine-tune answers tool poses, which dreamcli solves to joints).
3. Compare fine-tune and base on the same plan, `--verifier monitor+manual`, a few runs each: unplug success first,
   then stacking. Put the winner's id in `plans/hill_climb_v4.yaml` (`checkpoint:`). If the fine-tune unplugs better
   but stacks worse, splitting the checkpoint by stage is a harness change to make then.
4. Match: `--live` with `monitor` only.
