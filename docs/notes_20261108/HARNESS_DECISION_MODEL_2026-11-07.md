# Harness decision model: runtime, arms, balls, prompts, and the decision tree as tunable equations

7 Nov 2026 (late). This builds on `HARNESS_SPEED_STATE_SPACE_2026-11-07.md`. It uses:
- today's inference logs from the NUC (1,650 chunks over 52 runs);
- the runner on `main` / `start-pose-live-guard`;
- kaupau's `hybrid-harness` branch, which does prompt generation better (section 4);
- the sequential vs combined two-arm benchmark (`codex/dual-arm-experiment`, `runs_dual_arm/`).

## 0. Findings that drive the design

| # | Finding | Evidence |
|---|---|---|
| F1 | **The arms sit still about 30% of the time.** Dream's pacer runs `sync` whenever the median round trip is over 900 ms: it waits for each 2.4 s chunk, and chunks arrive every 3.54 s (p50). In `async` mode chunks arrive every 1.55 s with no gap. 65% of today's chunks were sync. | `robots-inference.jsonl`; station.log `dream.pacing: pacing sync ... over the 900 ms threshold` |
| F2 | The round trip is about 950 ms even with a fast upload (≈650 ms of gateway overhead), and upload size makes it worse. Each request is ~1.6 MB. With a 5 Mbps uplink (p10) the round trip is ~1.9 s, and only 4% of those chunks could run async. | same log, `request_bytes`, `up_mbps` |
| F3 | A fresh `/start` costs 2.9 s (p50) before motion begins; `/prompt` on a running session costs nothing extra. | run_start → chunk 1 |
| F4 | The harness costs about 5-8 s per stage boundary: 3 s poll, ~4 s monitor call, then confirm 2 means one more look. | runner constants, stress sim |
| F5 | **Dream-1 does not reliably obey arm naming.** "with the left arm, …" succeeded 0/4 and the right arm still moved 10-19 s. "Left does X while right does Y" succeeded 0/4. Two goals without naming arms finished one goal in 2/4 trials. (YAM sim, n = 12.) | `runs_dual_arm/20261107T230412Z/results.jsonl` |
| F6 | Base Dream-1 follows short, specific prompts that name objects by colour and position much better than the generic task texts. It refuses cable work without the "no power" sentence (4 × DA-4000 today). | `hybrid-harness` plan notes, inference log errors |
| F7 | The harness's own 2 MB debug-view frame fetches slowed Dream to 2-6 s per chunk. The add-on's `/frame/<cam>.jpg?w=640` frames are ~35 KB. | `hybrid-harness` HARNESS_GUIDE 6g |

## 1. Runtime: where the seconds go, ranked by expected gain

Model of one stage: `T_stage = T_start + T_robot / u + T_detect + T_decide`, where `u` is the share of time the arms actually move (F1).

| Lever | Saves | Cost / risk | How to measure |
|---|---|---|---|
| **L1. Get Dream into async.** Wired ethernet on the NUC. Stop the harness pulling debug-view trips (use hybrid's light frames, F7). Ask Dream whether the 900 ms pacing threshold or the image size per request can be configured. | u ≈ 0.70 → ~0.95, about 25% less robot time on every stage | network work only; any station config change is the station owner's decision | share of `async` chunks and chunk-gap p50 in `robots-inference.jsonl`, before vs after |
| **L2. Event-driven stage done.** A local estimator (gripper sensors from `/arms`, detector boxes, cable-end boxes) advances when its posterior is high. The monitor confirms in parallel and can veto. | 4-7 s per boundary, 20-35 s per match | a false done on a pass bar is a DQ, so it is gated by the bound in section 5.3 | `stage_start` − true-done time in replayed logs |
| **L3. Never `/start` again mid-match.** Recoveries and retries use `/prompt`, not stop + start. | 2.9 s each time | none; it's already mostly the case | count `start` calls per match |
| **L4. Skip checks the sensors can answer.** "Are both grippers empty?" (release_cable, arms_clear, put-downs) comes from `/arms` gripper readings in milliseconds, not a 4 s synchronous VLM look. Hybrid already lets the sensors overrule the monitor's `holding`. | ~4 s per skippable stage, 3-4 per match | sensor calibration (hybrid has this station's readings) | `entry_check` latency |
| **L5. Faster judge.** Use hybrid's stitched grid, two staggered calls in flight, and 4 history frames. | a verdict every ~5 s instead of ~10 s | merge from hybrid | `latency_s` in verdict events |

L1 and L2 dominate. L1 is a network and ops change, not code.

## 2. Both arms at once

The station API has **one prompt for one bimanual policy**. There is no per-arm command and no way to run two sessions. So "both arms at the same time" has to come from one of these:

| Mode | How | Works today? | Use for |
|---|---|---|---|
| **A. One worker, the other parked (current).** | The right arm does everything; the left is parked by the reset prompt. | yes | default |
| **B. Watcher.** | The stage's `watch:` puts one arm's wrist camera on the work, and the other arm works. The harness re-sends the park prompt if the watcher drifts (`watch_lost`). | partly: depends on Dream holding the watcher still (F5 says naming the arm is unreliable) | close views of the port, the ball in the gripper, the tower top |
| **C. Combined prompt, two goals.** | "put the X … and the Y …" without naming arms; Dream assigns them. | 2/4 single-goal successes in the sim, both arms moving at once | parallel independent goals, e.g. ball + ball, once tested on Metal |
| **D. Trained bimanual skill.** | Record teleop demos where the left arm watches or helps while the right works, under one fixed task text. | needs data | the only route to dependable two-arm behaviour |

The harness can't make the two arms independent tasks. It **can** decide which mode each stage uses, which arm is the worker (section 3), and when a combined prompt is worth the risk. That decision gets a weight like the others (section 5).

Experiment E1 (sim first, then station): 3 modes × 20 trials each on the same seeded scenes. Measure success, time to done, both-arms-moving seconds, and collisions or knocks. 20 per mode separates 30% from 70% success. A 0.5 vs 0.8 gap needs about 40 per mode for 80% power.

## 3. Picking the arm from the layout

What the harness has: the overhead camera, colour detector boxes (balls, cubes, sticks), cable-end boxes (new), gripper state, and joints (`/arms`, after a station restart).

1. **Table frame.** One homography H from overhead pixels to table metres. Click the four table corners once per station (or mark them with tags) and store it in the plan. The arm bases `b_L` and `b_R` and the bin, hub and tower positions all go in that frame.
2. **Cost of arm a for object o:**

       c_a(o) = w_reach · ‖p_o − b_a‖ / R_a             reach, normalised by the arm's reach
              + w_cross · 1[o is past the midline for a]   crossing the body
              + w_clear · Σ_k exp(−d(path_a,o, k)² / σ²)   obstacles near the straight path: tower, hub, illegal balls
              + w_busy  · 1[a holds something or is the watcher]
              + w_skill · (1 − s_a(task))                  arm-specific skill prior: demos exist only for the right arm

   Choose `a* = argmin_a c_a(o)`. If `c_a*(o) > c_max` for every arm, the object is unreachable: skip it or pick another object.
3. **Turning the choice into an action.** F5 means naming the arm in the prompt is not enough on its own. In order of dependability:
   - (i) choose the **object** the chosen arm is best placed for, and name it by colour and position ("the orange ball nearest the right arm"). Dream tends to use the nearer arm;
   - (ii) for a fine-tune, record per-arm task texts ("… nearest the left arm" as well as "… nearest the right arm");
   - (iii) the arm phrase as a hint only, and the monitor checks which arm moved. Count a wrong-arm move as a failed attempt.
4. Experiment E2: does base Dream-1 on Metal pick the nearer arm? Place objects at 5 positions from left to right, 4 trials each, with and without an arm phrase. Log which arm moved first. The result sets `w_skill` and tells us whether (iii) is worth anything.

## 4. Prompting Dream

**Build on kaupau's `hybrid-harness` branch.** What it does that main doesn't:
- **Templates with slots:** `tower_prompts` with `<piece>` / `<top>`, cable prompts as plan variables, and a `prompt_prefix` that is put back on every rewrite.
- **Closed-loop rewriting:** the monitor returns `next_prompt` when it sees a wrong object or step, a plain stall, or a failure about to happen. The runner sends it, with a cooldown of 15 s and at most 3 per attempt. Every rewrite is logged as `prompt_rewrite`.
- **Station lessons in the monitor's system prompt:** plugs are pulled straight up; carry a block high and lower it straight down; Dream ignores fine adjustments like "1 cm higher". There is also a running operator-lessons file, for testing only.
- **`tools/label_rollout.py`:** labels Dream's own run records with stage, attempt, outcome and the clean task text. This is the bridge to fine-tune data.

Formalised: the prompt is a policy `π_P(x) = template_s(slots(x))`, with `rewrite(x)` as a gated fallback. Slots come from the state estimate: object colour and position, arm (section 3), next tower piece, and which plug is next. A rewrite fires only when

    g_rw = w_wrong·1[wrong object/step] + w_stall·(t − t_rise)/stall_s + w_risk·P(knock) − w_churn·n_rewrites  >  θ_rw
    and  t − t_last_prompt ≥ τ_cool

Hybrid's fixed rules are one point in this space: `θ_rw` = "the monitor chose to write one", τ_cool = 15, and a hard cap of 3.

**Prompt experiments (E3).** The unit is one stage from a seeded scene, scored on success within T, time to first contact, and refusals.
- Factors: specificity (generic task text vs colour + position), motion adverbs ("lift high, lower straight down"), arm phrase (none / hint), preamble (on / off, cables only), and length.
- Plan: a fractional factorial screen in the YAM sim with real Dream-1, which is cheap. Then confirm the best 2-3 on Metal at 10-20 trials each.
- Prompts already sent today, from the inference log, cover about 40 variants across 52 runs, but the log has no outcomes. Join them to harness `events.jsonl` with `label_rollout.py` to get a free first ranking.

The fine-tune rule still holds. A fine-tune follows only its recorded texts, so whatever the base-model experiments settle on is what the teleop labels must say. Hybrid runs a fine-tune with `--no-rewrite` plus the recorded texts.

## 5. The decision tree as weighted equations

### 5.1 What the runner does today, as rules

At each check (every `check_every_s` = 3 s), per stage attempt:
- **done** if the verifier says done `confirm` (2) times in a row, or twice within `DONE_WINDOW_S` (15 s), or with `immediate` set;
- **fail** if `failing` comes `confirm` times in a row and elapsed ≥ `FAIL_MIN_FRACTION` (0.5) × `timeout_s`; or progress drops by `REGRESSION_DROP` (0.3) below the attempt's max; or progress hasn't risen by `STALL_RISE` (0.05) for `stall_s`; or `timeout_s` passes, with one extension of `EXTEND_S` (20 s) while the gripper is on the target and `CONFIRM_GRACE_S` (10 s) while a done is pending;
- **on fail:** abort, skip, or the recovery menu in order (`max_uses` each), then `home_and_retry` if allowed, up to `max_attempts`, never with less than `HOME_RESERVE_S` (30 s) left;
- **guard trips** after `guard.confirm` illegal-ball answers (live: 3 distinct frames at 10 Hz), or when blind for `max_blind_s`.

### 5.2 The same tree as equations

State `x_t`: stage s, elapsed τ, time left `L = T − t`, the evidence features `φ(t)`, attempt count n, recovery uses, and the arm and object state.

**Done evidence (log-odds, recursive):**

    ℓ_t = γ · ℓ_{t−1} + Σ_i w_i · φ_i(t)

| φ_i | Source | Today's implicit weight |
|---|---|---|
| monitor done (+1) / not (−1), × confidence | VLM | +1 / reset to 0 |
| progress p_t (0..1) | VLM | used only for stall and regression |
| gripper went from closed on an object to open, near the target | `/arms` | none (new) |
| object at goal (detector box inside the bin or on the tower) | detector | none (new) |
| plug count in the hub fell (cable-end boxes) | classifier | none (new) |
| same frames as last time | sha1 | φ = 0 (not counted) |

Done when `ℓ_t ≥ θ_done`. Today's rule is the special case γ = 1 on agreement and 0 on disagreement, `w_monitor = 1`, `θ_done = confirm = 2`.

**Fail hazard:**

    h_t = σ( b + w_f·n_fail + w_r·max(0, p_max − p_t − δ) + w_s·(τ − τ_rise)/stall_s + w_T·τ/timeout_s − w_on·1[on target] )

The attempt ends when `h_t > θ_fail`. The hard rules above are the corners of this function: they fire when any one term is large.

**Action choice when an attempt ends (or at each tick, for `continue`):** a ∈ {continue, re-prompt / rewrite, reset prompt, home + retry, skip, abort}.

    Q(a | x) = P_a(succ within c_a) · V_s  +  W(L − c_a)  −  λ_risk · R_a

- `V_s` = the stage's points (rubric v4: first cable +10, extra cable +5, tower pass +20, +3 per inch, ball +5), plus the value it unlocks (pass bars gate later stages).
- `W(L)` = expected points from the stages after s with L seconds left. Compute it backwards over the stage list once per plan, using each stage's success-vs-time curve `F_s(τ)`.
- `c_a` = time the action costs (re-prompt ~1 s, reset 5-20 s, home ~10 s plus `/start` 2.9 s).
- `R_a` = risk: a DQ from a false done or a touch out of order, a knocked tower, an illegal ball.

Pick `a* = argmax Q`. `on_fail: skip` and `max_attempts` become consequences of Q, not hand-set values.

**Optimal timeout (replaces a fixed `timeout_s`):** keep going on stage s while the success rate per second, times its value, beats what the remaining time is worth elsewhere:

    f_s(τ) / (1 − F_s(τ)) · V_s   ≥   dW/dL

That gives each stage a timeout that depends on its own success curve and on how much match is left. Optional stages get short timeouts late in the match and longer ones early.

### 5.3 The pass-bar constraint

A false done on a pass bar (unplug, stack_2, stack_3, tower_pass) is a DQ. With per-look false-positive rate α for the evidence and k independent looks required, `P(false done) ≈ α^k`. Choose `θ_done` per stage so that `P(false done) ≤ ε` (e.g. 1%), then minimise the expected detection delay. This is the one hard constraint in the tuning. Everything else trades off in `J` below.

### 5.4 Parameters and tuning

θ = {w_i, γ, θ_done(s), w_f, w_r, w_s, w_T, w_on, θ_fail, λ_risk, θ_rw, w_wrong, w_churn, τ_cool, c_a weights (section 3), check period, live-guard Hz and frames, start-pose tolerance}. Today's constants (`confirm`, `STALL_RISE`, `REGRESSION_DROP`, `FAIL_MIN_FRACTION`, `DONE_WINDOW_S`, `EXTEND_S`, `CONFIRM_GRACE_S`, `HOME_RESERVE_S`, `SKIP_MIN_CONFIDENCE`, `REWRITE_*`) are its starting point.

Objective, estimated in the stress sim and the rubric-v4 referee:

    J(θ) = E[score] − μ · P(DQ) − ν · P(illegal ball)

1. **Calibrate the sim's evidence model from real logs.** Per stage, how often the monitor says done when it's not, and the reverse (label saved frames by hand). Also the detector's miss and false-alarm rates, `F_s(τ)` from the runs, and refusal and IK-error rates. This is the STEP / SIMPLER warning: tuning against a sim with the wrong noise picks the wrong θ.
2. **Search:** CMA-ES or coordinate hill climbing over θ. Use common random numbers (the same seeded scenes for every candidate), about 200 sim matches per candidate, run on the cluster.
3. **Confirm on Metal:** the best 2 θ vs today's, 5 supervised matches each. Promote only if J is better and P(DQ) didn't rise.

## 6. Ball handling

Golf ball: 42.7 mm, rolls when nudged. The bin is white, the same colour as the illegal white ball. Dream does the grasp. The harness controls **which ball, from where, with what prompt, and whether the grasp worked**:

1. **Choose the ball** (overhead detector + homography): legal colour; reachable by the chosen arm (section 3); clearance from the tower `d_tower ≥ 2·D_ball` and from any illegal ball `d_illegal ≥ 1.5·D_ball` so the gripper can't take both; not against an edge it can roll off. Score `s(b) = −c_a*(b) − w_tw·exp(−d_tower²/σ²) − w_il·exp(−d_illegal²/σ²)` and send the best: "pick up the orange ball nearest the right arm, straight down from above, and drop it into the white bin".
2. **Grasp check, local and fast.** After the gripper closes, its opening g from `/arms` should sit in a band around the ball's width, `g ∈ [g_ball − δ, g_ball + δ]`, for N ticks. Fully closed means a miss; wider means it's on something else. This is the same idea as hybrid's "closed fully on nothing" sensor rule. Calibrate `g_ball` once by closing on a ball.
3. **Rolled away:** if the target's overhead position jumps by more than D_ball between frames without being held, re-target with a new prompt naming the ball's new position. Don't retry blind.
4. **The illegal ball:** the live guard (10 Hz, `/stop` in under 0.5 s) stays as the hard stop. The ball choice in step 1 lowers the chance it's ever needed.
5. **Dream's own ball skill:** there are no ball demos yet; balls rely on base Dream-1. To make it dependable, record ball teleop (top-down grasp, carry high, release over the bin) under one fixed text, and add one demo set starting next to the white ball so the policy learns to go around it.

## 7. What to build, in order

1. Merge `hybrid-harness` (prompting, light frames, faster judge, sensor overrides) with `start-pose-live-guard` (start pose, live guard, cable-end classifier).
2. **L1 network check:** ethernet, and light frames only. Re-measure the async share. Ops only; the biggest single gain.
3. Overhead homography + arm/ball choice (sections 3 and 6) as a `layout.py` that feeds prompt slots.
4. A `decide.py` with the equations in 5.2. Default weights reproduce today's rules exactly, with a test proving it. Then calibrate and tune in the sim (5.4).
5. Experiments E1 (dual-arm modes), E2 (nearer-arm choice), E3 (prompt factors), sim first, then Metal.
