# How fast the harness sees and reacts, and a state-space version

7 Nov 2026. These are the numbers measured today, followed by a proposal for a state-space harness.

## What runs, and how fast

| Piece | Rate / latency | Source |
|---|---|---|
| Dream-1 low-level control | 20 Hz, 48-step chunks (2.4 s) | station config |
| Dream inference round trip | p10 0.80 s, p50 1.06 s, p90 2.52 s (gateway 0.95 s, model serve 0.29 s) | robots-inference.jsonl |
| Debug-view trips (frames + state the harness polls) | ~3.4 s median between trips | `/trips` |
| Harness poll loop (`check_every_s`) | 3 s | plan |
| Haiku monitor (stage done? illegal ball?) | ~4 s per call (assumed in the stress sim; not measured live) | stress sim |
| Local colour detector, guard on both wrist views | 76 ms (236 ms before the speed-up) | bench |
| Live frames, `GET /frame/<cam>.jpg` (station add-on) | 30 fps at source, 4 ms per grab, ~200 grabs/s from a local server | bench |
| Live guard thread | 10 Hz, trips after 3 distinct frames in a row → ~0.3-0.5 s to `/stop` | `live_guard.py` |
| Cable-end classifier (new) | 12.4 ms per 640-wide frame (~4.6 orange blobs each), CPU, cv2.dnn | `cable_end.py`, 16 table photos |

The slow path is decision → evidence → decision. A stage finishes; the harness notices on the next 3 s poll; the monitor takes about 4 s to confirm; then the next prompt goes out and Dream takes about 1 s for its first chunk. So each stage boundary costs **about 5-8 s of idle robot time**. With 4 or 5 stages per task, that adds up to 25-40 s of a 3-minute match.

The safety path is already fast. The live guard reacts in under 0.5 s, and it runs separately from the monitor.

## Can the harness itself be a fast, real-time state-space system?

Yes, with one caveat about what the harness controls. It does not command the arms; Dream does that at 20 Hz. The harness has only discrete inputs: which prompt to send and when, plus stop, home and retry. So the "system" the harness controls is the task. Its state looks like this:

    x = [ stage progress (which sub-goal is done),
          object state: ball/block/cable-end positions (detector + cable-end classifier),
          arm state: joints, grippers, velocities (from /arms, 20-30 Hz),
          risk: illegal-ball proximity, distance to start pose, time left ]

    x_{k+1} = f(x_k, u_k) + w_k,   u_k ∈ {keep, next prompt, re-prompt, stop, home}
    y_k     = h(x_k) + v_k         (joint encoders, colour boxes, cable-end boxes, occasional VLM verdict)

How to build it:

1. **Estimator (10-30 Hz, all local).** Use a Bayes or HMM filter over discrete stage progress, plus a small Kalman filter per tracked object. It takes in:
   - joints and grippers from `/arms`;
   - colour boxes and cable-end boxes from live frames;
   - the monitor's verdict, which arrives every few seconds as a slower, high-trust measurement.

   Stage-done becomes a posterior crossing a threshold, not a 3 s poll followed by a 4 s VLM call. Example: "gripper closed on a USB-A box and the arm is lifting" means unplug is done with p > 0.95.
2. **Policy over u (event-driven).** Send the next prompt the moment the posterior crosses the threshold, so Dream starts its next chunk about 1 s later. Keep the VLM as a confirm-or-veto check that runs in parallel instead of blocking.
3. **Hill-climb the harness parameters, not the robot.** The tunables are the thresholds, the confirm frames, the re-prompt timing, and the home/retry rules. Each evaluation runs in the stress sim (about 5 s per simulated match), and the cluster can run thousands of them. The STEP and SIMPLER caveats from the papers doc apply: the sim's noise model has to match the real detector's miss and false-alarm rates. Calibrate it on logged runs from `nuc_data/`. Then confirm the winners on the real station with a small number of supervised matches.

Expected gain: if the estimator calls stage-done within about 0.5 s instead of 5-8 s, each match saves 20-35 s. On a timed task that is the difference between finishing and not finishing. Dream's own speed (about 1 s per round trip, 2.4 s chunks) is the floor, and we can't change it.

What's needed that we don't have yet:
- `/arms` `positions_rad` live: the code is deployed, but the station must be restarted for it to take effect.
- A stage-done observation model for each stage. Fit it from the teleop datasets: gripper and joint traces at segment boundaries are labelled by the task-segment prompts.
- Detector and classifier error rates on logged runs, to calibrate the sim.

## Suggested order

1. Fit the stage-done models from the LeRobot datasets, offline, a few hours.
2. Add an `Estimator` thread next to `LiveGuard`. Log its posterior alongside the monitor's verdicts for a few real matches in shadow mode, with no control authority.
3. Once it agrees with the monitor more than 95% of the time, let it advance stages, keeping the monitor as a veto.
4. Hill-climb thresholds and timings in the stress sim on the cluster.
