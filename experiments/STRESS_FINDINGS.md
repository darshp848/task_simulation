# Harness stress test, 8 Nov 2026

`python experiments/stress_match.py --seeds 4` runs whole `plans/hill_climb_v4.yaml` matches on a virtual clock
(about 5 s each). The real Runner and MonitorVerifier run unchanged; the station, table, policy and the monitor's
Claude call are scripted. Monitor latency 4 s unless noted. Points come from the rubric v4 referee in the script.
Faults are modelled on the station NUC's `robots-inference.jsonl` from 7 Nov: 42 runs, 10 errors (DA-4000
"unsafe instruction" refusal x4, LinAlgError "SVD did not converge" x3, ControlLoopError "action buffer empty
12 s" x2, session expired x1), plus 68 stalls.

| scenario | main (3055bf3) points, 4 seeds | this branch | what broke on main |
|---|---|---|---|
| happy | 87 74 90 116 | 84 89 90 86 | |
| refusal_at_start (DA-4000 on first start) | 0 0 0 0, match ended | 79 84 90 86 | `_start_run` treated any error on the way up as fatal |
| refusal_twice | 0 0 0 0, ended | 79 74 85 81 | same |
| starve_mid (ControlLoopError) | 10 10 15 15, ended | 56 84 90 96 | not classed as an inference fault |
| linalg_mid (SVD did not converge) | 10 0 10 10, ended | 84 79 90 86 | same |
| no_release (policy keeps the unplugged plug) | replugged it 4/4 | 0/4 | plug_1 pushed the held, unplugged cable back in (forbidden) |
| stuck_cable (plug won't leave the gripper) | replugged 4/4 | 10 each, no replug | same; now plug stages are skipped |
| corrupt_first_frame (12 s of bad JPEGs) | 20 15 20 20 | 84 89 90 86 | stage-start frame rides in every request: the API 400s for the whole unplug stage |
| arm_loss | ended, then sent /start to a dead arm | ended, no /start | `_issue` restarted a station in error without looking at the error |
| api_outage_balls (monitor down, black balls) | robot bins balls unwatched | stops after 20 s blind | guard silent when unanswered |
| black_ball (black_p 0.3) | lost 1/4 | lost 3/4 (noise; see below) | guard latency |
| reach_across (unplug brushes a block) | DQ 4/4 | DQ 4/4 | not fixable by the harness |

## Fixes on this branch
1. Station faults: refusals (DA-4000) are told apart from inference faults. A refused prompt is restarted once,
   then that attempt fails and the match goes on (skip stages are skipped, pass-bar stages retry). ControlLoopError,
   LinAlgError and DreamSessionExpired restart like gateway timeouts. A fault while starting is handled the same way.
2. Before a fresh /start, an arm/motor error ends the match instead of being cleared by the start.
3. Stage `skip_unless:` (a question asked before the prompt). The v4 plug stages run only if both grippers are empty;
   a new `release_cable` stage (T6 prompt, skipped if already empty) follows `put_cable_down`.
4. Stage `confirm:` override. Short stages that share a prompt with the next one (balls, higher_*, put-downs,
   arms_clear) use 1: with confirm 2 and a 4 s monitor they could not pass inside 10-12 s.
5. At timeout, one 10 s grace when a "done" is waiting for its confirming look.
6. Monitor drops frames that are not JPEGs, so one bad frame can't blind a stage.
7. Two "done" verdicts on byte-identical frames count once (frozen debug view must not confirm a tower).
8. Guard: a "yes" from a call that was in flight across a stage change is no longer dropped; `max_blind_s` (20 in v4)
   stops the run when a ball stage has had no guard answer that long.

Main dropped the plug-in stages on 7 Nov (merged here), so the re-plug fault can no longer happen with the v4 plan;
`skip_unless` stays for any plan that plugs in, and `release_cable` still frees a held plug before stacking.

## Local detector (`--detect plans/table_colors.yaml`)
`harness/detect.py` finds coloured blobs and calls each a ball, cube or lying stick from its shape. Tuned on the
top-down photo of station C (Drive IMG_2698; crop in `tests/data/`): it finds the white ball (illegal), the green,
yellow and orange balls, blue and pink cubes, and the yellow and pink sticks; the black gripper fingers are not
reported. The runner asks it about the guard on every poll (wrist views: an "avoid" ball between the fingers), and
the monitor gets the frames with labelled boxes. Policy inputs and prompts are untouched.
Sim, black_p 0.3, frames 2 s old, 6 seeds: matches lost to a binned illegal ball 4/6 with the monitor guard alone,
2/6 with the local guard. Colours must be re-checked on the station's own overhead and wrist cameras.

## Still open
- The illegal-ball guard still loses some races: pick to bin takes a few seconds, the debug view's frames are a trip
  old (~3.4 s cadence on the station). The local detector helps (above) but can't see faster than the frames.
- higher_* stages risk the tower: a knock-down after the pass bar keeps the +20 but loses every height point.
- Touching a block during unplug is instant; nothing that reads camera frames afterwards can prevent that DQ.
- Watch blocks: fixes eat the stage timeout, and after `max_fixes` the stage runs unwatched without a log line.
- Time-budget end is reported as `ok: false, "time budget exceeded"`, which is the normal end of a v4 match.
