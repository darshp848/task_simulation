# Harness v2: changes backed by the Hill Climb reading list

Everything in DESIGN.md still holds unless changed here. Backward compatible: v1 plans still load and run
the same way (all new fields optional with defaults that reproduce v1).

Sources: the event prep guide; Sentinel (runtime monitoring), FLARE
(retry vs reset recovery), Hi Robot (hierarchical high level), SARM (stage-aware progress), Robometer
(progress dynamics), RTC and DreamZero (latency / prompt-change delay), IWR and RECAP (data loop).

## 1. A monitor, not a yes/no checker (Sentinel, SARM, Robometer, Hi Robot)

New verifier `monitor` (`verifiers.MonitorVerifier`), Claude-backed. Per call it sends:
- a frame **history**, not one image: the stage's first overhead frame + the last `history_frames` (default 6)
  overhead frames kept from earlier checks (oldest first), + the newest wrist frames. Images labelled
  ("t=+12s overhead", ...). Single-image QA mostly cries failure (Sentinel: TNR <= 7%).
- text: the overall task, the full stage list with the current one marked, the stage's `success` question,
  `expect` (optional, what correct behaviour looks like), elapsed seconds in the stage and its `timeout_s`,
  the last prompt sent to the robot, the stage's recovery menu (name + `when`), and this instruction:
  "First write 2-4 short questions about the robot and objects that decide this, and answer them from the
  images. Do not call the stage failing unless it is unlikely to finish within the time left. Then output
  JSON only between <json> and </json>:"
  `{"stage_done": bool, "progress": 0..1, "status": "ok"|"stalled"|"failing", "holding": {"left": str, "right": str},
    "recovery": <index into the recovery menu or null>, "scene_ok": bool, "confidence": 0..1, "reason": str}`
- Parsing: take the text between the markers; on any error return `MonitorVerdict(done=None, ...)`.
  Validate: `recovery` must be null or a valid index; clamp numbers.
- Runs **asynchronously**: the runner keeps polling status while a call is in flight (a background thread,
  at most one call in flight; results are consumed by the next tick). Calls take seconds (Sentinel: 14 s mean
  on GPT-4o), and a blocking call would starve health checks.

`MonitorVerdict(Verdict)` adds `progress: float|None`, `status: str|None`, `recovery: int|None`,
`scene_ok: bool|None`, `holding: dict|None`. Other verifiers return plain `Verdict` (treated as status None).

## 2. Debounced decisions (Sentinel, prep guide "false success is expensive")

Plan-level `confirm` (default 1 = v1 behaviour; recommend 2 with monitor):
- a stage succeeds only after `confirm` consecutive `done True` verdicts (manual `y` counts as confirmed at once);
- a monitor `status: failing` ends the attempt only after `confirm` consecutive failing verdicts AND elapsed
  >= 50% of `timeout_s` (manual `n` stays immediate);
- verdicts with no frames (empty dict) are never counted as success (missing data can't pass).

## 3. Stall detection from progress dynamics (Robometer, SARM)

Stage field `stall_s` (default null = off). If the monitor reports progress, keep the max progress seen in
the attempt; if it has not risen by >= 0.05 for `stall_s` seconds, or the monitor says `stalled` `confirm`
times in a row, the attempt fails with reason "stalled". A drop of >= 0.3 below the attempt's max
(regression, e.g. object dropped) logs a `regression` event and counts as a failing verdict.

## 4. Recovery menu with applicability, chosen from a closed set (FLARE, Hi Robot)

Stage `recovery` items may be a string (v1) or a mapping:
`{name, prompt, when, kind: retry|reset, settle_s (default 8), max_uses (default 1)}`.
- `kind: retry` = environment still on the task path: just re-issue the stage prompt (FLARE ID error).
- `kind: reset` = scene broken (toppled/dropped/stuck object): send the recovery prompt, run it for
  `settle_s`..`min(timeout_s/2, 20)` s, then ask the verifier for `scene_ok` (monitor) before re-issuing the
  stage prompt; if the scene is still not ok, that counts as another failed attempt.
- Selection on a failed attempt: the monitor's `recovery` index if it gave one and that item still has uses
  left; else the next unused item in list order; never the same item twice in a row; when none is left, fall
  back to `on_fail`.

## 5. Prompt-change settle time (DreamZero, RTC)

Stage `settle_s` (default = `min_run_s`): after any prompt change no verifier is asked and no stall clock
runs. A VLA executes the chunk already in flight and DreamZero-style models keep ~seconds of visual context,
so the old prompt lingers. Log `prompt_change` events with the status `chunks` count, and log the first
check where the monitor's `progress` moved, so the delay can be measured on the station.

## 6. Entry check (prep guide "entry conditions")

Stage `skip_if_done` (default false): before sending a stage's prompt, ask the verifier once; if it says done
with confidence >= 0.8 (monitor) or the operator typed y, record the stage ok with 0 attempts and move on.

## 7. Data loop and evaluation (IWR, RECAP, SARM, prep guide)

- `runs/<ts>/segments.jsonl`: one line per stage attempt and per recovery: `{stage, attempt, kind:
  stage|recovery, prompt, t_start, t_end (epoch s), outcome, reason, max_progress}`. These line up with
  Dream's own rollout recordings by wall clock, so successful stage segments can be cut out for fine-tuning
  and failed ones excluded (SARM drops negative-progress data; IWR keeps the policy's own data around
  corrections).
- `runs/<ts>/manifest.json`: plan file path + sha256, harness version, station status at start (checkpoint,
  prompt), CLI args, host, start/end time.
- `python -m harness.report runs/ [--last N]`: a table over runs: run ok, stages ok/total, per-stage success
  rate, mean attempts, mean seconds, recoveries used and their success rate, most common failure reasons.
  Also counts "false success" when a stage after a successful one fails immediately with
  `scene_ok false` (a lead, not proof).

## 8. Small things

- `MonitorVerifier` model default `claude-haiku-5-5`; plan `claude.model` overrides; `max_tokens` default 4000 (was 600; it cut off every Haiku 5.5 reply)
  (the questions need room).
- CLI `--verifier` accepts `monitor` and `monitor+manual` (operator key wins, like claude+manual).
- `ClaudeVerifier` (v1) unchanged.
- All new behaviour has tests against fake_station with a scripted fake monitor (no network).
