# Report inputs (local-only material, 7–9 Nov 2026)

This branch exists to feed the Hill Climb technical report (hill-climb-harness `technical-report`). It collects material that was only on Darsh's laptop and never pushed:

- `docs/HARNESS_DECISION_MODEL_2026-11-07.md`: the decision tree written as equations (evidence, hazard, Q-value, timeouts), arm choice, balls, dual-arm modes.
- `docs/HARNESS_SPEED_STATE_SPACE_2026-11-07.md`: the latency/pacing state space (sync above ~900 ms RTT, ~650 ms fixed gateway overhead).
- `docs/HARNESS_DIRECTIONS_2026-11-07.md`: harness directions from 7 Nov.
- `docs/HARNESS_SIM_INSTRUCTIONS.md`: how the early sims were run.
- `docs/FT_DIAGNOSIS_2026-11-08.md`: why the relay-final fine-tune failed (gravity sag in the Quest demos).
- `docs/codex_*_report.md`: Codex's end-of-task reports for the GCP cloud runner, the GCP console audit and the OpenRouter judge bench. The billing ID and the console link are redacted.
- `codex_branch_patches/`: the three hill-climb-harness commits that exist only locally, as `git format-patch` output, on top of `codex/watcher-dimos` (89c9621). In order:
  1. match supervisor (bound the match, abandon optional work);
  2. LiveKit camera observer;
  3. validation of the match decision tree on 300-second runs.

  They were never merged into main.

Apply the patches with `git am` on a branch cut from 89c9621 to read them in context.
