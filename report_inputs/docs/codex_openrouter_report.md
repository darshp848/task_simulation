Committed as `700b301` on `hedged-calls`. No push, paid API calls, account creation, or key entry.

Files added:

- [Setup checklist](C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/hedged-calls/docs/OPENROUTER_SETUP.md), with exact PowerShell and `gcloud` commands.
- [Benchmark tool](C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/hedged-calls/tools/bench_judges.py), supporting singles, races, hedges, dry-run, CSV/JSON reports, and spend stops.
- [Offline tests](C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/hedged-calls/tests/test_bench_judges.py) and [research/results](C:/Users/darsh/TAMU/EMPE_Lab/Actuator-controls/hill_climb_harness_test/hedged-calls/docs/OPENROUTER_RESEARCH.md).

Updated monitor accounting to include losing calls and corrected a pre-existing race in the async test.

Verification: **354 passed, 1 skipped**, with only the two documented Windows CLI failures. All 32 new tests passed. The real CLI dry-run validated **108 trials across 18 configurations**, including MuJoCo renders.

Candidates were checked against the [live catalog](https://openrouter.ai/api/v1/models):

| Model | Reason |
|---|---|
| Haiku 5.5 | Existing baseline |
| Gemini 3.1 Flash-Lite | Flash-Lite alternative and race partner |
| Qwen 3.8 Flash | Inexpensive vision output |
| GLM 5.3 Flash | Lowest advertised token prices among these candidates |
| Llama 4 Scout | Independent vision family |
| GPT-5 nano | Small, inexpensive vision model |
| Mistral Small 2603 | Mistral vision alternative |
| Gemma 4 26B | Low-cost sparse vision alternative |

Speed and accuracy remain unmeasured because live calls were prohibited. The spend stop can overshoot by the concurrent trial already running.

Follow the linked setup checklist in this order:

1. USER ACTION NEEDED: Sign in at [OpenRouter](https://openrouter.ai).
2. USER ACTION NEEDED: Add a small balance on [Credits](https://openrouter.ai/settings/credits).
3. USER ACTION NEEDED: Create a key on [API keys](https://openrouter.ai/settings/keys) and set its credit limit.
4. USER ACTION NEEDED: Store it in the Windows user variable `OPENROUTER_API_KEY` using the masked PowerShell command.
5. USER ACTION NEEDED: Create `openrouter-api-key` in Secret Manager project `hill-climb-sims` from stdin.
6. USER ACTION NEEDED: Grant the cloud runner access and inject the secret into its environment.
7. USER ACTION NEEDED: Optionally review privacy, logging, and provider policies.
8. USER ACTION NEEDED: Run the documented dry-run.
9. USER ACTION NEEDED: When ready to spend credits, run the single-model check.
10. USER ACTION NEEDED: Compare candidates, races, and hedges using the documented commands.