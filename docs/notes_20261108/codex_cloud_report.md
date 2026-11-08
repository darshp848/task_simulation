Committed as `4d3ed2b` on `cloud-sims`. Nothing pushed, no cloud resources changed, and no real Dream or Claude calls made.

Built the sim camera/gripper endpoints, checked sim colours on rendered frames, added oracle/API judges, the container, 1,000-match matrix, provisioning/build/submit scripts, and result collection. Files and complete instructions are in [cloud/README.md](cloud/README.md).

Validation:

- Full suite: **318 passed, 1 skipped**, plus the two documented Windows failures.
- Final focused tests: **16 passed**.
- Bash syntax, offline dry-runs, and local render checks passed.
- Smoke match: **301.156 s wall, 136.531 CPU-seconds, 689.34 MiB peak RAM**. Physics advanced 194.05 seconds.
- Linux container installation and OSMesa rendering remain untested until Cloud Build.

Selected one match per `e2-highcpu-2` Spot VM. At an **assumed $0.030/hour**, compute-only costs are:

| Matches | VM cost | Dream usage |
|---:|---:|---:|
| 100 | $0.25 | 500 min |
| 1,000 | $2.50 | 5,000 min |
| 5,000 | $12.50 | 25,000 min |

Startup, retries, API charges, storage, and builds are additional. See [cloud/COST.md](cloud/COST.md).

Run in Git Bash after completing the user actions below:

```bash
export PYTHON='../hill_climb_harness/.venv/Scripts/python.exe'
gcloud auth login
bash cloud/setup.sh --execute
bash cloud/build.sh --tag cloud-v1 --execute
bash cloud/submit.sh --tag cloud-v1 --job hill-climb-pilot-v1 --pilot --parallelism 8 --execute
gcloud batch jobs describe hill-climb-pilot-v1 --location us-central1 --project hill-climb-sims
"$PYTHON" -m cloud.collect --prefix gs://hill-climb-sims-results/runs/hill-climb-pilot-v1 --execute
```

Inspect the completed pilot before running:

```bash
bash cloud/submit.sh --tag cloud-v1 --job hill-climb-full-v1 --parallelism 8 --execute
"$PYTHON" -m cloud.collect --prefix gs://hill-climb-sims-results/runs/hill-climb-full-v1 --local cloud/generated/full --out cloud/generated/full.jsonl --execute
```

`USER ACTION NEEDED:` Install gcloud and authenticate; create/link `hill-climb-sims` to billing account `<BILLING_ACCOUNT_ID>`; accept required API terms and obtain IAM permissions; save Dream credentials locally; upload the Anthropic secret only for API judging; resolve any bucket-name conflict, quota, capacity, or network restrictions; verify current pricing and budget recipients; and check Linux performance and Dream concurrency in the pilot. Exact details are recorded in the README.