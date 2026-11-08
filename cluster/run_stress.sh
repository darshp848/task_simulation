#!/bin/sh
# Offline stress sweep (experiments/stress_match.py) in the hill-climb-sim image: no Dream key, no Anthropic key,
# no GPU. One container, JOBS processes; each match is ~5 s of CPU.
#   cluster/run_stress.sh [seeds] [jobs] [label] [scenario ...]
#   cluster/run_stress.sh 50 7 sweep1                   # every scenario x 50 seeds on 7 cores
#   SEED0=51 cluster/run_stress.sh 50 2 sweep1 happy    # another node takes seeds 51-100 of one scenario
# Results: $OUT/<label>/matches.jsonl (one line per match) and summary.txt.
set -eu
SEEDS=${1:-20}
JOBS=${2:-$(($(nproc) - 1))}
LABEL=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
shift 3 2>/dev/null || shift $#
D=${D:-/cluster/shared/dev/darsh/projects/task_simulation}
OUT=$D/runs_stress/$LABEL
mkdir -p "$OUT"
ARGS=""
for s in "$@"; do ARGS="$ARGS -s $s"; done
docker run --rm -v "$D/repo:/repo:ro" -v "$OUT:/out" -w /repo hill-climb-sim \
  python experiments/stress_match.py --seeds "$SEEDS" --seed0 "${SEED0:-1}" -j "$JOBS" --out /out/matches.jsonl $ARGS \
  | tee "$OUT/summary.txt"
