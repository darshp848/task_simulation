#!/bin/sh
# Parallel Dream-1 matches on the simulated relay table, one container per worker (own network namespace, so
# every sim_station uses its default ports). The Dream key is mounted read-only from ~/.dreamlabs/credentials.
#   [PLAN=path/in/repo.yaml] cluster/run_matches.sh <workers> <matches-per-worker> [run label]
set -eu
WORKERS=${1:-3}
MATCHES=${2:-2}
LABEL=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
D=/cluster/shared/dev/darsh/projects/hill_climb_sim
OUT=$D/runs/$LABEL
mkdir -p "$OUT"
for i in $(seq 1 "$WORKERS"); do
  docker rm -f "hc-match-$i" >/dev/null 2>&1 || true
  docker run -d --name "hc-match-$i" \
    -v "$D/repo:/repo:ro" -v "$OUT:/out" -v "$HOME/.dreamlabs/credentials:/root/.dreamlabs/credentials:ro" \
    -w /repo --device /dev/dri hill-climb-sim sh -c "
      python sim_station.py --policy dream --scene hill_climb > /out/w$i-station.log 2>&1 &
      for t in \$(seq 300); do python -c 'import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8091/status\")' 2>/dev/null && break; sleep 1; done
      python experiments/sim_match.py --matches $MATCHES --out /out/w$i --plan ${PLAN:-plans/hill_climb_v4.yaml} > /out/w$i-match.log 2>&1
      echo exit=\$? >> /out/w$i-match.log" >/dev/null
  echo "started hc-match-$i -> $OUT/w$i"
  sleep 20   # stagger model warm-up
done
