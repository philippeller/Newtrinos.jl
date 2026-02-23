#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Pass --dry-run to validate all configs without running scans
EXTRA_ARGS=""
if [[ "${1:-}" == "--dry-run" ]]; then
  EXTRA_ARGS="--dry-run"
  echo "=== DRY RUN MODE ==="
fi

mkdir -p logs

for model in NNM NND; do
  for ordering in NO IO; do
    tag="${model}_${ordering}"
    logfile="logs/junotao_${tag}_$(date +%Y-%m-%d).log"
    echo "=== Starting $tag -> $logfile ==="
    ./run-julia-local.sh junotao.jl "$model" "$ordering" $EXTRA_ARGS 2>&1 | tee "$logfile"
    echo "=== Finished $tag ==="
  done
done

echo "All runs complete."
