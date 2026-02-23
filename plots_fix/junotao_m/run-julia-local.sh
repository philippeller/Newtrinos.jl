#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JULIA_DEPOT_PATH="$ROOT/.julia_depot"
export JULIA_PROJECT="$ROOT"
export JULIA_NUM_THREADS=auto
exec "$HOME/.juliaup/bin/julialauncher" +1.11 "$@"
