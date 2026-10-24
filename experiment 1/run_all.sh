#!/usr/bin/env bash
# Launch all four benchmarks concurrently, preserving logs and exit codes.
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$EXPERIMENT_DIR")"
cd "$PROJECT_DIR"
MODAL_BIN="${MODAL_BIN:-$PROJECT_DIR/env/bin/modal}"
if [[ ! -x "$MODAL_BIN" ]]; then
  MODAL_BIN="$(command -v modal)"
fi
mkdir -p "$EXPERIMENT_DIR/logs"
LOG_DIR="$(mktemp -d "$EXPERIMENT_DIR/logs/$(date +%Y%m%d_%H%M%S)_XXXXXX")"
printf 'Logs: %s\n' "$LOG_DIR"
pids=()
for name in text2sql translation tool_call debug; do
  (
    set +e
    "$MODAL_BIN" run modal_app.py --config "experiment 1/config_${name}.yaml"
    status=$?
    printf '%s\n' "$status" > "$LOG_DIR/${name}.exit_code"
    exit "$status"
  ) > "$LOG_DIR/${name}.log" 2>&1 &
  pids+=("$!")
  printf '%s\t%s\n' "$name" "$!" >> "$LOG_DIR/pids.tsv"
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
printf 'All runs finished; see %s for logs and exit codes.\n' "$LOG_DIR"
exit "$status"
