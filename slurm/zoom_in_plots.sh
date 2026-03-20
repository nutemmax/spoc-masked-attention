#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_plot_summary_zoom.sh <SWEEP_DIR>
#
# Example:
#   bash run_plot_summary_zoom.sh results/mask-last
#   bash run_plot_summary_zoom.sh results/mask-last/toeplitz_rho0p9_experiment

SWEEP_DIR="${1:-}"

if [[ -z "$SWEEP_DIR" ]]; then
    echo "Usage: bash run_plot_summary_zoom.sh <SWEEP_DIR>"
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

# Common zooms for presentation
XMAXS="500,1000,2000"

# Optional custom suffix for output names
SUFFIX="${SUFFIX:-}"

echo "Running zoomed summary plots for: $SWEEP_DIR"
echo "Using xmaxs: $XMAXS"

CMD=(
    "$PYTHON_BIN" scripts/plot_summary_zoom.py
    --sweep-dir "$SWEEP_DIR"
    --include-full
    --xmaxs "$XMAXS"
)

if [[ -n "$SUFFIX" ]]; then
    CMD+=(--suffix "$SUFFIX")
fi

echo "Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done."