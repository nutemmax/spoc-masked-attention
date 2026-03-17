#!/bin/bash
#SBATCH --job-name=aggregate_sweep
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
#SBATCH --partition=academic
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

# ====== CONFIG (edit this only) ======
SWEEP_ROOT="results/sweep_ntrain/default"
# SWEEP_ROOT="results/sweep_alpha/default"

# ====== SETUP ======
PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

module purge
module load gcc/13.2.0
module load python/3.11.7

source .venv/bin/activate

# ====== RUN ======
echo "Aggregating sweeps under: $SWEEP_ROOT"

python -u scripts/aggregate_sweep.py --sweep-dir "$SWEEP_ROOT"