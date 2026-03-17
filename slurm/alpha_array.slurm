#!/bin/bash
#SBATCH --job-name=alpha_sweep
#SBATCH --output=logs/alpha_%A_%a.out
#SBATCH --error=logs/alpha_%A_%a.err
#SBATCH --partition=academic
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-5

set -euo pipefail

# ====== CONFIG ======
CONFIG_PATH="configs/default.yaml"
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
SWEEP_TIMESTAMP=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
SWEEP_DIR_REL="results/sweep_alpha/${CONFIG_NAME}/${CONFIG_NAME}_${SWEEP_TIMESTAMP}"
SEED=42
ALPHAS_CSV="0.5,1.0,2.0,3.0,5.0,10.0"

# ====== SETUP ======
PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

mkdir -p logs

module purge
module load gcc/13.2.0
module load python/3.11.7

source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ====== SELECT VALUE ======
IFS=',' read -ra ALPHAS <<< "$ALPHAS_CSV"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TASK_ID" -ge "${#ALPHAS[@]}" ]; then
  echo "Invalid TASK_ID=$TASK_ID"
  exit 1
fi

ALPHA=${ALPHAS[$TASK_ID]}

echo "Running alpha=$ALPHA (task $TASK_ID)"
echo "Saving under: $SWEEP_DIR_REL"

# ====== RUN ======
python -u scripts/run_experiment.py \
  --config "$CONFIG_PATH" \
  --alpha "$ALPHA" \
  --seed "$SEED" \
  --save-root "$SWEEP_DIR_REL"