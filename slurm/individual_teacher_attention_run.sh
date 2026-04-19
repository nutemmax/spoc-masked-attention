#!/bin/bash
#SBATCH --job-name=teacher_attn
#SBATCH --output=logs/teacher_attn_%j.out
#SBATCH --error=logs/teacher_attn_%j.err
#SBATCH --partition=academic
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-user=emma.anastassova@epfl.ch
#SBATCH --mail-type=FAIL

set -euo pipefail

PROJECT_DIR="/home/anastass/spoc-masked-attention"
cd "$PROJECT_DIR"

mkdir -p logs

module purge
module load gcc/13.2.0
module load python/3.11.7

source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

CONFIG_PATH="/home/anastass/spoc-masked-attention/configs/default-teacher-attention.yaml"
NTRAIN=1000
SEED=42

CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
SAVE_ROOT="results/teacher-attention/individual/${CONFIG_NAME}_ntrain_${NTRAIN}"

echo "Running teacher-attention experiment"
echo "Config:      $CONFIG_PATH"
echo "n_train:     $NTRAIN"
echo "seed:        $SEED"
echo "save_root:   $SAVE_ROOT"

python -u scripts/run_teacher_attention_experiment.py \
  --config "$CONFIG_PATH" \
  --n-train "$NTRAIN" \
  --seed "$SEED" \
  --save-root "$SAVE_ROOT"