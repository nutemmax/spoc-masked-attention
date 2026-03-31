#!/bin/bash
#SBATCH --job-name=diag_long
#SBATCH --output=logs/diag_long_%j.out
#SBATCH --error=logs/diag_long_%j.err
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

CONFIG_PATH="/home/anastass/spoc-masked-attention/configs/maskrandom/iter1000/cov_toeplitz_maskrandom_rho0p9_lambda1e-05_beta1_d50_T5_lr0p001_iter1000.yaml"
NTRAIN=2000
SEED=42

CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
SAVE_ROOT="results/individual/${CONFIG_NAME}_ntrain_${NTRAIN}"

echo "Running diagnostic long run"
echo "Config:      $CONFIG_PATH"
echo "n_train:     $NTRAIN"
echo "seed:        $SEED"
echo "save_root:   $SAVE_ROOT"

python -u scripts/run_experiment_new.py \
  --config "$CONFIG_PATH" \
  --n-train "$NTRAIN" \
  --seed "$SEED" \
  --save-root "$SAVE_ROOT"