#!/bin/bash
#SBATCH --job-name=quadratic_0p99
#SBATCH --output=logs/quadratic/%A_%a.out
#SBATCH --error=logs/quadratic/%A_%a.err
#SBATCH --partition=academic
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mail-user=emma.anastassova@epfl.ch
#SBATCH --mail-type=FAIL
#SBATCH --mem=24G
#SBATCH --array=0-25%10

# comment : %5 means max 5 jobs running at once
# 0-19 for original setup

set -euo pipefail
CONFIG_PATH="/home/anastass/spoc-masked-attention/configs/presentation/cov_toeplitz_maskrandom_rho0p99_lambda1e-05_beta1_d500_T50_lr0p001_iter10000.yaml"

ITERS=$(python - <<EOF
import yaml
with open("$CONFIG_PATH", "r") as f:
    config = yaml.safe_load(f)
print(config["training"]["n_steps"])
EOF
)

MASKING_STRATEGY=$(python - <<EOF
import yaml
with open("$CONFIG_PATH", "r") as f:
    config = yaml.safe_load(f)
print(config["data"]["masking_strategy"])
EOF
)

ITERS=$(python - <<EOF
import yaml
with open("$CONFIG_PATH", "r") as f:
    config = yaml.safe_load(f)
print(config["training"]["n_steps"])
EOF
)

COVARIANCE_TYPE=$(python - <<EOF
import yaml
with open("$CONFIG_PATH", "r") as f:
    config = yaml.safe_load(f)
print(config["data"]["covariance_type"])
EOF
)

# ====== CONFIG ======
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
SWEEP_TIMESTAMP=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
SWEEP_DIR_REL="results/mask-${MASKING_STRATEGY}/presentation/iter_${ITERS}/${CONFIG_NAME}/${CONFIG_NAME}_${SWEEP_TIMESTAMP}"
SEED=42
# NTRAIN_CSV="10,30,50,100,150,200,250,500,750,1000,1500,1750,2000,2500,3000,5000,10000,20000,50000,100000"
NTRAIN_CSV="10,20,30,40,50,60,80,100,125,150,175,200,250,300,250,400,500,750,1000,1250,1500,1750,2000,2500,5000,10000"


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
IFS=',' read -ra NTRAINS <<< "$NTRAIN_CSV"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TASK_ID" -ge "${#NTRAINS[@]}" ]; then
  echo "Invalid TASK_ID=$TASK_ID"
  exit 1
fi

NTRAIN=${NTRAINS[$TASK_ID]}

echo "Running config=$CONFIG_NAME n_train=$NTRAIN (task $TASK_ID)"
echo "Saving under: $SWEEP_DIR_REL"

# ====== RUN ======
python -u scripts/run_experiment.py \
  --config "$CONFIG_PATH" \
  --n-train "$NTRAIN" \
  --seed "$SEED" \
  --save-root "$SWEEP_DIR_REL"