#!/bin/bash
#SBATCH --job-name=teacher_ntrain
#SBATCH --output=logs/teacher/teacher_ntrain_%A_%a.out
#SBATCH --error=logs/teacher/teacher_ntrain_%A_%a.err
#SBATCH --partition=academic
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=emma.anastassova@epfl.ch
#SBATCH --mail-type=FAIL
#SBATCH --mem=4G
#SBATCH --array=0-25%13

set -euo pipefail

CONFIG_PATH=$1
# CONFIG_PATH="/home/anastass/spoc-masked-attention/configs/teacher_attention/teacher_attention_init_standard_gaussian_rstar_50_bstar_1_beta_1_d50_T5_lambda1e-05_lr0p001_iter5000.yaml"


# ====== READ CONFIG INFO ======
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

TEACHER_INIT=$(python - <<EOF
import yaml
with open("$CONFIG_PATH", "r") as f:
    config = yaml.safe_load(f)
print(config["teacher"]["init"])
EOF
)

R_STAR=$(python - <<EOF
import yaml
with open("$CONFIG_PATH", "r") as f:
    config = yaml.safe_load(f)
r_star = config["teacher"]["r_star"]
print("d" if r_star is None else r_star)
EOF
)

BETA_STAR=$(python - <<EOF
import yaml
with open("$CONFIG_PATH", "r") as f:
    config = yaml.safe_load(f)
print(config["teacher"]["beta_star"])
EOF
)

# ====== CONFIG ======
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
SWEEP_TIMESTAMP=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
SEED=42

SWEEP_DIR_REL="results/teacher-attention/mask-${MASKING_STRATEGY}/iter_${ITERS}/tuning-lambda-lr-2/${CONFIG_NAME}/${CONFIG_NAME}_${SWEEP_TIMESTAMP}"

NTRAIN_CSV="10,20,30,40,50,60,70,80,90,100,125,150,175,200,250,300,400,500,750,1000,1250,1500,1750,2000,2500,5000"

# ====== SETUP ======
PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

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

echo "Running teacher-attention n_train sweep"
echo "Config:       $CONFIG_PATH"
echo "Config name:  $CONFIG_NAME"
echo "Teacher init: $TEACHER_INIT"
echo "r_star:       $R_STAR"
echo "beta_star:    $BETA_STAR"
echo "n_train:      $NTRAIN"
echo "seed:         $SEED"
echo "task:         $TASK_ID"
echo "Saving under: $SWEEP_DIR_REL"

# ====== RUN ======
python -u scripts/run_teacher_attention_experiment.py \
  --config "$CONFIG_PATH" \
  --n-train "$NTRAIN" \
  --seed "$SEED" \
  --save-root "$SWEEP_DIR_REL"