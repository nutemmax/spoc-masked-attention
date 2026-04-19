#!/bin/bash
#SBATCH --mail-user=emma.anastassova@epfl.ch
#SBATCH --output=logs/grids/teacher_grids_%A_%a.out
#SBATCH --error=logs/grids/teacher_grids_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --partition=academic

set -euo pipefail

PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

mkdir -p logs/grids

CONFIG_DIR="/home/anastass/spoc-masked-attention/configs/teacher_attention/tuning-scaled-gaussian-init/"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Config directory does not exist: $CONFIG_DIR"
  exit 1
fi

shopt -s nullglob
CONFIGS=("$CONFIG_DIR"/*.yaml)

if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "No yaml configs found in: $CONFIG_DIR"
  exit 1
fi

echo "Submitting teacher-attention n_train sweeps"
echo "Config dir: $CONFIG_DIR"
echo "Number of configs: ${#CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
  echo "Submitting n_train sweep for $config"
  sbatch slurm/linear_ntrain_teacher_attention.sh "$config"
done