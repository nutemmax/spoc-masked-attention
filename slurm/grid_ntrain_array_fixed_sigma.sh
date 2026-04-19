#!/bin/bash
#SBATCH --mail-user=emma.anastassova@epfl.ch
#SBATCH --output=logs/grids/grids_%A_%a.out
#SBATCH --error=logs/grids/grids_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=academic
set -euo pipefail

PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

# ITERS=5000
# CONFIG_DIR="configs/generated_grid_maskrandom_$ITERS"

CONFIG_DIR="/home/anastass/spoc-masked-attention/configs/numerics-maskrandom/batch3"

for config in "$CONFIG_DIR"/*.yaml; do
  echo "Submitting n_train sweep for $config in $CONFIG_DIR"
  sbatch slurm/linear_ntrain_fixed_sigma.slurm "$config"
done