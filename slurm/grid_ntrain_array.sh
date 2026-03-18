#!/bin/bash
#SBATCH --mail-user=emma.anastassova@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=academic
set -euo pipefail

PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

ITERS=1000
CONFIG_DIR="configs/generated_grid_biggerT_masklast_$ITERS"

for config in "$CONFIG_DIR"/*.yaml; do
  echo "Submitting n_train sweep for $config"
  sbatch slurm/ntrain_array.slurm "$config"
done