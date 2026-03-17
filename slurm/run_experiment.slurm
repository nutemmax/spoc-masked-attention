#!/bin/bash
#SBATCH --job-name=test_run_1k
#SBATCH --output=/home/anastass/spoc-masked-attention/logs/test_run_1k_%j.out
#SBATCH --error=/home/anastass/spoc-masked-attention/logs/test_run_1k_%j.err
#SBATCH --partition=academic
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

PROJECT_DIR=/home/anastass/spoc-masked-attention
CONFIG_PATH=configs/default.yaml

cd "$PROJECT_DIR"

module purge
module load gcc/13.2.0
module load python/3.11.7

source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u scripts/run_experiment.py --config "$CONFIG_PATH"