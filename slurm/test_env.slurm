#!/bin/bash
#SBATCH --job-name=spoc_test
#SBATCH --output=logs/spoc_test_%j.out
#SBATCH --error=logs/spoc_test_%j.err
#SBATCH --partition=academic
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

cd /home/anastass/spoc-masked-attention
mkdir -p logs

module purge
module load gcc/13.2.0
module load python/3.11.7

source .venv/bin/activate

echo "=== Basic job info ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

echo "=== Python info ==="
which python
python --version

echo "=== Import test ==="
python -c "import torch, numpy, matplotlib, yaml, pandas; print('Imports OK')"

echo "=== Done ==="