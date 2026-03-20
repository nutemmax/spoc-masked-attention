# SWEEP_ROOT="results/mask-last/"
SWEEP_ROOT="results/mask-last/presentation/"

# ====== SETUP ======
PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

module purge
module load gcc/13.2.0
module load python/3.11.7

source .venv/bin/activate

# ====== RUN ======
echo "Aggregating sweeps under: $SWEEP_ROOT"

python -u scripts/aggregate_sweep.py --sweep-dir "$SWEEP_ROOT" --force