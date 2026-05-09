# ====== SETUP ======
PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"

module purge
module load gcc/13.2.0
module load python/3.11.7
source .venv/bin/activate

# ====== RUN ======
SWEEP_ROOT="/home/anastass/spoc-masked-attention/results/teacher-attention/iter_5000/lambda-scaling-test-kappa0p6/"
echo "Aggregating sweeps under: $SWEEP_ROOT"
python -u scripts/aggregate_teacher_attention_sweep.py --sweep-dir "$SWEEP_ROOT" --force