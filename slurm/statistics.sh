# ====== SETUP =======
PROJECT_DIR=/home/anastass/spoc-masked-attention
cd "$PROJECT_DIR"
module purge
module load gcc/13.2.0
module load python/3.11.7
source .venv/bin/activate
SWEEP_ROOT="/home/anastass/spoc-masked-attention/results/collective/"

# ========== COLLECTIVE STATISTICS ============
# echo "Aggregating sweeps under: $SWEEP_ROOT"
# python -u scripts/aggregate_teacher_attention_sweep.py --sweep-dir "$SWEEP_ROOT/" --force

# ===== KAPPA STAR : recovery crossings and cosine similarity plotted by d =========

# kappa_star 1
python scripts/recovery_crossings.py \
  --root $SWEEP_ROOT/kappa_star_1/

python -u scripts/plot_cosine_by_d.py \
  --root $SWEEP_ROOT/kappa_star_1

# kappa_star 0p8
python scripts/recovery_crossings.py \
  --root $SWEEP_ROOT/kappa_star_0p8/

python -u scripts/plot_cosine_by_d.py \
  --root $SWEEP_ROOT/kappa_star_0p8

# kappa_star 0p6
python scripts/recovery_crossings.py \
  --root $SWEEP_ROOT/kappa_star_0p6/

python -u scripts/plot_cosine_by_d.py \
  --root $SWEEP_ROOT/kappa_star_0p6

# kappa_star 0p4
python -u scripts/plot_cosine_by_d.py \
  --root $SWEEP_ROOT/kappa_star_0p4

python scripts/recovery_crossings.py \
  --root $SWEEP_ROOT/kappa_star_0p4/

# kappa_star 0p2
python scripts/recovery_crossings.py \
  --root $SWEEP_ROOT/kappa_star_0p2/

python -u scripts/plot_cosine_by_d.py \
  --root $SWEEP_ROOT/kappa_star_0p2

# Crossings over kappa
python scripts/crossings_over_kappa.py \
  --root $SWEEP_ROOT/iter_5000
