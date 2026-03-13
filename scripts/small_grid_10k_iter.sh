#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="configs/grids_experiments"
mkdir -p "$CONFIG_DIR"

ALPHAS="0.5,1.0,2.0,3.0,5.0"

TS=(5)
LAMBDAS=(0.00001 0.0001 0.001)
LEARNING_RATES=(0.001 0.01 0.1)

D=50
R=50
BETA=1.0
N_STEPS=10000
SEED=0
N_POPULATION=5000

# Fixed covariance choice for this small grid
COVARIANCE_TYPE="toeplitz"
RHO=0.5

echo "Starting small grid..."
echo "T values: ${TS[*]}"
echo "lambda values: ${LAMBDAS[*]}"
echo "learning rates: ${LEARNING_RATES[*]}"
echo "alpha sweep: $ALPHAS"

count=0

for T in "${TS[@]}"; do
  for lambda_reg in "${LAMBDAS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      count=$((count + 1))

      config_path="${CONFIG_DIR}/config_T${T}_lsambda${lambda_reg}_lr${lr}_beta${BETA}_D${D}.yaml"

      cat > "$config_path" <<EOF
experiment:
  save_root: results/individual_runs
  run_name: null
  seed: $SEED
  experiment_number: null

data:
  T: $T
  d: $D
  covariance_type: $COVARIANCE_TYPE
  rho: $RHO
  length_scale: null
  eta: null
  mask_value: 1.0

model:
  r: $R
  beta: $BETA
  normalize_sqrt_d: false
  dtype: float64
  device: cpu

training:
  alpha: 1.0
  n_steps: $N_STEPS
  learning_rate: $lr
  lambda_reg: $lambda_reg

evaluation:
  n_population: $N_POPULATION
EOF

      echo
      echo "============================================================"
      echo "[$count] Running sweep:"
      echo "  T=$T | lambda=$lambda_reg | lr=$lr | d=r=$D | beta=$BETA | steps=$N_STEPS"
      echo "  covariance=$COVARIANCE_TYPE | rho=$RHO"
      echo "============================================================"

      python scripts/sweep_alpha.py \
        --config "$config_path" \
        --alphas "$ALPHAS"
    done
  done
done

echo
echo "All small-grid sweeps completed."