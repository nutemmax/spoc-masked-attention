#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
# bash scripts/grid_sweep_alpha.sh
# This script creates temporary YAML configs and launches sweep_alpha.py over many hyperparameter combinations.

CONFIG_DIR="configs/generated_gridsearch"
mkdir -p "$CONFIG_DIR"

# Alpha values for sweep_alpha.py
ALPHAS="0.5,1.0,2.0,3.0,5.0"

# Grid values
DS=(50)
TS=(4 6)
BETAS=(1.0 10.0)
LAMBDAS=(0.0 0.1 1.0)

# Structured covariance rho values
RHOS=(0.2 0.5 0.8)

# Shared defaults
N_STEPS=500
LEARNING_RATE=0.001
MASK_VALUE=1.0
DTYPE="float64"
DEVICE="cpu"
SEED=0
N_POPULATION=5000
NORMALIZE_SQRT_D="True"

echo "Starting grid sweep..."
echo "Alpha list: $ALPHAS"

count=0

# -------------------------------------------------------------------
# Identity covariance block
# -------------------------------------------------------------------
for d in "${DS[@]}"; do
  for T in "${TS[@]}"; do
    for lambda_reg in "${LAMBDAS[@]}"; do
      for beta in "${BETAS[@]}"; do

        count=$((count + 1))

        config_path="${CONFIG_DIR}/config_identity_d${d}_T${T}_lambda${lambda_reg}_beta${beta}.yaml"

        cat > "$config_path" <<EOF
experiment:
  save_root: results/individual_runs
  run_name: null
  seed: $SEED
  experiment_number: null

data:
  T: $T
  d: $d
  covariance_type: identity
  rho: null
  length_scale: null
  eta: null
  mask_value: $MASK_VALUE

model:
  r: $d
  beta: $beta
  normalize_sqrt_d: $NORMALIZE_SQRT_D
  dtype: $DTYPE
  device: $DEVICE

training:
  alpha: 1.0
  n_steps: $N_STEPS
  learning_rate: $LEARNING_RATE
  lambda_reg: $lambda_reg

evaluation:
  n_population: $N_POPULATION
EOF

        echo
        echo "============================================================"
        echo "[$count] Running sweep:"
        echo "  cov=identity | d=r=$d | T=$T | lambda=$lambda_reg | beta=$beta"
        echo "============================================================"

        python scripts/sweep_alpha.py \
          --config "$config_path" \
          --alphas "$ALPHAS"

      done
    done
  done
done

# -------------------------------------------------------------------
# Toeplitz and tridiagonal covariance blocks
# -------------------------------------------------------------------
for cov in toeplitz tridiagonal; do
  for rho in "${RHOS[@]}"; do
    for d in "${DS[@]}"; do
      for T in "${TS[@]}"; do
        for lambda_reg in "${LAMBDAS[@]}"; do
          for beta in "${BETAS[@]}"; do

            count=$((count + 1))

            config_path="${CONFIG_DIR}/config_${cov}_rho${rho}_d${d}_T${T}_lambda${lambda_reg}_beta${beta}.yaml"

            cat > "$config_path" <<EOF
experiment:
  save_root: results/individual_runs
  run_name: null
  seed: $SEED
  experiment_number: null

data:
  T: $T
  d: $d
  covariance_type: $cov
  rho: $rho
  length_scale: null
  eta: null
  mask_value: $MASK_VALUE

model:
  r: $d
  beta: $beta
  normalize_sqrt_d: $NORMALIZE_SQRT_D
  dtype: $DTYPE
  device: $DEVICE

training:
  alpha: 1.0
  n_steps: $N_STEPS
  learning_rate: $LEARNING_RATE
  lambda_reg: $lambda_reg

evaluation:
  n_population: $N_POPULATION
EOF

            echo
            echo "============================================================"
            echo "[$count] Running sweep:"
            echo "  cov=$cov | rho=$rho | d=r=$d | T=$T | lambda=$lambda_reg | beta=$beta"
            echo "============================================================"

            python sweep_alpha.py \
              --config "$config_path" \
              --alphas "$ALPHAS"

          done
        done
      done
    done
  done
done

echo
echo "All sweeps completed."