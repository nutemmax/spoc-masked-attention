# SPOC — Masked Single-Head Attention

This repository contains the implementation for a research project on **masked self-supervised learning in single-head attention models** in the **high-dimensional regime**.

The goal is to study how attention weights are learned when reconstructing masked tokens from correlated Gaussian sequences, and to analyze the resulting spectral properties of the learned attention matrix.

References (NB to keep track of):

- *Single-Head Attention in High Dimensions: A Theory of Generalization, Weight Spectra, and Scaling Laws*
- *A Random Matrix Theory of Masked Self-Supervised Regression*

# Installation

Clone the repo:

```bash
git clone https://github.com/<username>/spoc-masked-attention.git
cd spoc-masked-attention
```
Create the virtual environment:

```bash
python3 -m venv spoc
source spoc/bin/activate
```

Install dependencies : 
```bash
pip install -r requirements.txt
```

# Repo structure
```bash
spoc-masked-attention/
├── configs/                          # YAML configuration files defining experiments
├── logs/                             # Slurm stdout/stderr logs from cluster runs
├── notebooks/
│   └── hyperparam_sweep_analysis.ipynb  
├── results/                          # all experiment outputs (runs, sweeps, metrics, plots)
├── scripts/                          # main Python entrypoints for running and analyzing experiments
│   ├── aggregate_sweep.py            # aggregates runs into summary metrics, plots, and sweep_config.json
│   ├── evaluate_bayes_baseline.py    # computes Bayes-optimal baseline performance
│   ├── repo_tree.py                  # utility to print repository structure
│   ├── run_experiment.py             # runs a single experiment from a config
│   └── sweep_alpha.py                # (legacy/local) script for sweeping over alpha values
├── slurm/                            # Slurm job scripts for running experiments on the cluster
│   ├── old/                          # deprecated or older job scripts
│   │   ├── grid_sweep_alpha.sh       # old grid search script
│   │   ├── run_experiment.slurm      # old single-run Slurm script
│   │   └── small_grid_10k_iter.sh    # old experiment setup
│   ├── aggregate_sweep.slurm         # runs aggregation over completed sweeps
│   ├── alpha_array.slurm             # array job for sweeping over alpha values
│   └── ntrain_array.slurm            # array job for sweeping over training set sizes (n_train)
├── src/                              # core source code (modular, reusable components)
│   ├── baselines/                    # baseline methods
│   │   ├── __init__.py
│   │   └── bayes.py                  # Bayes-optimal Gaussian reconstruction baseline
│   ├── data/                         # synthetic data generation and preprocessing
│   │   ├── __init__.py
│   │   ├── covariance.py             # covariance matrix construction (identity, structured, etc.)
│   │   ├── generator.py              # Gaussian sequence generation
│   │   └── masking.py                # masking mechanism for corrupted inputs
│   ├── evaluation/                   # evaluation metrics and analysis tools
│   │   ├── __init__.py
│   │   └── metrics.py                # performance, spectral, and convergence metrics
│   ├── models/                       # model definitions
│   │   ├── __init__.py
│   │   └── attention.py              # tied single-head attention model
│   ├── training/                     # training logic
│   │   ├── __init__.py
│   │   ├── losses.py                 # loss functions and regularization
│   │   └── trainer.py                # training loop and experiment orchestration
│   └── utils/                        # utility functions
│       ├── __init__.py
│       ├── config.py                 # config loading and override logic
│       ├── io.py                     # saving/loading experiment outputs
│       └── plots.py                  # plotting utilities
├── .gitignore                        # ignored files (envs, cache, results, etc.)
├── cluster-commands.sh               # helper commands for running jobs on the cluster
├── README.md                         # project documentation and usage instructions
└── requirements.txt                  # Python dependencies
```