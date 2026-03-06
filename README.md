# SPOC — Masked Single-Head Attention

This repository contains the implementation for a research project on **masked self-supervised learning in single-head attention models** in the **high-dimensional regime**.

The goal is to study how attention weights are learned when reconstructing masked tokens from correlated Gaussian sequences, and to analyze the resulting spectral properties of the learned attention matrix.

The project is inspired by the following papers:

- *Single-Head Attention in High Dimensions: A Theory of Generalization, Weight Spectra, and Scaling Laws*
- *A Random Matrix Theory of Masked Self-Supervised Regression*

# Installation

Clone the repository:

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
├── README.md                           # project overview, setup instructions, usage
├── requirements.txt                    # pinned Python dependencies
├── .gitignore                          # files and folders excluded from Git
│
├── configs/                            # experiment configuration files
│   ├── default.yaml                    # default configuration used for standard runs
│   ├── experiment_identity.yaml        # config for independent-token baseline (Sigma = I)
│   └── experiment_tridiagonal.yaml     # config for structured tridiagonal covariance
│
├── notebooks/                          # exploratory notebooks, sanity checks, analysis
│
├── scripts/                            # executable entrypoints for experiments
│   ├── run_experiment.py               # runs one experiment from a given config
│   ├── sweep_alpha.py                  # runs multiple experiments over alpha values
│   ├── evaluate_bayes_baseline.py      # computes Bayes-optimal baseline performance
│   └── plot_results.py                 # generates plots from saved results
│
├── src/                                # main source code
│   ├── data/                           # synthetic data generation and masking logic
│   │   ├── __init__.py                 # marks data as a Python module
│   │   ├── generator.py                # generates Gaussian sequence samples X
│   │   ├── covariance.py               # builds covariance matrices (identity, tridiagonal, etc.)
│   │   └── masking.py                  # applies masking and constructs corrupted inputs
│   │
│   ├── models/                         # model definitions
│   │   ├── __init__.py                 # marks models as a Python module
│   │   ├── attention.py                # single-head tied-attention model and forward pass
│   │   └── parameters.py               # helper functions for W, S, initialization, shapes
│   │
│   ├── training/                       # training logic
│   │   ├── __init__.py                 # marks training as a Python module
│   │   ├── losses.py                   # reconstruction loss and regularization terms
│   │   ├── trainer.py                  # training loop and experiment orchestration
│   │   └── optimizer.py                # optimizer creation and training utilities
│   │
│   ├── baselines/                      # baseline methods
│   │   ├── __init__.py                 # marks baselines as a Python module
│   │   └── bayes.py                    # Bayes-optimal Gaussian reconstruction baseline
│   │
│   ├── evaluation/                     # evaluation metrics and analysis
│   │   ├── __init__.py                 # marks evaluation as a Python module
│   │   ├── metrics.py                  # generic experiment metrics
│   │   ├── spectra.py                  # eigenvalues, spectral concentration, spectrum utilities
│   │   └── risk.py                     # population risk and related evaluation functions
│   │
│   ├── visualization/                  # plotting and visualization helpers
│   │   ├── __init__.py                 # marks visualization as a Python module
│   │   ├── plots.py                    # main plotting functions for losses, risks, norms, spectra
│   │   └── attention_maps.py           # visualization of attention matrices
│   │
│   └── utils/                          # miscellaneous utilities
│       ├── __init__.py                 # marks utils as a Python module
│       ├── io.py                       # saving/loading results, configs, arrays, checkpoints
│       ├── seed.py                     # random seed handling and reproducibility
│       └── config.py                   # configuration loading/parsing helpers
│
├── results/                            # saved outputs from experiments
│   ├── raw/                            # raw outputs directly produced by runs
│   ├── processed/                      # processed summaries and aggregated results
│   └── figures/                        # plots and generated figures
│
├── tests/                              # unit tests
│   ├── test_covariance.py              # tests for covariance matrix generation
│   ├── test_masking.py                 # tests for masking logic
│   ├── test_bayes.py                   # tests for Bayes baseline formulas
│   └── test_attention.py               # tests for attention model behavior
│
└── experiments/                        # experiment bookkeeping
    ├── logs/                           # run logs and console outputs
    ├── checkpoints/                    # saved model weights/checkpoints
    └── summaries/                      # compact summaries of completed experiments
```