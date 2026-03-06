# SPOC — Masked Single-Head Attention

This repository contains the implementation for a research project on **masked self-supervised learning in single-head attention models** in the **high-dimensional regime**.

The goal is to study how attention weights are learned when reconstructing masked tokens from correlated Gaussian sequences, and to analyze the resulting spectral properties of the learned attention matrix.

The project is inspired by:

- *Single-Head Attention in High Dimensions: A Theory of Generalization, Weight Spectra, and Scaling Laws*
- *A Random Matrix Theory of Masked Self-Supervised Regression*

---

# Project Overview

We study a simplified Transformer-like model where a **single-head attention mechanism** is trained to reconstruct masked tokens.

Key aspects:

- Gaussian synthetic data
- token correlations controlled by a covariance matrix
- random single-token masking
- reconstruction loss
- comparison with the **Bayes-optimal predictor**

The project investigates:

- training loss
- population risk
- weight norms
- eigenvalue spectrum of the learned attention matrix

as a function of the scaling parameter

\[
\alpha = \frac{n}{d^2}
\]

---

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
spoc-masked-attention
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── experiment_identity.yaml
│   └── experiment_tridiagonal.yaml
|
├── notebooks/          # exploratory analysis
│   ├── 01_sanity_checks.ipynb
│   ├── 02_first_training_runs.ipynb
│   └── 03_spectrum_analysis.ipynb
|
├── scripts/            # experiment entrypoints
│   ├── run_experiment.py
│   ├── sweep_alpha.py
│   ├── evaluate_bayes_baseline.py
│   └── plot_results.py         
│
├── src/
│   ├── data/           # data generation and masking
│   ├── models/         # attention model
│   ├── training/       # training loops and losses
│   ├── baselines/      # Bayes-optimal predictor
│   ├── evaluation/     # metrics and spectra
│   ├── visualization/  # plotting utilities
│   └── utils/          # misc helpers
│
├── results/            # saved experiment outputs
│   ├── raw/
│   ├── processed/
│   └── figures/
├── tests/              # unit tests
│   ├── test_covariance.py
│   ├── test_masking.py
│   ├── test_bayes.py
│   └── test_attention.py
└── experiments/        # experiment logs/checkpoints
    ├── logs/
    ├── checkpoints/
    └── summaries/ 
```