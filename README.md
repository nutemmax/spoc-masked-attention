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