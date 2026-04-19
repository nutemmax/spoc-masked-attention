from __future__ import annotations
import numpy as np
import torch
from torch import Tensor

def attention_matrix(W : np.ndarray) -> np.ndarray:
    """Computes the attention matrix S = W W^T/sqrt(dr)"""
    if W.ndim != 2:
        raise ValueError("W must be a 2d array.")
    d, r = W.shape
    return (W @ W.T)/ np.sqrt(d*r)

def compute_eigenvalues_symmetric(matrix : np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2d array.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be square.")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Input must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.sort(eigenvalues)[::-1]

def compute_trace(matrix : np.ndarray) -> np.float64:
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2d array.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be square.")
    return np.trace(matrix)

def compute_frobenius_norm(matrix : np.ndarray) -> float:
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2d array.")
    return float(np.linalg.norm(matrix, 'fro'))

def compute_weights_norm(W : np.ndarray) -> float:
    d = W.shape[0]
    norm = compute_frobenius_norm(W)**2/d
    return norm


def compute_spectral_concentration(eigvals: np.ndarray) -> float:
    """
    Compute R1 = lambda_1 / Tr(S).
    """
    if eigvals.ndim != 1:
        raise ValueError("eigvals must be a 1d array.")
    if eigvals.size == 0:
        raise ValueError("eigvals must be non-empty.")

    trace = float(np.sum(eigvals))
    if np.isclose(trace, 0.0):
        return float("nan")

    return float(eigvals[0] / trace)

def compute_spectral_observables(W: np.ndarray) -> dict[str, float | np.ndarray]:
    """Compute spectral metrics associated with W."""
    S = attention_matrix(W)
    eigvals = compute_eigenvalues_symmetric(S)
    trace = float(compute_trace(S))
    top_eig = float(eigvals[0])
    min_eig = float(eigvals[-1])
    spectral_concentration = compute_spectral_concentration(eigvals)
    weight_norm = compute_weights_norm(W)

    return {
        "S": S,
        "eigenvalues": eigvals,
        "trace": trace,
        "top_eigenvalue": top_eig,
        "min_eigenvalue": min_eig,
        "spectral_concentration": spectral_concentration,
        "weight_norm": weight_norm,
    }

def masked_mse_per_coordinate(
    X: np.ndarray,
    predictions: np.ndarray,
    mask_indices: np.ndarray,
) -> float:
    """
    Computes the masked reconstruction error (MSE) per coordinate
    Inputs : 
        X : (n_samples, T, d) array of input samples
        predictions : (n_samples, d) array of predicted masked tokens
        mask_indices : (n_samples, ) array of masked token indices per each sample
    Returns : 
        MSE : scalar value of the MSE per coordinate, avg over the dataset and over d coordinates
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, _, d = X.shape
    if predictions.shape != (n_samples, d):
        raise ValueError("predictions must have shape (n_samples, d).")
    if mask_indices.shape != (n_samples,):
        raise ValueError("mask_indices must have shape (n_samples,).")

    true_masked_tokens = X[np.arange(n_samples), mask_indices, :]
    mse = np.mean(np.sum((true_masked_tokens - predictions) ** 2, axis=1) / d)

    return float(mse)


def compute_effective_rank(eigvals: np.ndarray, eps: float = 1e-12) -> float:
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = eigvals[eigvals > eps]

    if eigvals.size == 0:
        return 0.0

    p = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def matrix_cosine_similarity_torch(A: Tensor, B: Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between two matrices."""
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    denom = torch.linalg.norm(A_flat) * torch.linalg.norm(B_flat)
    if denom.item() <= eps:
        return float("nan")

    return float(torch.dot(A_flat, B_flat).item() / denom.item())


def relative_frobenius_error_torch(A: Tensor, B: Tensor, eps: float = 1e-12) -> float:
    """Relative Frobenius error ||A-B||_F / ||B||_F."""
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")

    denom = torch.linalg.norm(B, ord="fro")
    if denom.item() <= eps:
        return float("nan")
    return float((torch.linalg.norm(A - B, ord="fro") / denom).item())


@torch.no_grad()
def attention_level_error_torch(
    model,
    X_tilde: Tensor,
    A_star: Tensor,
    normalize_by_T2: bool = True,
) -> float:
    """Mean attention-level error between student and teacher attention."""
    if X_tilde.ndim != 3:
        raise ValueError("X_tilde must have shape (batch, T, d).")
    if A_star.ndim != 3:
        raise ValueError("A_star must have shape (batch, T, T).")
    if X_tilde.shape[0] != A_star.shape[0]:
        raise ValueError("X_tilde and A_star must have the same batch size.")
    model.eval()
    A_student = model.attention_weights(X_tilde)
    if A_student.shape != A_star.shape:
        raise ValueError("A_student and A_star must have the same shape.")
    
    sq_error = torch.sum((A_student - A_star) ** 2, dim=(1, 2))
    if normalize_by_T2:
        T = A_star.shape[1]
        sq_error = sq_error / (T * T)

    return float(torch.mean(sq_error).item())


@torch.no_grad()
def teacher_recovery_metrics_torch(
    model,
    S_star: Tensor,
    X_tilde: Tensor | None = None,
    A_star: Tensor | None = None,
) -> dict[str, float]:
    """Teacher-recovery metrics."""
    S = model.attention_matrix()

    metrics = {
        "cosine_S_S_star": matrix_cosine_similarity_torch(S, S_star),
        "relative_error_S_S_star": relative_frobenius_error_torch(S, S_star),
    }

    if X_tilde is not None and A_star is not None:
        metrics["attention_level_error"] = attention_level_error_torch(
            model=model,
            X_tilde=X_tilde,
            A_star=A_star,
            normalize_by_T2=True,
        )

    return metrics

