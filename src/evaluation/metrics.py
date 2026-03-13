from __future__ import annotations
import numpy as np

def attention_matrix(W : np.ndarray) -> np.ndarray:
    """Computes the attention matrix S = W W^T"""
    if W.ndim != 2:
        raise ValueError("W must be a 2d array.")
    d, r = W.shape
    return (W @ W.T)/ np.sqrt(d*r)

def compute_eigenvalues_symmetric(matrix : np.ndarray) -> np.ndarray:
    """Computes the eigenvalues of a symmetric matrix"""
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2d array.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be square.")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Input must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.sort(eigenvalues)[::-1]

def compute_trace(matrix : np.ndarray) -> np.float64:
    """Computes the trace of a matrix"""
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2d array.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be square.")
    return np.trace(matrix)

def compute_frobenius_norm(matrix : np.ndarray) -> float:
    """Computes the Frobenius norm of a matrix"""
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
    """Compute spectral observables associated with W."""
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





