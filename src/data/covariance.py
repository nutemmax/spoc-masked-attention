from __future__ import annotations

import numpy as np


def make_identity_covariance(T: int) -> np.ndarray:
    """
    Return the identity covariance matrix of size T x T.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    return np.eye(T, dtype=np.float64)


def make_tridiagonal_covariance(T: int, rho: float) -> np.ndarray:
    """
    Return the tridiagonal covariance matrix with ones on the diagonal
    and rho on the first off-diagonals
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")

    sigma = np.eye(T, dtype=np.float64)

    for i in range(T - 1):
        sigma[i, i + 1] = rho
        sigma[i + 1, i] = rho

    return sigma


def build_covariance(covariance_type: str, T: int, rho: float | None = None) -> np.ndarray:
    """
    Factory function for covariance matrices.
    """
    covariance_type = covariance_type.lower()

    if covariance_type == "identity":
        return make_identity_covariance(T)

    if covariance_type == "tridiagonal":
        if rho is None:
            raise ValueError("rho must be provided for tridiagonal covariance.")
        return make_tridiagonal_covariance(T, rho)

    raise ValueError(
        f"Unknown covariance_type='{covariance_type}'. "
        "Supported values are: 'identity', 'tridiagonal'."
    )


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Check whether a matrix is PD using eigenvals
    """
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigvals > tol))