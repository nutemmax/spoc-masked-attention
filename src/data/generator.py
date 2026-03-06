from __future__ import annotations

import numpy as np


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric square root of a positive semidefinite matrix
    using eigendecomposition.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    sqrt_eigvals = np.sqrt(eigvals_clipped)
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


def generate_gaussian_sequences(
    n_samples: int,
    sigma: np.ndarray,
    d: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Gaussian sequence samples X of shape (n_samples, T, d)
    according to the teacher model:

        X = Sigma^{1/2} G,

    where G has iid standard Gaussian entries.

    Parameters:
    n_samples : int
        Number of samples to generate.
    sigma : np.ndarray
        Covariance matrix of shape (T, T).
    d : int
        Token dimension.
    rng : np.random.Generator | None
        Optional numpy random generator.

    Returns:
    np.ndarray
        Array of shape (n_samples, T, d).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    T = sigma.shape[0]
    if sigma.shape != (T, T):
        raise ValueError("sigma must be square of shape (T, T).")

    sigma_sqrt = matrix_sqrt_psd(sigma)

    g = rng.standard_normal(size=(n_samples, T, d))
    x = np.einsum("ab,nbk->nak", sigma_sqrt, g)

    return x


def compute_n_from_alpha(alpha: float, d: int) -> int:
    """
    Compute n = alpha * d^2 and cast to int
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")

    return int(round(alpha * (d ** 2)))