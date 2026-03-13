from __future__ import annotations
import numpy as np
from src.data.masking import build_masked_dataset


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric square root of a positive semidefinite matrix
    using eigendecomposition.
    """
    # eigenvalues and eigenvectors of a complex Hermitian/real symmetric matrix
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if np.min(eigvals) < -1e-10:
        raise ValueError("Matrix is not positive semidefinite")
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None) # for numerical stability
    sqrt_eigvals = np.sqrt(eigvals_clipped)

    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


def generate_gaussian_sequences(
    n_samples: int,
    sigma: np.ndarray,
    d: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Gaussian sequence samples X of shape (n_samples, T, d) according to the 
    teacher model: X = Sigma^{1/2} G, where G has iid standard Gaussian entries.

    Parameters:
    n_samples : int
        Number of samples to generate
    sigma : np.ndarray
        Covariance matrix of shape (T, T).
    d : int
        Token dim
    rng : np.random.Generator | None
        Optional numpy random generator

    Returns:
    np.ndarray
        Array of shape (n_samples, T, d)
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")
    if not np.allclose(sigma, sigma.T, atol=1e-12):
        raise ValueError("Sigma must be symmetric.")

    if rng is None:
        rng = np.random.default_rng()

    T = sigma.shape[0]
    if sigma.shape != (T, T):
        raise ValueError("Sigma must be square of shape (T, T).")

    sigma_sqrt = matrix_sqrt_psd(sigma)

    # generate G with shape (n_samples, T, d) and standard normal entries
    g = rng.standard_normal(size=(n_samples, T, d)).astype(np.float64)
    # generate x with shape (n_samples, T, d) according to teacher model: x = sigma^{1/2} g
    x = np.empty((n_samples, T, d), dtype=np.float64)
    for mu in range(n_samples):
        x[mu] = sigma_sqrt @ g[mu]

    return x


def compute_n_from_alpha(alpha: float, d: int) -> int:
    """
    Compute n = alpha * d^2 and cast to int
    """
    if alpha <= 0:
        raise ValueError("Alpha must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")

    return int(round(alpha * (d ** 2)))


def generate_single_mask_dataset(n_samples: int, sigma: np.ndarray, d: int, mask_value: float = 1.0, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate X, X_tilde, Y_target and mask indices for single-token masking."""
    if rng is None:
        rng = np.random.default_rng()

    X = generate_gaussian_sequences(n_samples=n_samples, sigma=sigma, d=d, rng=rng)
    X_tilde, Y_target, mask_indices = build_masked_dataset(X=X, mask_value=mask_value, rng=rng)

    return X, X_tilde, Y_target, mask_indices


def generate_single_mask_dataset_from_alpha(alpha: float, sigma: np.ndarray, d: int, mask_value: float = 1.0, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single-token masked dataset from alpha."""
    n_samples = compute_n_from_alpha(alpha=alpha, d=d)
    return generate_single_mask_dataset(
        n_samples=n_samples,
        sigma=sigma,
        d=d,
        mask_value=mask_value,
        rng=rng,
    )