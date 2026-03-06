from __future__ import annotations

import numpy as np


def make_mask_embedding(T: int, d: int, mask_value: float = 1.0) -> np.ndarray:
    """
    Create the mask embedding matrix U = 1_T u^T,
    where u = (mask_value, ..., mask_value) in R^d.

    Returns an array of shape (T, d).
    """
    if T <= 0 or d <= 0:
        raise ValueError("T and d must be positive.")

    u = np.full((d,), fill_value=mask_value, dtype=np.float64)
    U = np.ones((T, 1), dtype=np.float64) @ u.reshape(1, d)
    return U


def sample_single_random_mask_indices(
    n_samples: int,
    T: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample one masked token index uniformly for each sample.
    """
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    return rng.integers(low=0, high=T, size=n_samples)


def apply_single_token_mask(
    X: np.ndarray,
    mask_indices: np.ndarray,
    mask_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply single-token masking to a batch of sequences.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, T, d).
    mask_indices : np.ndarray
        Mask positions of shape (n_samples,).
    mask_value : float
        Value used in the mask embedding.

    Returns
    -------
    X_tilde : np.ndarray
        Corrupted input of shape (n_samples, T, d).
    Y_target : np.ndarray
        Target matrix of shape (n_samples, T, d), defined as:
            Y = M_a U + P_a X
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, d = X.shape

    if mask_indices.shape != (n_samples,):
        raise ValueError("mask_indices must have shape (n_samples,).")

    U = make_mask_embedding(T=T, d=d, mask_value=mask_value)

    X_tilde = X.copy()
    Y_target = np.broadcast_to(U, (n_samples, T, d)).copy()

    for i in range(n_samples):
        a = mask_indices[i]
        X_tilde[i, a, :] = U[a, :]
        Y_target[i, a, :] = X[i, a, :]

    return X_tilde, Y_target


def build_masked_dataset(
    X: np.ndarray,
    mask_value: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full masking pipeline:
    - sample one random mask index per sample
    - build corrupted input X_tilde
    - build target Y_target

    Returns:
    X_tilde : np.ndarray
    Y_target : np.ndarray
    mask_indices : np.ndarray
    """
    n_samples, T, _ = X.shape
    mask_indices = sample_single_random_mask_indices(n_samples=n_samples, T=T, rng=rng)
    X_tilde, Y_target = apply_single_token_mask(X=X, mask_indices=mask_indices, mask_value=mask_value)
    return X_tilde, Y_target, mask_indices