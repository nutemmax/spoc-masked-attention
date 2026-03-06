from __future__ import annotations
import numpy as np


def make_mask_embedding(T: int, d: int, mask_value: float = 1.0) -> np.ndarray:
    """Creates and returns the mask embedding matrix U = 1_T u^T as an array of shape (T, d)."""
    if T <= 0 or d <= 0:
        raise ValueError("T and d must be positive.")
    return np.full((T, d), fill_value=mask_value, dtype=np.float64)


def sample_single_random_mask_indices(
    n_samples: int,
    T: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
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
    Apply single-token masking to a batch of sequences

    Returns:
    - X_tilde of shape (n_samples, T, d)
    - Y_target of shape (n_samples, T, d)
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
        a = int(mask_indices[i])
        X_tilde[i, a, :] = U[a, :]
        Y_target[i, a, :] = X[i, a, :]

    return X_tilde, Y_target


def build_masked_dataset(
    X: np.ndarray,
    mask_value: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full single-token masking pipeline for a batch of sequences X."""
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, _ = X.shape
    mask_indices = sample_single_random_mask_indices(n_samples=n_samples, T=T, rng=rng)
    X_tilde, Y_target = apply_single_token_mask(X=X, mask_indices=mask_indices, mask_value=mask_value)

    return X_tilde, Y_target, mask_indices


# ===== MULTI-TOKEN MASKING =====
def resolve_number_of_masked_tokens(
    T: int,
    n_masked_tokens: int | None = None,
    mask_fraction: float | None = None,
) -> int:
    """Resolve the number of masked tokens from either an integer or a fraction"""
    if T <= 0:
        raise ValueError("T must be positive.")

    if (n_masked_tokens is None) == (mask_fraction is None):
        raise ValueError("Provide exactly one of n_masked_tokens or mask_fraction.")

    if n_masked_tokens is not None:
        if not (1 <= n_masked_tokens <= T):
            raise ValueError("n_masked_tokens must satisfy 1 <= n_masked_tokens <= T.")
        return int(n_masked_tokens)
    
    assert mask_fraction is not None
    if not (0.0 < mask_fraction <= 1.0):
        raise ValueError("mask_fraction must satisfy 0 < mask_fraction <= 1.")

    m = int(round(mask_fraction * T))
    m = max(1, m)
    m = min(T, m)

    return m


def sample_multi_random_mask_matrix(
    n_samples: int,
    T: int,
    n_masked_tokens: int | None = None,
    mask_fraction: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample a boolean mask matrix of shape (n_samples, T);
    Each row contains exactly m masked positions sampled uniformly without replacement
    """
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    if rng is None:
        rng = np.random.default_rng()

    m = resolve_number_of_masked_tokens(
        T=T,
        n_masked_tokens=n_masked_tokens,
        mask_fraction=mask_fraction,
    )

    mask_matrix = np.zeros((n_samples, T), dtype=bool)

    for i in range(n_samples):
        masked_positions = rng.choice(T, size=m, replace=False)
        mask_matrix[i, masked_positions] = True

    return mask_matrix


def apply_multi_token_mask(
    X: np.ndarray,
    mask_matrix: np.ndarray,
    mask_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply multi-token masking to a batch of sequences

    Returns:
    - X_tilde of shape (n_samples, T, d)
    - Y_target of shape (n_samples, T, d)

    Masked rows are replaced by the mask embedding in X_tilde
    Unmasked rows are replaced by the mask embedding in Y_target
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, d = X.shape

    if mask_matrix.shape != (n_samples, T):
        raise ValueError("mask_matrix must have shape (n_samples, T).")

    U = make_mask_embedding(T=T, d=d, mask_value=mask_value)

    X_tilde = X.copy()
    Y_target = np.broadcast_to(U, (n_samples, T, d)).copy()

    for i in range(n_samples):
        masked_rows = mask_matrix[i]
        X_tilde[i, masked_rows, :] = U[masked_rows, :]
        Y_target[i, masked_rows, :] = X[i, masked_rows, :]

    return X_tilde, Y_target


def build_multi_masked_dataset(
    X: np.ndarray,
    n_masked_tokens: int | None = None,
    mask_fraction: float | None = None,
    mask_value: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full multi-token masking pipeline

    Returns:
    - X_tilde of shape (n_samples, T, d)
    - Y_target of shape (n_samples, T, d)
    - mask_matrix of shape (n_samples, T)
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, _ = X.shape

    mask_matrix = sample_multi_random_mask_matrix(
        n_samples=n_samples,
        T=T,
        n_masked_tokens=n_masked_tokens,
        mask_fraction=mask_fraction,
        rng=rng,
    )

    X_tilde, Y_target = apply_multi_token_mask(
        X=X,
        mask_matrix=mask_matrix,
        mask_value=mask_value,
    )

    return X_tilde, Y_target, mask_matrix