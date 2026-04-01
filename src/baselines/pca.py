from __future__ import annotations
import numpy as np


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten X of shape (n, T, d) into shape (n, T*d).
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n, T, d).")
    n, T, d = X.shape
    return X.reshape(n, T * d)


def get_masked_block_indices(T: int, d: int, token_idx: int) -> np.ndarray:
    """
    Return the flattened indices corresponding to token 'token_idx'.
    """
    start = token_idx * d
    end = (token_idx + 1) * d
    return np.arange(start, end, dtype=int)


def fit_pca(
    X_train: np.ndarray,
    n_components: int | None = None,
) -> dict[str, np.ndarray | int]:
    """
    Fit PCA on full unmasked training data.
    """
    if X_train.ndim != 3:
        raise ValueError("X_train must have shape (n, T, d).")

    n, T, d = X_train.shape
    Z = flatten_sequences(X_train)  # (n, T*d)

    mean = Z.mean(axis=0)
    Z_centered = Z - mean

    max_rank = min(n - 1, T * d)
    if max_rank <= 0:
        raise ValueError("Need at least 2 training samples to fit PCA.")

    if n_components is None:
        k = max_rank
    else:
        k = int(n_components)
        if k <= 0:
            raise ValueError("n_components must be positive.")
        k = min(k, max_rank)

    # SVD of centered data
    # Z_centered = U S Vt, principal directions are rows of Vt
    _, _, Vt = np.linalg.svd(Z_centered, full_matrices=False)
    components = Vt[:k].T  # shape (T*d, k)

    return {
        "mean": mean,
        "components": components,
        "n_components": k,
        "T": T,
        "d": d,
    }


def reconstruct_masked_sample(
    x: np.ndarray,
    token_idx: int,
    pca_model: dict[str, np.ndarray | int],
    reg: float = 1e-8,
) -> np.ndarray:
    """
    Reconstruct one sample x of shape (T, d) when token 'token_idx' is treated as missing.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (T, d).")

    T = int(pca_model["T"])
    d = int(pca_model["d"])
    mean = np.asarray(pca_model["mean"])
    U = np.asarray(pca_model["components"])  # shape (T*d, k)

    if x.shape != (T, d):
        raise ValueError(f"x must have shape ({T}, {d}).")

    z = x.reshape(T * d)
    masked_idx = get_masked_block_indices(T, d, token_idx)

    observed_mask = np.ones(T * d, dtype=bool)
    observed_mask[masked_idx] = False

    z_obs = z[observed_mask]
    mean_obs = mean[observed_mask]
    U_obs = U[observed_mask, :]  # shape ((T-1)*d, k)

    # Solve for PCA coefficients using only observed coordinates:
    # c = argmin ||(z_obs - mean_obs) - U_obs c||^2 + reg ||c||^2
    A = U_obs.T @ U_obs + reg * np.eye(U_obs.shape[1], dtype=U_obs.dtype)
    b = U_obs.T @ (z_obs - mean_obs)
    coeffs = np.linalg.solve(A, b)

    z_hat = mean + U @ coeffs
    return z_hat.reshape(T, d)


def predict_pca_masked(
    X: np.ndarray,
    mask_indices: np.ndarray,
    pca_model: dict[str, np.ndarray | int],
    reg: float = 1e-8,
) -> np.ndarray:
    """
    Predict only the masked token for each sample.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n, T, d).")
    if mask_indices.ndim != 1:
        raise ValueError("mask_indices must have shape (n,).")
    if X.shape[0] != mask_indices.shape[0]:
        raise ValueError("mask_indices must match batch size.")

    n, T, d = X.shape
    Y_hat_masked = np.zeros((n, d), dtype=X.dtype)

    for i in range(n):
        token_idx = int(mask_indices[i])
        x_hat = reconstruct_masked_sample(X[i], token_idx, pca_model, reg=reg)
        Y_hat_masked[i] = x_hat[token_idx]

    return Y_hat_masked


def evaluate_pca(X: np.ndarray, mask_indices: np.ndarray, pca_model: dict[str, np.ndarray | int], reg: float = 1e-8) -> float:
    if X.ndim != 3:
        raise ValueError("X must have shape (n, T, d).")
    if mask_indices.ndim != 1:
        raise ValueError("mask_indices must have shape (n,).")
    if X.shape[0] != mask_indices.shape[0]:
        raise ValueError("mask_indices must match batch size.")

    n, _, d = X.shape
    Y_hat_masked = predict_pca_masked(X, mask_indices, pca_model, reg=reg)
    true_masked = X[np.arange(n), mask_indices, :]

    sq_errors = np.sum((Y_hat_masked - true_masked) ** 2, axis=1) / d
    return float(np.mean(sq_errors))