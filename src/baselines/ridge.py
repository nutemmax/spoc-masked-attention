from __future__ import annotations
import numpy as np


def _fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    X: shape (n_samples, p)
    y: shape (n_samples,)
    returns w: shape (p,)
    """
    p = X.shape[1]
    A = X.T @ X + lambda_reg * np.eye(p, dtype=X.dtype)
    b = X.T @ y
    return np.linalg.solve(A, b)


def fit_ridge_per_feature(
    X_train: np.ndarray,
    lambda_reg: float,
) -> dict[int, np.ndarray]:
    """
    Fit ridge baseline for token prediction.

    X_train: shape (n, T, d)

    Returns:
        weights_by_token[a]: shape (d, T-1)
        For each token a and feature k, weights_by_token[a][k] predicts X[:, a, k]
        from X[:, other_tokens, k].
    """
    n, T, d = X_train.shape
    weights_by_token: dict[int, np.ndarray] = {}

    for a in range(T):
        other_tokens = [b for b in range(T) if b != a]
        W_a = np.zeros((d, T - 1), dtype=X_train.dtype)

        for k in range(d):
            X_feat = X_train[:, other_tokens, k]   # shape (n, T-1)
            y_feat = X_train[:, a, k]              # shape (n,)
            w = _fit_ridge_closed_form(X_feat, y_feat, lambda_reg=lambda_reg)
            W_a[k] = w

        weights_by_token[a] = W_a

    return weights_by_token


def predict_ridge(
    X: np.ndarray,
    weights_by_token: dict[int, np.ndarray],
    mask_indices: np.ndarray,
) -> np.ndarray:
    """
    Predict only the masked token for each sample.

    X: shape (n, T, d)
    mask_indices: shape (n,)
    returns Y_hat_masked: shape (n, d)
    """
    n, T, d = X.shape
    Y_hat_masked = np.zeros((n, d), dtype=X.dtype)

    for i in range(n):
        a = int(mask_indices[i])
        other_tokens = [b for b in range(T) if b != a]
        W_a = weights_by_token[a]  # shape (d, T-1)

        for k in range(d):
            x_in = X[i, other_tokens, k]          # shape (T-1,)
            Y_hat_masked[i, k] = np.dot(W_a[k], x_in)

    return Y_hat_masked


def evaluate_ridge(
    X: np.ndarray,
    mask_indices: np.ndarray,
    weights_by_token: dict[int, np.ndarray],
) -> float:
    """
    Mean masked-token reconstruction loss:
    average over samples of ||pred - true||^2 / d
    """
    n, _, d = X.shape
    Y_hat_masked = predict_ridge(X, weights_by_token, mask_indices)

    true_masked = X[np.arange(n), mask_indices, :]
    sq_errors = np.sum((Y_hat_masked - true_masked) ** 2, axis=1) / d
    return float(np.mean(sq_errors))