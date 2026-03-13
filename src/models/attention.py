from __future__ import annotations
import math
import torch
from torch import Tensor, nn


def _ensure_3d(X: Tensor) -> tuple[Tensor, bool]:
    """Convert (T, d) inputs to (1, T, d)."""
    if X.ndim == 2:
        return X.unsqueeze(0), True
    if X.ndim == 3:
        return X, False
    raise ValueError("Input must have shape (T, d) or (batch, T, d).")


def _restore_shape(X: Tensor, squeezed: bool) -> Tensor:
    """Restore original batch shape."""
    return X.squeeze(0) if squeezed else X


def compute_attention_matrix(W: Tensor) -> Tensor:
    """Compute S = (1 / sqrt(rd)) W W^T."""
    if W.ndim != 2:
        raise ValueError("W must have shape (d, r).")
    d, r = W.shape
    return (W @ W.T) / math.sqrt(d * r)


def compute_score_matrix(X: Tensor, S: Tensor, normalize_sqrt_d: bool = True) -> Tensor:
    """Compute H_S(X) for one sample or a batch."""
    X_batch, squeezed = _ensure_3d(X)

    if S.ndim != 2:
        raise ValueError("S must have shape (d, d).")

    batch_size, T, d = X_batch.shape
    if S.shape != (d, d):
        raise ValueError("S must have shape (d, d) matching the last dimension of X.")

    gram = X_batch @ S @ X_batch.transpose(1, 2)
    trace = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
    identity = torch.eye(T, device=X_batch.device, dtype=X_batch.dtype).unsqueeze(0)
    centered = gram - (trace / T).view(batch_size, 1, 1) * identity
    if normalize_sqrt_d:
        scores = centered / math.sqrt(d)
    else:
        scores = centered
    return _restore_shape(scores, squeezed)


def beta_row_softmax(Z: Tensor, beta: float) -> Tensor:
    """Apply row-wise softmax to beta Z."""
    return torch.softmax(beta * Z, dim=-1)


def compute_attention_weights(X: Tensor, W: Tensor, beta: float, normalize_sqrt_d: bool = True) -> Tensor:
    """Compute A_S(X) with tied weights."""
    S = compute_attention_matrix(W)
    H = compute_score_matrix(X, S, normalize_sqrt_d)
    return beta_row_softmax(H, beta)


def predict_with_attention(X_tilde: Tensor, W: Tensor, beta: float, normalize_sqrt_d: bool = True) -> Tensor:
    """Compute A_S(X_tilde) X_tilde."""
    X_batch, squeezed = _ensure_3d(X_tilde)
    A = compute_attention_weights(X_batch, W, beta, normalize_sqrt_d)
    Y_hat = A @ X_batch
    return _restore_shape(Y_hat, squeezed)


class TiedSingleHeadAttention(nn.Module):
    """Single-head tied-attention model with no value projection."""

    def __init__(
        self,
        d: int,
        r: int,
        beta: float,
        normalize_sqrt_d: bool = True,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if d <= 0 or r <= 0:
            raise ValueError("d and r must be positive.")
        if beta <= 0:
            raise ValueError("beta must be positive.")

        self.d = d
        self.r = r
        self.beta = float(beta)
        self.normalize_sqrt_d = normalize_sqrt_d
        # weights initialized with 1/sqrt(r) scaling for better convergence
        W0 = torch.randn(d, r, dtype=dtype, device=device) / math.sqrt(r)
        self.W = nn.Parameter(W0)

    def attention_matrix(self) -> Tensor:
        """Return S = (1 / sqrt(rd)) W W^T."""
        return compute_attention_matrix(self.W)

    def score_matrix(self, X: Tensor) -> Tensor:
        """Return H_S(X)."""
        return compute_score_matrix(X, self.attention_matrix(), self.normalize_sqrt_d)

    def attention_weights(self, X: Tensor) -> Tensor:
        """Return A_S(X)."""
        H = self.score_matrix(X)
        return beta_row_softmax(H, self.beta)

    def forward(self, X_tilde: Tensor) -> Tensor:
        """Return A_S(X_tilde) X_tilde."""
        return predict_with_attention(X_tilde, self.W, self.beta, self.normalize_sqrt_d)

    def masked_predictions(self, X_tilde: Tensor, mask_indices: Tensor) -> Tensor:
        """Return predicted masked tokens of shape (batch, d)."""
        Y_hat = self.forward(X_tilde)
        Y_hat_batch, _ = _ensure_3d(Y_hat)

        if mask_indices.ndim != 1:
            raise ValueError("mask_indices must have shape (batch,).")
        if mask_indices.shape[0] != Y_hat_batch.shape[0]:
            raise ValueError("mask_indices must match the batch size.")
        
        mask_indices = mask_indices.to(device=Y_hat_batch.device, dtype=torch.long)
        batch_indices = torch.arange(Y_hat_batch.shape[0], device=Y_hat_batch.device)
        return Y_hat_batch[batch_indices, mask_indices, :]