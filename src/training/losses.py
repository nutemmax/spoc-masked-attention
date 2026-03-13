from __future__ import annotations
import torch
from torch import Tensor


def masked_mse_loss(Y_hat: Tensor, X: Tensor, mask_indices: Tensor) -> Tensor:
    """Masked reconstruction MSE per coordinate."""
    if Y_hat.ndim != 3 or X.ndim != 3:
        raise ValueError("Y_hat and X must have shape (batch, T, d).")
    if Y_hat.shape != X.shape:
        raise ValueError("Y_hat and X must have the same shape.")
    if mask_indices.ndim != 1:
        raise ValueError("mask_indices must have shape (batch,).")
    if mask_indices.shape[0] != X.shape[0]:
        raise ValueError("mask_indices must match batch size.")

    batch_size, _, d = X.shape
    mask_indices = mask_indices.to(device=X.device, dtype=torch.long)
    batch_indices = torch.arange(batch_size, device=X.device)

    pred_masked = Y_hat[batch_indices, mask_indices, :]
    true_masked = X[batch_indices, mask_indices, :]

    return torch.mean(torch.sum((pred_masked - true_masked) ** 2, dim=1) / d)


def reconstruction_loss(model, X_tilde: Tensor, X: Tensor, mask_indices: Tensor) -> Tensor:
    """Batch reconstruction loss."""
    Y_hat = model(X_tilde)
    return masked_mse_loss(Y_hat, X, mask_indices)


def regularized_training_objective(model, X_tilde: Tensor, X: Tensor, mask_indices: Tensor, lambda_reg: float) -> Tensor:
    """Reconstruction loss + L2 regularization."""
    recon = reconstruction_loss(model, X_tilde, X, mask_indices)
    reg = lambda_reg * torch.sum(model.W ** 2)
    return recon + reg