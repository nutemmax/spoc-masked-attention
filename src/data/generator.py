from __future__ import annotations
from typing import overload, Literal

import numpy as np
import torch

from src.data.masking import build_masked_dataset, build_masked_dataset_torch


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    # compute the symmetric sqrt of a PSD matrix using eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if np.min(eigvals) < -1e-10:
        raise ValueError("Matrix is not positive semidefinite")

    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    sqrt_eigvals = np.sqrt(eigvals_clipped)

    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


def matrix_sqrt_psd_torch(matrix: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    if torch.min(eigvals).item() < -1e-10:
        raise ValueError("Matrix is not positive semidefinite")

    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    sqrt_eigvals = torch.sqrt(eigvals_clipped)

    return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T


def generate_gaussian_sequences(
    n_samples: int,
    sigma: np.ndarray,
    d: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate Gaussian sequence samples X of shape (n_samples, T, d) according to the
    teacher model: X = Sigma^{1/2} G, where G has iid standard Gaussian entries.
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

    g = rng.standard_normal(size=(n_samples, T, d)).astype(np.float64)
    x = np.einsum("ij,njd->nid", sigma_sqrt, g)

    return x


def generate_gaussian_sequences_torch(
    n_samples: int,
    sigma: torch.Tensor,
    d: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Sigma must be square of shape (T, T).")

    sigma = sigma.to(dtype=dtype, device=device)
    T = sigma.shape[0]
    sigma_sqrt = matrix_sqrt_psd_torch(sigma)

    g = torch.randn(n_samples, T, d, dtype=dtype, device=device)
    x = torch.einsum("ij,njd->nid", sigma_sqrt, g)

    return x


def compute_n_from_alpha(alpha: float, d: int) -> int:
    if alpha <= 0:
        raise ValueError("Alpha must be positive.")
    if d <= 0:
        raise ValueError("d must be positive.")

    return int(round(alpha * (d ** 2)))


def generate_single_mask_dataset(
    n_samples: int,
    sigma: np.ndarray,
    d: int,
    mask_value: float = 1.0,
    rng: np.random.Generator | None = None,
    masking_strategy: str = "random",
    return_targets: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate X, X_tilde, and mask indices for single-token masking.
    If return_targets=False:
        returns (X, X_tilde, mask_indices)
    If return_targets=True:
        returns (X, X_tilde, Y_target, mask_indices)
    """
    if rng is None:
        rng = np.random.default_rng()

    X = generate_gaussian_sequences(
        n_samples=n_samples,
        sigma=sigma,
        d=d,
        rng=rng,
    )

    masked_outputs = build_masked_dataset(
        X=X,
        mask_value=mask_value,
        rng=rng,
        masking_strategy=masking_strategy,
        return_targets=return_targets,
    )

    if return_targets:
        X_tilde, Y_target, mask_indices = masked_outputs
        return X, X_tilde, Y_target, mask_indices

    X_tilde, mask_indices = masked_outputs
    return X, X_tilde, mask_indices


def generate_single_mask_dataset_torch(
    n_samples: int,
    sigma: torch.Tensor,
    d: int,
    mask_value: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    masking_strategy: str = "random",
    return_targets: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch version
    X = generate_gaussian_sequences_torch(
        n_samples=n_samples,
        sigma=sigma,
        d=d,
        dtype=dtype,
        device=device,
    )

    masked_outputs = build_masked_dataset_torch(
        X=X,
        mask_value=mask_value,
        masking_strategy=masking_strategy,
        return_targets=return_targets,
    )

    if return_targets:
        X_tilde, Y_target, mask_indices = masked_outputs
        return X, X_tilde, Y_target, mask_indices

    X_tilde, mask_indices = masked_outputs
    return X, X_tilde, mask_indices


def generate_single_mask_dataset_from_alpha(
    alpha: float,
    sigma: np.ndarray,
    d: int,
    mask_value: float = 1.0,
    rng: np.random.Generator | None = None,
    masking_strategy: str = "random",
    return_targets: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = compute_n_from_alpha(alpha=alpha, d=d)
    return generate_single_mask_dataset(
        n_samples=n_samples,
        sigma=sigma,
        d=d,
        mask_value=mask_value,
        rng=rng,
        masking_strategy=masking_strategy,
        return_targets=return_targets,
    )


def generate_single_mask_dataset_from_alpha_torch(
    alpha: float,
    sigma: torch.Tensor,
    d: int,
    mask_value: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    masking_strategy: str = "random",
    return_targets: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_samples = compute_n_from_alpha(alpha=alpha, d=d)
    return generate_single_mask_dataset_torch(
        n_samples=n_samples,
        sigma=sigma,
        d=d,
        mask_value=mask_value,
        dtype=dtype,
        device=device,
        masking_strategy=masking_strategy,
        return_targets=return_targets,
    )