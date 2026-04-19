from __future__ import annotations

import math
import torch
from torch import Tensor
from typing import cast

from src.data.masking import build_masked_dataset_torch
from src.models.attention import (
    beta_row_softmax,
    compute_attention_matrix,
    compute_score_matrix,
)


def resolve_teacher_rank(r_star: int | None, d: int) -> int:
    if d <= 0:
        raise ValueError("d must be positive.")

    if r_star is None:
        return d

    r_star = int(r_star)
    if r_star <= 0:
        raise ValueError("r_star must be positive.")

    return r_star


def generate_teacher_weights_torch(
    d: int,
    r_star: int | None = None,
    teacher_init: str = "standard_gaussian",
    sigma_star: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate teacher weights W_star and matrix S_star."""
    if d <= 0:
        raise ValueError("d must be positive.")
    if sigma_star <= 0:
        raise ValueError("sigma_star must be positive.")

    r_star_resolved = resolve_teacher_rank(r_star, d)

    if teacher_init == "standard_gaussian":
        scale = 1.0
    elif teacher_init == "scaled_gaussian":
        scale = float(sigma_star)
    else:
        raise ValueError(
            "Unknown teacher_init. Use 'standard_gaussian' or 'scaled_gaussian'."
        )

    W_star = scale * torch.randn(
        d,
        r_star_resolved,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    S_star = compute_attention_matrix(W_star)

    return W_star, S_star


def compute_teacher_attention_torch(
    G: Tensor,
    S_star: Tensor,
    beta_star: float,
    normalize_sqrt_d: bool = True,
) -> Tensor:
    """Compute A_star(G)."""
    if G.ndim != 3:
        raise ValueError("G must have shape (n_samples, T, d).")
    if S_star.ndim != 2:
        raise ValueError("S_star must have shape (d, d).")
    if beta_star <= 0:
        raise ValueError("beta_star must be positive.")

    _, _, d = G.shape
    if S_star.shape != (d, d):
        raise ValueError("S_star must have shape (d, d) matching G.")

    H_star = compute_score_matrix(
        X=G,
        S=S_star,
        normalize_sqrt_d=normalize_sqrt_d,
    )
    return beta_row_softmax(H_star, beta=float(beta_star))


def generate_teacher_attention_sequences_torch(
    n_samples: int,
    T: int,
    d: int,
    S_star: Tensor,
    beta_star: float,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    normalize_sqrt_d: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Generate G, A_star(G), and X = A_star(G)G."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if T <= 0 or d <= 0:
        raise ValueError("T and d must be positive.")

    S_star = S_star.to(dtype=dtype, device=device)

    if S_star.shape != (d, d):
        raise ValueError("S_star must have shape (d, d).")

    G = torch.randn(
        n_samples,
        T,
        d,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    A_star = compute_teacher_attention_torch(
        G=G,
        S_star=S_star,
        beta_star=beta_star,
        normalize_sqrt_d=normalize_sqrt_d,
    )
    X = A_star @ G

    return {
        "G": G,
        "A_star": A_star,
        "X": X,
    }


def generate_single_mask_teacher_attention_dataset_torch(
    n_samples: int,
    T: int,
    d: int,
    S_star: Tensor,
    beta_star: float,
    mask_value: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    masking_strategy: str = "random",
    normalize_sqrt_d: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Generate teacher-attention data with single-token masking."""
    data = generate_teacher_attention_sequences_torch(
        n_samples=n_samples,
        T=T,
        d=d,
        S_star=S_star,
        beta_star=beta_star,
        dtype=dtype,
        device=device,
        normalize_sqrt_d=normalize_sqrt_d,
        generator=generator,
    )

    masked_outputs = build_masked_dataset_torch(
    X=data["X"],
    mask_value=mask_value,
    masking_strategy=masking_strategy,
    return_targets=False,
)

    X_tilde = masked_outputs[0]
    mask_indices = masked_outputs[1]

    data["X_tilde"] = X_tilde
    data["mask_indices"] = mask_indices

    return data