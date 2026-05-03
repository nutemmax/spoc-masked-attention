from __future__ import annotations

import numpy as np
import torch


# ========= SINGLE-TOKEN MASKING ==========

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

    return rng.integers(low=0, high=T, size=n_samples, dtype=np.int64)


def sample_single_last_mask_indices(
    n_samples: int,
    T: int,
) -> np.ndarray:
    """Always mask the last token."""
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    return np.full(shape=(n_samples,), fill_value=T - 1, dtype=np.int64)


def sample_all_mask_indices(
    n_samples: int,
    T: int,
) -> np.ndarray:
    """Return all possible single-token mask indices for each sample."""
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    return np.tile(np.arange(T, dtype=np.int64), reps=n_samples)


def sample_k_random_mask_indices(
    n_samples: int,
    T: int,
    masks_per_sample: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample k distinct single-token mask indices for each sample."""
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    masks_per_sample = int(masks_per_sample)
    if masks_per_sample <= 0:
        raise ValueError("masks_per_sample must be positive.")
    if masks_per_sample > T:
        raise ValueError("masks_per_sample cannot exceed T.")

    if rng is None:
        rng = np.random.default_rng()

    mask_indices = np.empty(n_samples * masks_per_sample, dtype=np.int64)

    for i in range(n_samples):
        chosen = rng.choice(T, size=masks_per_sample, replace=False)
        start = i * masks_per_sample
        end = start + masks_per_sample
        mask_indices[start:end] = chosen

    return mask_indices


def repeat_samples_for_masking(
    X: np.ndarray,
    repeats_per_sample: int,
) -> np.ndarray:
    """Repeat each sample a fixed number of times along the batch dimension."""
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")
    if repeats_per_sample <= 0:
        raise ValueError("repeats_per_sample must be positive.")

    return np.repeat(X, repeats=repeats_per_sample, axis=0)


def apply_single_token_mask(
    X: np.ndarray,
    mask_indices: np.ndarray,
    mask_value: float = 1.0,
    return_targets: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply single-token masking to a batch of sequences.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, _, _ = X.shape

    if mask_indices.shape != (n_samples,):
        raise ValueError("mask_indices must have shape (n_samples,).")

    X_tilde = X.copy()
    rows = np.arange(n_samples)
    X_tilde[rows, mask_indices, :] = mask_value

    if not return_targets:
        return X_tilde, mask_indices

    Y_target = np.full_like(X, fill_value=mask_value)
    Y_target[rows, mask_indices, :] = X[rows, mask_indices, :]

    return X_tilde, Y_target, mask_indices


def build_masked_dataset(
    X: np.ndarray,
    mask_value: float = 1.0,
    rng: np.random.Generator | None = None,
    masking_strategy: str = "random",
    return_targets: bool = False,
    masks_per_sample: int = 1,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full single-token masking pipeline for a batch of sequences X.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, _ = X.shape

    if masking_strategy == "random":
        X_masked_source = X
        mask_indices = sample_single_random_mask_indices(
            n_samples=n_samples,
            T=T,
            rng=rng,
        )
    elif masking_strategy == "last":
        X_masked_source = X
        mask_indices = sample_single_last_mask_indices(
            n_samples=n_samples,
            T=T,
        )
    elif masking_strategy == "all":
        X_masked_source = repeat_samples_for_masking(
            X=X,
            repeats_per_sample=T,
        )
        mask_indices = sample_all_mask_indices(
            n_samples=n_samples,
            T=T,
        )
    elif masking_strategy == "k_random":
        masks_per_sample = int(masks_per_sample)
        X_masked_source = repeat_samples_for_masking(
            X=X,
            repeats_per_sample=masks_per_sample,
        )
        mask_indices = sample_k_random_mask_indices(
            n_samples=n_samples,
            T=T,
            masks_per_sample=masks_per_sample,
            rng=rng,
        )
    else:
        raise ValueError(
            f"Unknown masking_strategy='{masking_strategy}'. Use 'random', 'last', 'all', or 'k_random'."
        )

    return apply_single_token_mask(
        X=X_masked_source,
        mask_indices=mask_indices,
        mask_value=mask_value,
        return_targets=return_targets,
    )


# ========= TORCH SINGLE-TOKEN MASKING ==========

def sample_single_random_mask_indices_torch(
    n_samples: int,
    T: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Torch version of random single-token mask sampling."""
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    return torch.randint(
        low=0,
        high=T,
        size=(n_samples,),
        device=device,
        dtype=torch.long,
    )


def sample_single_last_mask_indices_torch(
    n_samples: int,
    T: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Torch version that always masks the last token."""
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    return torch.full(
        size=(n_samples,),
        fill_value=T - 1,
        device=device,
        dtype=torch.long,
    )


def sample_all_mask_indices_torch(
    n_samples: int,
    T: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Torch version that returns all possible single-token mask indices for each sample."""
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    return torch.arange(T, device=device, dtype=torch.long).repeat(n_samples)


def sample_k_random_mask_indices_torch(
    n_samples: int,
    T: int,
    masks_per_sample: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Torch version that samples k distinct single-token mask indices for each sample."""
    if n_samples <= 0 or T <= 0:
        raise ValueError("n_samples and T must be positive.")

    masks_per_sample = int(masks_per_sample)
    if masks_per_sample <= 0:
        raise ValueError("masks_per_sample must be positive.")
    if masks_per_sample > T:
        raise ValueError("masks_per_sample cannot exceed T.")

    random_scores = torch.rand(n_samples, T, device=device)
    chosen_masks = torch.argsort(random_scores, dim=1)[:, :masks_per_sample]

    return chosen_masks.reshape(-1).to(dtype=torch.long)


def repeat_samples_for_masking_torch(
    X: torch.Tensor,
    repeats_per_sample: int,
) -> torch.Tensor:
    """Repeat each sample a fixed number of times along the batch dimension."""
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")
    if repeats_per_sample <= 0:
        raise ValueError("repeats_per_sample must be positive.")

    return X.repeat_interleave(repeats_per_sample, dim=0)


def apply_single_token_mask_torch(
    X: torch.Tensor,
    mask_indices: torch.Tensor,
    mask_value: float = 1.0,
    return_targets: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch version of single-token masking.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, _, _ = X.shape

    if mask_indices.shape != (n_samples,):
        raise ValueError("mask_indices must have shape (n_samples,).")

    X_tilde = X.clone()
    rows = torch.arange(n_samples, device=X.device)
    X_tilde[rows, mask_indices, :] = mask_value

    if not return_targets:
        return X_tilde, mask_indices

    Y_target = torch.full_like(X, fill_value=mask_value)
    Y_target[rows, mask_indices, :] = X[rows, mask_indices, :]

    return X_tilde, Y_target, mask_indices

def build_multi_random_masked_dataset_torch(
    X: torch.Tensor,
    mask_value: float = 1.0,
    masks_per_sample: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build repeated single-token loss terms from multi-token corrupted inputs."""
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, _ = X.shape

    masks_per_sample = int(masks_per_sample)
    if masks_per_sample <= 0:
        raise ValueError("masks_per_sample must be positive.")
    if masks_per_sample > T:
        raise ValueError("masks_per_sample cannot exceed T.")

    random_scores = torch.rand(n_samples, T, device=X.device)
    chosen_masks = torch.argsort(random_scores, dim=1)[:, :masks_per_sample]

    X_tilde_base = X.clone()
    sample_indices = torch.arange(n_samples, device=X.device).unsqueeze(1)
    X_tilde_base[sample_indices, chosen_masks, :] = float(mask_value)

    X_tilde = X_tilde_base.repeat_interleave(masks_per_sample, dim=0)
    mask_indices = chosen_masks.reshape(-1).to(dtype=torch.long)

    return X_tilde, mask_indices


def build_masked_dataset_torch(
    X: torch.Tensor,
    mask_value: float = 1.0,
    masking_strategy: str = "random",
    return_targets: bool = False,
    masks_per_sample: int = 1,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, _ = X.shape

    if masking_strategy == "random":
        X_masked_source = X
        mask_indices = sample_single_random_mask_indices_torch(
            n_samples=n_samples,
            T=T,
            device=X.device,
        )
    elif masking_strategy == "last":
        X_masked_source = X
        mask_indices = sample_single_last_mask_indices_torch(
            n_samples=n_samples,
            T=T,
            device=X.device,
        )
    elif masking_strategy == "all":
        X_masked_source = repeat_samples_for_masking_torch(
            X=X,
            repeats_per_sample=T,
        )
        mask_indices = sample_all_mask_indices_torch(
            n_samples=n_samples,
            T=T,
            device=X.device,
        )
    elif masking_strategy == "k_random":
        masks_per_sample = int(masks_per_sample)
        X_masked_source = repeat_samples_for_masking_torch(
            X=X,
            repeats_per_sample=masks_per_sample,
        )
        mask_indices = sample_k_random_mask_indices_torch(
            n_samples=n_samples,
            T=T,
            masks_per_sample=masks_per_sample,
            device=X.device,
        )
    elif masking_strategy == "multi_random":
        return build_multi_random_masked_dataset_torch(
            X=X,
            mask_value=mask_value,
            masks_per_sample=masks_per_sample,
        )
    else:
        raise ValueError(
            f"Unknown masking_strategy='{masking_strategy}'. Use 'random', 'last', 'all', 'k_random', or 'multi_random'."
        )

    return apply_single_token_mask_torch(
        X=X_masked_source,
        mask_indices=mask_indices,
        mask_value=mask_value,
        return_targets=return_targets,
    )


# ========= MULTI-TOKEN MASKING ==========

def resolve_number_of_masked_tokens(
    T: int,
    n_masked_tokens: int | None = None,
    mask_fraction: float | None = None,
) -> int:
    """Resolve the number of masked tokens from either an integer or a fraction."""
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
    Sample a boolean mask matrix of shape (n_samples, T).
    Each row contains exactly m masked positions sampled uniformly without replacement.
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
    Apply multi-token masking to a batch of sequences.
    Returns:
    - X_tilde of shape (n_samples, T, d)
    - Y_target of shape (n_samples, T, d)
    Masked rows are replaced by the mask embedding in X_tilde.
    Unmasked rows are replaced by the mask embedding in Y_target.
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
    Full multi-token masking pipeline.
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