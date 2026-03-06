from __future__ import annotations

import numpy as np

from src.data.covariance import build_covariance, is_positive_definite
from src.data.generator import generate_gaussian_sequences
from src.data.masking import build_masked_dataset
from src.baselines.bayes import (
    predict_bayes_batch,
    masked_mse_per_coordinate,
    bayes_population_risk_uniform_mask,
)


def main() -> None:
    rng = np.random.default_rng(42)

    T = 4
    d = 64
    n_samples = 2000
    rho = 0.5

    sigma = build_covariance(covariance_type="tridiagonal", T=T, rho=rho)

    print("Sigma:")
    print(sigma)
    print(f"Positive definite: {is_positive_definite(sigma)}")

    X = generate_gaussian_sequences(n_samples=n_samples, sigma=sigma, d=d, rng=rng)
    _, _, mask_indices = build_masked_dataset(X, mask_value=1.0, rng=rng)

    predictions = predict_bayes_batch(X=X, sigma=sigma, mask_indices=mask_indices)
    empirical_mse = masked_mse_per_coordinate(X=X, predictions=predictions, mask_indices=mask_indices)
    theoretical_risk = bayes_population_risk_uniform_mask(sigma=sigma)

    print(f"Empirical Bayes masked MSE: {empirical_mse:.6f}")
    print(f"Theoretical Bayes population risk: {theoretical_risk:.6f}")


if __name__ == "__main__":
    main()