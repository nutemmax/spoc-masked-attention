from __future__ import annotations
import numpy as np


def get_observed_indices(T: int, masked_index: int) -> list[int]:
    """Return the list of token indices excluding the masked index"""
    if not (0 <= masked_index < T):
        raise ValueError("masked_index out of range.")
    return [i for i in range(T) if i != masked_index]


def bayes_regression_coefficients(sigma: np.ndarray, masked_index: int) -> np.ndarray:
    """Compute the Bayes-optimal regression coefficients for predicting x_a from x_{-a}
    Returns an array of shape (T - 1,)"""
    
    if sigma.ndim != 2:
        raise ValueError("sigma must be a 2d array.")

    T = sigma.shape[0]
    if sigma.shape != (T, T):
        raise ValueError("Sigma must be square.")
    if not np.allclose(sigma, sigma.T, atol=1e-12):
        raise ValueError("Sigma must be symmetric.")

    observed = get_observed_indices(T, masked_index)
    sigma_a_obs = sigma[masked_index, observed]
    sigma_obs_obs = sigma[np.ix_(observed, observed)]

    coeffs = np.linalg.solve(sigma_obs_obs.T, sigma_a_obs.T).T
    return coeffs


def predict_bayes_single_sample(
    X_sample: np.ndarray,
    sigma: np.ndarray,
    masked_index: int,
) -> np.ndarray:
    """
    Predicts the masked token for a single sample using the Bayes-optimal Gaussian predictor
    """
    if X_sample.ndim != 2:
        raise ValueError("X_sample must have shape (T, d).")

    T, _ = X_sample.shape
    if sigma.shape != (T, T):
        raise ValueError("sigma must have shape (T, T) matching X_sample.")

    observed = get_observed_indices(T, masked_index)
    coeffs = bayes_regression_coefficients(sigma=sigma, masked_index=masked_index)

    x_obs = X_sample[observed, :]
    x_hat = coeffs @ x_obs

    return x_hat


def predict_bayes_batch(
    X: np.ndarray,
    sigma: np.ndarray,
    mask_indices: np.ndarray,
) -> np.ndarray:
    """
    Predicts the masked token for each sample in a batch.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, T, d = X.shape

    if sigma.shape != (T, T):
        raise ValueError("sigma must have shape (T, T) matching X.")
    if mask_indices.shape != (n_samples,):
        raise ValueError("mask_indices must have shape (n_samples,).")

    predictions = np.zeros((n_samples, d), dtype=np.float64)

    for i in range(n_samples):
        predictions[i] = predict_bayes_single_sample(
            X_sample=X[i],
            sigma=sigma,
            masked_index=int(mask_indices[i]),
        )

    return predictions


def bayes_conditional_variance(sigma: np.ndarray, masked_index: int) -> float:
    """
    Computes the scalar conditional variance sigma_{a|-a}^2
    """
    if sigma.ndim != 2:
        raise ValueError("sigma must be a 2d array.")

    T = sigma.shape[0]
    if sigma.shape != (T, T):
        raise ValueError("sigma must be square.")
    if not np.allclose(sigma, sigma.T, atol=1e-12):
        raise ValueError("sigma must be symmetric.")

    observed = get_observed_indices(T, masked_index)

    sigma_aa = sigma[masked_index, masked_index]
    sigma_a_obs = sigma[masked_index, observed]
    sigma_obs_a = sigma[observed, masked_index]
    sigma_obs_obs = sigma[np.ix_(observed, observed)]

    cond_var = sigma_aa - sigma_a_obs @ np.linalg.solve(sigma_obs_obs, sigma_obs_a)
    return float(cond_var)


def bayes_population_risk_uniform_mask(sigma: np.ndarray) -> float:
    """
    Computes the Bayes population risk averaged uniformly over mask positions
    This is the true optimal risk achievable risk under the teacher model; does not depend on the dataset X
    """
    T = sigma.shape[0]
    risks = [bayes_conditional_variance(sigma, a) for a in range(T)]
    return float(np.mean(risks))


def masked_mse_per_coordinate(
    X: np.ndarray,
    predictions: np.ndarray,
    mask_indices: np.ndarray,
) -> float:
    """
    Computes the empirical masked reconstruction error (MSE) per coordinate
    Inputs : 
        X : (n_samples, T, d) array of input samples
        predictions : (n_samples, d) array of predicted masked tokens
        mask_indices : (n_samples, ) array of masked token indices per each sample

    Returns : 
        MSE : scalar value of the MSE per coordinate, avg over the dataset and over d coordinates
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, _, d = X.shape
    if predictions.shape != (n_samples, d):
        raise ValueError("predictions must have shape (n_samples, d).")
    if mask_indices.shape != (n_samples,):
        raise ValueError("mask_indices must have shape (n_samples,).")

    true_masked_tokens = X[np.arange(n_samples), mask_indices, :]
    mse = np.mean(np.sum((true_masked_tokens - predictions) ** 2, axis=1) / d)

    return float(mse)


def bayes_population_risk_empirical(X: np.ndarray, sigma : np.ndarray, mask_indices: np.ndarray) -> float:
    """
    Estimates the Bayes population risk by averaging the masked MSE per coordinate over the dataset X using the Bayes predictor's predictions.
    It should converge to the true Bayes population risk as n_samples -> inf, but can be computed for any finite X.
    """
    predictions = predict_bayes_batch(X=X, sigma=sigma, mask_indices=mask_indices)
    risk = masked_mse_per_coordinate(X=X, predictions=predictions, mask_indices=mask_indices)
    return risk