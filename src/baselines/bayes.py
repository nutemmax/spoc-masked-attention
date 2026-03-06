from __future__ import annotations

import numpy as np


def get_observed_indices(T: int, masked_index: int) -> list[int]:
    """
    Return the list of token indices excluding the masked index.
    """
    if not (0 <= masked_index < T):
        raise ValueError("masked_index out of range.")
    return [i for i in range(T) if i != masked_index]


def bayes_regression_coefficients(sigma: np.ndarray, masked_index: int) -> np.ndarray:
    """
    Compute the Bayes-optimal regression coefficients for predicting x_a
    from x_{-a}:
    
        c_a = Sigma_{a,-a} Sigma_{-a,-a}^{-1}

    Returns:
    np.ndarray
        Array of shape (T - 1,).
    """
    T = sigma.shape[0]
    observed = get_observed_indices(T, masked_index)

    sigma_a_obs = sigma[masked_index, observed]              # shape (T-1,)
    sigma_obs_obs = sigma[np.ix_(observed, observed)]        # shape (T-1, T-1)

    coeffs = sigma_a_obs @ np.linalg.inv(sigma_obs_obs)
    return coeffs


def predict_bayes_single_sample(
    X_sample: np.ndarray,
    sigma: np.ndarray,
    masked_index: int,
) -> np.ndarray:
    """
    Predict the masked token for a single sample using the Bayes-optimal
    Gaussian predictor.

    Parameters:
    X_sample : np.ndarray
        Array of shape (T, d).
    sigma : np.ndarray
        Covariance matrix of shape (T, T).
    masked_index : int
        Index of the masked token.

    Returns:
    np.ndarray
        Predicted token of shape (d,).
    """
    if X_sample.ndim != 2:
        raise ValueError("X_sample must have shape (T, d).")

    T, _ = X_sample.shape
    observed = get_observed_indices(T, masked_index)
    coeffs = bayes_regression_coefficients(sigma=sigma, masked_index=masked_index)

    x_obs = X_sample[observed, :]   # shape (T-1, d)
    x_hat = coeffs @ x_obs          # shape (d,)

    return x_hat


def predict_bayes_batch(
    X: np.ndarray,
    sigma: np.ndarray,
    mask_indices: np.ndarray,
) -> np.ndarray:
    """
    Predict the masked token for each sample in a batch.

    Parameters : 
    X : np.ndarray
        Array of shape (n_samples, T, d).
    sigma : np.ndarray
        Covariance matrix of shape (T, T).
    mask_indices : np.ndarray
        Mask positions of shape (n_samples,).

    Returns : np.ndarray
        Predictions of shape (n_samples, d).
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (n_samples, T, d).")

    n_samples, _, d = X.shape

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
    Compute the scalar conditional variance of the masked token given the observed tokens:
    sigma_{a|-a}^2 = Sigma_aa - Sigma_{a,-a} Sigma_{-a,-a}^{-1} Sigma_{-a,a}.
    """
    T = sigma.shape[0]
    observed = get_observed_indices(T, masked_index)

    sigma_aa = sigma[masked_index, masked_index]
    sigma_a_obs = sigma[masked_index, observed]
    sigma_obs_a = sigma[observed, masked_index]
    sigma_obs_obs = sigma[np.ix_(observed, observed)]

    cond_var = sigma_aa - sigma_a_obs @ np.linalg.inv(sigma_obs_obs) @ sigma_obs_a
    return float(cond_var)


def bayes_population_risk_uniform_mask(sigma: np.ndarray) -> float:
    """
    Compute the Bayes population risk averaged uniformly over mask positions:
    L_Bayes = (1/T) sum_a sigma_{a|-a}^2.
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
    Compute the empirical masked reconstruction MSE per coordinate:
        (1 / (n d)) sum_i ||x_{a_i}^{(i)} - x_hat^{(i)}||_2^2
    """
    n_samples, _, d = X.shape

    if predictions.shape != (n_samples, d):
        raise ValueError("predictions must have shape (n_samples, d).")
    if mask_indices.shape != (n_samples,):
        raise ValueError("mask_indices must have shape (n_samples,).")

    true_masked_tokens = X[np.arange(n_samples), mask_indices, :]
    mse = np.mean(np.sum((true_masked_tokens - predictions) ** 2, axis=1) / d)
    return float(mse)