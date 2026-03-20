from __future__ import annotations
import numpy as np


def make_identity_covariance(T: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    return np.eye(T, dtype=np.float64)


def make_tridiagonal_covariance(T: int, rho: float, tol : float = 1e-10) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    sigma = np.eye(T, dtype=np.float64)
    for i in range(T - 1):
        sigma[i, i + 1] = rho
        sigma[i + 1, i] = rho

    # psd check
    eigvals = np.linalg.eigvalsh(sigma)
    min_eig = eigvals.min()
    if min_eig < -tol:
        raise ValueError(
            f"Covariance matrix is not PSD. "
            f"Smallest eigenvalue = {min_eig:.3e} < -{tol:.1e}."
        )
    return sigma


def make_toeplitz_covariance(T: int, rho: float) -> np.ndarray:
    if T<= 0:
        raise ValueError("T must be a positive integer.")
    if not (-1 < rho < 1):
        raise ValueError("Rho must be in the range (-1, 1) for a valid covariance matrix.")
    
    sigma = np.eye(T, dtype = np.float64)
    for k in range(1,T) : 
        value  = rho**k
        sigma[np.arange(T-k), np.arange(k, T)] = value
        sigma[np.arange(k, T), np.arange(T-k)] = value
    return sigma

def make_exponential_covariance(T: int, length_scale: float) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")

    indices = np.arange(T)
    distances = np.abs(indices[:, None] - indices[None, :])
    sigma = np.exp(-distances / length_scale)
    return sigma.astype(np.float64)


def make_matern_covariance(T: int, length_scale: float) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")

    indices = np.arange(T)
    distances = np.abs(indices[:, None] - indices[None, :])
    scaled_distances = distances / length_scale
    sigma = (1.0 + scaled_distances) * np.exp(-scaled_distances)
    return sigma.astype(np.float64)



def make_circulant_ar1_covariance(T: int, rho: float) -> np.ndarray:
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    if not (-1 < rho < 1):
        raise ValueError("rho must be in the range (-1, 1).")

    indices = np.arange(T)
    abs_distances = np.abs(indices[:, None] - indices[None, :])
    circular_distances = np.minimum(abs_distances, T - abs_distances)
    sigma = rho ** circular_distances
    return sigma.astype(np.float64)


def make_toeplitz_bump_covariance(T: int, rho: float, eta: float) -> np.ndarray:
    """returns toeplitz/ar(1) covariance with an extra nearest-neighbor bump"""
    if T <= 0:
        raise ValueError("T must be a positive integer.")
    if not (-1 < rho < 1):
        raise ValueError("rho must be in the range (-1, 1).")

    indices = np.arange(T)
    distances = np.abs(indices[:, None] - indices[None, :])

    sigma = (rho ** distances).astype(np.float64)
    sigma += eta * (distances == 1)

    return sigma


def build_covariance(
    covariance_type: str,
    T: int,
    rho: float | None = None,
    length_scale: float | None = None,
    eta: float | None = None,
) -> np.ndarray:
    """Factory function for covariance matrices."""

    covariance_type = covariance_type.lower()
    if covariance_type == "identity":
        return make_identity_covariance(T)

    if covariance_type == "tridiagonal":
        if rho is None:
            raise ValueError("rho must be provided for tridiagonal covariance.")
        return make_tridiagonal_covariance(T, rho)

    if covariance_type == "toeplitz":
        if rho is None:
            raise ValueError("rho must be provided for toeplitz covariance.")
        return make_toeplitz_covariance(T, rho)

    if covariance_type == "exponential":
        if length_scale is None:
            raise ValueError("length_scale must be provided for exponential covariance.")
        return make_exponential_covariance(T, length_scale)

    if covariance_type == "matern":
        if length_scale is None:
            raise ValueError("length_scale must be provided for matern covariance.")
        return make_matern_covariance(T, length_scale)

    if covariance_type == "circulant_ar1":
        if rho is None:
            raise ValueError("rho must be provided for circulant_ar1 covariance.")
        return make_circulant_ar1_covariance(T, rho)

    if covariance_type in ["toeplitz_bump", "toeplitz_nn"]:
        if rho is None:
            raise ValueError("rho must be provided for toeplitz_bump covariance.")
        if eta is None:
            raise ValueError("eta must be provided for toeplitz_bump covariance.")
        return make_toeplitz_bump_covariance(T, rho, eta)

    raise ValueError(
        f"Unknown covariance_type='{covariance_type}'. "
        "Supported values are: "
        "'identity', 'tridiagonal', 'toeplitz', 'exponential', "
        "'matern', 'circulant_ar1', 'toeplitz_bump', 'toeplitz_nn'."
    )


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-12) -> bool:
    """check whether a matrix is PD by checking if all eigenvals are >0"""
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigvals > tol))