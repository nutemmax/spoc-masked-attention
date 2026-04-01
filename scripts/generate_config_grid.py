from __future__ import annotations

import copy
import itertools
from pathlib import Path
import numpy as np
import yaml
import sys

# import covariance check PSD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.covariance import build_covariance, is_positive_definite

iters = 5000
masking_str = "random"
OUTPUT_DIR = Path("configs/numerics-maskrandom")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_CONFIG_PATH = Path("configs/default.yaml")


def is_valid_covariance_config(
    covariance_type: str,
    T: int,
    rho: float | None,
    length_scale: float | None,
    eta: float | None,
) -> bool:
    try:
        sigma = build_covariance(
            covariance_type=covariance_type,
            T=T,
            rho=rho,
            length_scale=length_scale,
            eta=eta,
        )
    except Exception as e:
        print(
            f"[skip] Invalid covariance config: "
            f"cov={covariance_type}, T={T}, rho={rho}, length_scale={length_scale}, eta={eta}. "
            f"Reason: {e}"
        )
        return False

    if not is_positive_definite(sigma):
        print(
            f"[skip] Non-PD covariance: "
            f"cov={covariance_type}, T={T}, rho={rho}, length_scale={length_scale}, eta={eta}"
        )
        return False

    return True

def load_base_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_float(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("+", "")


def covariance_uses_rho(covariance_type: str) -> bool:
    return covariance_type in {
        "toeplitz",
        "tridiagonal",
        "circulant_ar1",
        "toeplitz_bump",
        "toeplitz_nn",
    }


def covariance_uses_length_scale(covariance_type: str) -> bool:
    return covariance_type in {
        "exponential",
        "matern",
    }


def covariance_uses_eta(covariance_type: str) -> bool:
    return covariance_type in {
        "toeplitz_bump",
        "toeplitz_nn",
    }


def build_config_name(
    covariance_type: str,
    rho: float | None,
    length_scale: float | None,
    eta: float | None,
    lambda_reg: float,
    beta: float,
    d: int,
    T: int,
    learning_rate: float,
    n_steps: int,
    masking_strategy: str,
) -> str:
    parts = [
        f"cov_{covariance_type}",
        f"mask{masking_strategy}",
    ]

    if rho is not None:
        parts.append(f"rho{format_float(rho)}")

    if length_scale is not None:
        parts.append(f"l{format_float(length_scale)}")

    if eta is not None:
        parts.append(f"eta{format_float(eta)}")

    parts.extend([
        f"lambda{format_float(lambda_reg)}",
        f"beta{format_float(beta)}",
        f"d{d}",
        f"T{T}",
        f"lr{format_float(learning_rate)}",
        f"iter{n_steps}",
    ])

    return "_".join(parts)


def keep_chessboard(index_tuple: tuple[int, ...]) -> bool:
    return sum(index_tuple) % 2 == 0


def get_active_values(
    covariance_type: str,
    rhos: list[float],
    length_scales: list[float],
    etas: list[float],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    active_rhos: list[float | None]
    active_length_scales: list[float | None]
    active_etas: list[float | None]

    active_rhos = rhos if covariance_uses_rho(covariance_type) else [None]
    active_length_scales = length_scales if covariance_uses_length_scale(covariance_type) else [None]
    active_etas = etas if covariance_uses_eta(covariance_type) else [None]

    if covariance_uses_rho(covariance_type) and len(active_rhos) == 0:
        raise ValueError(f"covariance_type='{covariance_type}' requires non-empty rhos")

    if covariance_uses_length_scale(covariance_type) and len(active_length_scales) == 0:
        raise ValueError(f"covariance_type='{covariance_type}' requires non-empty length_scales")

    if covariance_uses_eta(covariance_type) and len(active_etas) == 0:
        raise ValueError(f"covariance_type='{covariance_type}' requires non-empty etas")

    return active_rhos, active_length_scales, active_etas


def main() -> None:
    base_config = load_base_config(BASE_CONFIG_PATH)

    covariance_types = [
        "toeplitz",
        "identity",
        # "tridiagonal",
        # "exponential",
        # "matern",
        # "circulant_ar1",
        # "toeplitz_bump",
    ]

    rhos = [0.2, 0.9, 0.99]
    length_scales = [0.5, 1.0, 2]
    etas = [0.1, 0.2, 1.0]
    lambda_regs = [1e-5]
    betas = [1.0]
    ds = [50, 100, 200]
    Ts = [20, 50]
    learning_rates = [1e-3]
    n_steps_list = [iters]
    masking_strategies = [masking_str]

    count = 0

    for covariance_type in covariance_types:
        active_rhos, active_length_scales, active_etas = get_active_values(
            covariance_type=covariance_type,
            rhos=rhos,
            length_scales=length_scales,
            etas=etas,
        )

        dimensions = [
            active_rhos,
            active_length_scales,
            active_etas,
            lambda_regs,
            betas,
            ds,
            Ts,
            learning_rates,
            n_steps_list,
            masking_strategies,
        ]

        for index_tuple in itertools.product(*[range(len(dim)) for dim in dimensions]):
            if not keep_chessboard(index_tuple):
                continue

            (
                i_rho,
                i_ls,
                i_eta,
                i_lam,
                i_beta,
                i_d,
                i_T,
                i_lr,
                i_steps,
                i_mask,
            ) = index_tuple

            rho = active_rhos[i_rho]
            length_scale = active_length_scales[i_ls]
            eta = active_etas[i_eta]
            lambda_reg = lambda_regs[i_lam]
            beta = betas[i_beta]
            d = ds[i_d]
            T = Ts[i_T]
            learning_rate = learning_rates[i_lr]
            n_steps = n_steps_list[i_steps]
            masking_strategy = masking_strategies[i_mask]

            if covariance_type == "tridiagonal" and rho is not None and rho > 0.5:
                continue

            if not is_valid_covariance_config(covariance_type, T, rho, length_scale, eta):
                continue

            config = copy.deepcopy(base_config)

            config["data"]["covariance_type"] = covariance_type
            config["data"]["rho"] = rho
            config["data"]["length_scale"] = length_scale
            config["data"]["eta"] = eta
            config["data"]["d"] = d
            config["data"]["T"] = T
            config["data"]["masking_strategy"] = masking_strategy

            config["model"]["r"] = d
            config["model"]["beta"] = beta

            config["training"]["lambda_reg"] = lambda_reg
            config["training"]["learning_rate"] = learning_rate
            config["training"]["n_steps"] = n_steps

            config_name = build_config_name(
                covariance_type=covariance_type,
                rho=rho,
                length_scale=length_scale,
                eta=eta,
                lambda_reg=lambda_reg,
                beta=beta,
                d=d,
                T=T,
                learning_rate=learning_rate,
                n_steps=n_steps,
                masking_strategy=masking_strategy,
            )

            output_path = OUTPUT_DIR / f"{config_name}.yaml"
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            count += 1

    print(f"Generated {count} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()