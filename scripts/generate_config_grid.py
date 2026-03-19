from __future__ import annotations

import itertools
from pathlib import Path
import yaml
import copy

iters = 1000
masking_str = "random"
OUTPUT_DIR = Path(f"configs/generated_grid_biggerT_mask{masking_str}_{iters}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG_PATH = Path("configs/default.yaml")


def load_base_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_float(x: float) -> str:
    """Compact float formatting for filenames."""
    s = f"{x:.6g}"
    return s.replace(".", "p")


def build_config_name(
    covariance_type: str,
    rho,
    lambda_reg: float,
    beta: float,
    d: int,
    T: int,
    learning_rate: float,
    n_steps: int,
    masking_strategy: str,
) -> str:
    rho_str = "none" if rho is None else format_float(rho)

    return (
        f"cov_{covariance_type}"
        f"_mask{masking_strategy}"
        f"_rho{rho_str}"
        f"_lambda{format_float(lambda_reg)}"
        f"_beta{format_float(beta)}"
        f"_d{d}"
        f"_T{T}"
        f"_lr{format_float(learning_rate)}"
        f"_iter{n_steps}"
    )


def keep_chessboard(index_tuple: tuple[int, ...]) -> bool:
    return sum(index_tuple) % 1 == 0


def main() -> None:
    base_config = load_base_config(BASE_CONFIG_PATH)

    covariance_types = ["tridiagonal"]
    rhos = [0.5]
    lambda_regs = [1e-5]
    betas = [1.0]
    ds = [50]
    Ts = [5]
    learning_rates = [1e-3]
    n_steps_list = [iters]
    masking_strategies = [f"{masking_str}"]   # or ["random", "last"]

    dimensions = [
        covariance_types,
        rhos,
        lambda_regs,
        betas,
        ds,
        Ts,
        learning_rates,
        n_steps_list,
        masking_strategies,
    ]

    count = 0

    for index_tuple in itertools.product(*[range(len(dim)) for dim in dimensions]):
        if not keep_chessboard(index_tuple):
            continue

        (
            i_cov,
            i_rho,
            i_lam,
            i_beta,
            i_d,
            i_T,
            i_lr,
            i_steps,
            i_mask,
        ) = index_tuple

        covariance_type = covariance_types[i_cov]
        rho = rhos[i_rho]
        lambda_reg = lambda_regs[i_lam]
        beta = betas[i_beta]
        d = ds[i_d]
        T = Ts[i_T]
        learning_rate = learning_rates[i_lr]
        n_steps = n_steps_list[i_steps]
        masking_strategy = masking_strategies[i_mask]

        # consistency rules
        if covariance_type == "identity":
            rho = None

        # tridiagonal requires rho < 0.5 for PD
        if covariance_type == "tridiagonal" and rho is not None and rho > 0.5:
            continue

        config = copy.deepcopy(base_config)

        config["data"]["covariance_type"] = covariance_type
        config["data"]["rho"] = rho
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