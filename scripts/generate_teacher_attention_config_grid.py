from __future__ import annotations

import copy
import itertools
from pathlib import Path
import yaml


OUTPUT_DIR = Path("configs/teacher_attention")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BASE_CONFIG = {
    "experiment": {
        "save_root": "results/teacher-attention/individual",
        "run_name": None,
        "seed": 0,
    },
    "data": {
        "data_model": "teacher_attention",
        "T": 5,
        "d": 50,
        "mask_value": 1.0,
        "masking_strategy": "random",
    },
    "teacher": {
        "init": "standard_gaussian",
        "r_star": 50,
        "beta_star": 1.0,
        "sigma_star": 1.0,
    },
    "model": {
        "r": 50,
        "beta": 1.0,
        "normalize_sqrt_d": True,
        "dtype": "float64",
        "device": "cpu",
    },
    "training": {
        "alpha": 1.0,
        "n_train": None,
        "n_steps": 5000,
        "learning_rate": 0.001,
        "lambda_reg": 1.0e-5,
    },
    "evaluation": {
        "n_population": 5000,
        "eval_every": 25,
        "track_attention_error_during_training": True,
        "attention_metric_subset_size": 512,
        "pca_n_components": None,
    },
    "logging": {
        "use_wandb": True,
        "project": "spoc-masked-attention",
    },
}


def format_float(x: float | int | None) -> str:
    if x is None:
        return "NA"
    s = f"{float(x):.6g}"
    return s.replace(".", "p").replace("+", "")


def format_r_star(r_star: int | str | None) -> str:
    if r_star is None:
        return "d"
    return str(r_star)


def active_sigma_values(teacher_init: str, sigma_stars: list[float]) -> list[float]:
    if teacher_init == "standard_gaussian":
        return [1.0]
    if teacher_init == "scaled_gaussian":
        return sigma_stars
    raise ValueError(f"Unknown teacher init: {teacher_init}")


def build_config_name(
    teacher_init: str,
    sigma_star: float,
    r_star: int | str | None,
    beta_star: float,
    beta: float,
    d: int,
    T: int,
    lambda_reg: float,
    learning_rate: float,
    n_steps: int,
    pca_n_components: int | None,
) -> str:
    parts = [
        f"init_{teacher_init}",
        f"rstar_{format_r_star(r_star)}",
    ]

    if teacher_init == "scaled_gaussian":
        parts.append(f"sigstar_{format_float(sigma_star)}")

    parts.extend([
        f"bstar_{format_float(beta_star)}",
        f"beta_{format_float(beta)}",
        f"d{d}",
        f"T{T}",
        f"lambda{format_float(lambda_reg)}",
        f"lr{format_float(learning_rate)}",
        f"iter{n_steps}",
    ])

    if pca_n_components is not None:
        parts.append(f"pca{pca_n_components}")

    return "_".join(parts)


def keep_chessboard(index_tuple: tuple[int, ...]) -> bool:
    return sum(index_tuple) % 2 == 0


def main() -> None:
    use_chessboard = False

    teacher_inits = [
        "standard_gaussian",
        "scaled_gaussian",
    ]
    sigma_stars = [0.5, 1.0, 2.0]

    ds = [50]
    Ts = [5]

    r_star_values = [50]
    # beta_stars = [0.25, 0.5, 1.0, 2.0, 4.0]
    beta_stars = [1.0]
    betas = [1.0]
    lambda_regs = [1e-5] # best one
    learning_rates = [1e-3] # best one
    n_steps_list = [5000]

    pca_n_components_list: list[int | None] = [50] # later to be tuned
    # pca_n_components_list = [25, 50, 75, None]

    count = 0

    for teacher_init in teacher_inits:
        active_sigma_stars = active_sigma_values(teacher_init, sigma_stars)

        dimensions = [
            active_sigma_stars,
            ds,
            Ts,
            r_star_values,
            beta_stars,
            betas,
            lambda_regs,
            learning_rates,
            n_steps_list,
            pca_n_components_list,
        ]

        for index_tuple in itertools.product(*[range(len(dim)) for dim in dimensions]):
            if use_chessboard and not keep_chessboard(index_tuple):
                continue

            (
                i_sigma,
                i_d,
                i_T,
                i_rstar,
                i_beta_star,
                i_beta,
                i_lam,
                i_lr,
                i_steps,
                i_pca,
            ) = index_tuple

            sigma_star = active_sigma_stars[i_sigma]
            d = ds[i_d]
            T = Ts[i_T]
            r_star = r_star_values[i_rstar]
            beta_star = beta_stars[i_beta_star]
            beta = betas[i_beta]
            lambda_reg = lambda_regs[i_lam]
            learning_rate = learning_rates[i_lr]
            n_steps = n_steps_list[i_steps]
            pca_n_components = pca_n_components_list[i_pca]

            if r_star is not None and r_star != "d" and int(r_star) > d:
                print(f"[skip] r_star={r_star} > d={d}")
                continue

            config = copy.deepcopy(BASE_CONFIG)

            config["data"]["d"] = d
            config["data"]["T"] = T

            config["teacher"]["init"] = teacher_init
            config["teacher"]["r_star"] = r_star
            config["teacher"]["beta_star"] = beta_star
            config["teacher"]["sigma_star"] = sigma_star

            config["model"]["r"] = d
            config["model"]["beta"] = beta

            config["training"]["lambda_reg"] = lambda_reg
            config["training"]["learning_rate"] = learning_rate
            config["training"]["n_steps"] = n_steps

            config["evaluation"]["pca_n_components"] = pca_n_components

            config_name = build_config_name(
                teacher_init=teacher_init,
                sigma_star=sigma_star,
                r_star=r_star,
                beta_star=beta_star,
                beta=beta,
                d=d,
                T=T,
                lambda_reg=lambda_reg,
                learning_rate=learning_rate,
                n_steps=n_steps,
                pca_n_components=pca_n_components,
            )

            output_path = OUTPUT_DIR / f"{config_name}.yaml"
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            count += 1

    print(f"Generated {count} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()