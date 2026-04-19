from __future__ import annotations
import copy
from pathlib import Path
import torch
import yaml


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file '{config_path}' is empty.")
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a dictionary at top level.")

    return config


# def validate_config(config: dict) -> None:
#     """
#     Validate that all required top-level sections and nested keys are present in a config.
#     """
#     required = {
#         "experiment": ["save_root", "run_name", "seed"],
#         "data": [
#             "T",
#             "d",
#             "covariance_type",
#             "rho",
#             "length_scale",
#             "eta",
#             "mask_value",
#             "masking_strategy",
#         ],
#         "model": ["r", "beta", "normalize_sqrt_d", "dtype", "device"],
#         "training": ["n_steps", "learning_rate", "lambda_reg"],
#         "evaluation": ["n_population"],
#     }

#     for section, keys in required.items():
#         if section not in config:
#             raise ValueError(f"Missing required config section: '{section}'")
#         if not isinstance(config[section], dict):
#             raise ValueError(f"Config section '{section}' must be a dictionary.")

#         for key in keys:
#             if key not in config[section]:
#                 raise ValueError(f"Missing required config key: '{section}.{key}'")

#     training_cfg = config["training"]
#     has_alpha = "alpha" in training_cfg and training_cfg["alpha"] is not None
#     has_n_train = "n_train" in training_cfg and training_cfg["n_train"] is not None

#     if not (has_alpha or has_n_train):
#         raise ValueError("Config must provide at least one of 'training.alpha' or 'training.n_train'.")


def validate_config(config: dict) -> None:
    """Validate config for fixed-sigma or teacher-attention experiments."""
    required_common = {
        "experiment": ["save_root", "run_name", "seed"],
        "data": ["T", "d", "mask_value", "masking_strategy"],
        "model": ["r", "beta", "normalize_sqrt_d", "dtype", "device"],
        "training": ["n_steps", "learning_rate", "lambda_reg"],
        "evaluation": ["n_population"],
    }

    for section, keys in required_common.items():
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")
        if not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be a dictionary.")

        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required config key: '{section}.{key}'")

    data_model = config["data"].get("data_model", "fixed_sigma")

    if data_model == "fixed_sigma":
        fixed_sigma_keys = ["covariance_type", "rho", "length_scale", "eta"]
        for key in fixed_sigma_keys:
            if key not in config["data"]:
                raise ValueError(f"Missing required config key for fixed_sigma: 'data.{key}'")

    elif data_model == "teacher_attention":
        if "teacher" not in config:
            raise ValueError("Missing required config section for teacher_attention: 'teacher'")
        if not isinstance(config["teacher"], dict):
            raise ValueError("Config section 'teacher' must be a dictionary.")

        teacher_keys = ["init", "r_star", "beta_star", "sigma_star"]
        for key in teacher_keys:
            if key not in config["teacher"]:
                raise ValueError(f"Missing required config key: 'teacher.{key}'")

    else:
        raise ValueError(
            f"Unknown data_model='{data_model}'. Use 'fixed_sigma' or 'teacher_attention'."
        )

    training_cfg = config["training"]
    has_alpha = "alpha" in training_cfg and training_cfg["alpha"] is not None
    has_n_train = "n_train" in training_cfg and training_cfg["n_train"] is not None

    if not (has_alpha or has_n_train):
        raise ValueError("Config must provide at least one of 'training.alpha' or 'training.n_train'.")

def apply_overrides(config: dict,
    alpha: float | None = None,
    n_train: int | None = None,
    seed: int | None = None,
    save_root: str | None = None,
    run_name: str | None = None,
) -> dict:
    """
    Apply command-line overrides to a loaded config.
    """
    updated = copy.deepcopy(config)

    if alpha is not None:
        updated["training"]["alpha"] = float(alpha)
    if n_train is not None:
        updated["training"]["n_train"] = int(n_train)
    if seed is not None:
        updated["experiment"]["seed"] = int(seed)
    if save_root is not None:
        updated["experiment"]["save_root"] = str(save_root)
    if run_name is not None:
        updated["experiment"]["run_name"] = str(run_name)

    return updated


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()

    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32

    raise ValueError("dtype must be 'float64' or 'float32'.")