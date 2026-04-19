from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path


def get_logging_config(config: dict) -> dict:
    """
    Return logging config.
    If the logging section is missing, return an empty dict so that
    W&B logging is simply disabled by default.
    """
    logging_cfg = config.get("logging", {})
    if logging_cfg is None:
        return {}
    if not isinstance(logging_cfg, dict):
        raise ValueError("Config section 'logging' must be a dictionary if provided.")
    return logging_cfg


def is_wandb_enabled(config: dict) -> bool:
    logging_cfg = get_logging_config(config)
    return bool(logging_cfg.get("use_wandb", False))


def format_float_for_name(x: float) -> str:
    if float(x).is_integer():
        return f"{x:.1f}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def get_config_name(config: dict) -> str:
    """
    Get a config name for W&B grouping.

    Priority:
    1. logging.config_name if provided
    2. experiment.config_name if provided
    3. experiment.run_name if provided
    4. 'default'
    """
    logging_cfg = get_logging_config(config)

    if logging_cfg.get("config_name") is not None:
        return str(logging_cfg["config_name"])

    experiment_cfg = config.get("experiment", {})
    if experiment_cfg.get("config_name") is not None:
        return str(experiment_cfg["config_name"])

    if experiment_cfg.get("run_name") is not None:
        return str(experiment_cfg["run_name"])

    return "default"


def get_slurm_ids() -> dict[str, str | None]:
    # relevant Slurm identifiers if available
    return {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }


def get_effective_job_id() -> str | None:
    slurm_ids = get_slurm_ids()
    return slurm_ids["array_job_id"] or slurm_ids["job_id"]


def is_sweep_run(config: dict) -> bool:
    """
    Decide whether the current run should be treated as part of a sweep.
    Logic:
    - if logging.is_sweep is explicitly set, use it
    - else if SLURM_ARRAY_TASK_ID exists, treat as sweep
    - else default to False
    """
    logging_cfg = get_logging_config(config)
    if logging_cfg.get("is_sweep") is not None:
        return bool(logging_cfg["is_sweep"])

    return os.getenv("SLURM_ARRAY_TASK_ID") is not None


def build_wandb_group(config: dict) -> str:
    return get_config_name(config)


def build_wandb_job_type(config: dict) -> str:
    config_name = get_config_name(config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = get_effective_job_id()
    sweep_flag = is_sweep_run(config)

    run_kind = "sweep" if sweep_flag else "single"

    if job_id is not None:
        return f"{config_name}__{run_kind}__job_{job_id}"

    return f"{config_name}__{run_kind}__{timestamp}"


def build_wandb_run_name(
    config: dict,
    actual_n_train: int | None = None,
    alpha: float | None = None,
    seed: int | None = None,
) -> str:
    config_name = get_config_name(config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if seed is None:
        seed = config.get("experiment", {}).get("seed")

    parts = [config_name]

    if actual_n_train is not None:
        parts.append(f"ntrain_{int(actual_n_train)}")
    elif alpha is not None:
        parts.append(f"alpha_{format_float_for_name(float(alpha))}")

    if seed is not None:
        parts.append(f"seed_{int(seed)}")

    slurm_ids = get_slurm_ids()
    job_id = slurm_ids["array_job_id"] or slurm_ids["job_id"]
    task_id = slurm_ids["array_task_id"]

    if job_id is not None:
        parts.append(f"job_{job_id}")
    if task_id is not None:
        parts.append(f"task_{task_id}")

    parts.append(timestamp)

    return "__".join(parts)


def build_wandb_tags(config: dict) -> list[str]:
    tags: list[str] = []

    config_name = get_config_name(config)
    tags.append(f"config:{config_name}")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    teacher_cfg = config.get("teacher", {})

    data_model = data_cfg.get("data_model")
    if data_model is not None:
        tags.append(f"data:{data_model}")

    masking_strategy = data_cfg.get("masking_strategy")
    if masking_strategy is not None:
        tags.append(f"mask:{masking_strategy}")

    covariance_type = data_cfg.get("covariance_type")
    if covariance_type is not None:
        tags.append(f"cov:{covariance_type}")

    rho = data_cfg.get("rho")
    if rho is not None:
        tags.append(f"rho:{rho}")

    beta = model_cfg.get("beta")
    if beta is not None:
        tags.append(f"beta:{beta}")

    teacher_init = teacher_cfg.get("init")
    if teacher_init is not None:
        tags.append(f"teacher_init:{teacher_init}")

    r_star = teacher_cfg.get("r_star")
    if r_star is not None:
        tags.append(f"r_star:{r_star}")

    beta_star = teacher_cfg.get("beta_star")
    if beta_star is not None:
        tags.append(f"beta_star:{beta_star}")

    sigma_star = teacher_cfg.get("sigma_star")
    if sigma_star is not None:
        tags.append(f"sigma_star:{sigma_star}")

    lambda_reg = training_cfg.get("lambda_reg")
    if lambda_reg is not None:
        tags.append(f"lambda:{lambda_reg}")

    tags.append("sweep" if is_sweep_run(config) else "single")

    T = data_cfg.get("T")
    if T is not None:
        tags.append(f"T:{T}")

    d = data_cfg.get("d")
    if d is not None:
        tags.append(f"d:{d}")

    r = model_cfg.get("r")
    if r is not None:
        tags.append(f"r:{r}")

    if os.getenv("SLURM_JOB_ID") is not None or os.getenv("SLURM_ARRAY_JOB_ID") is not None:
        tags.append("slurm")

    return tags

def init_wandb_if_enabled(
    config: dict,
    actual_n_train: int | None = None,
    alpha: float | None = None,
    seed: int | None = None,
):
    if not is_wandb_enabled(config):
        return None

    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "W&B logging requested but wandb is not installed. "
            "Install it with `pip install wandb`."
        ) from e

    logging_cfg = get_logging_config(config)

    run_name = logging_cfg.get("run_name")
    if run_name is None:
        run_name = build_wandb_run_name(
            config=config,
            actual_n_train=actual_n_train,
            alpha=alpha,
            seed=seed,
        )

    wandb.init(
        project=logging_cfg.get("project", "spoc-masked-attention"),
        entity=logging_cfg.get("entity", None),
        name=run_name,
        group=build_wandb_group(config),
        job_type=build_wandb_job_type(config),
        tags=build_wandb_tags(config),
        config=copy.deepcopy(config),
    )

    return wandb

def log_training_history(
    wandb_module,
    history: dict[str, list[float]],
    eval_every: int,
) -> None:
    _ = eval_every # currently unused, but could be used in the future to validate that eval steps are logged at the expected intervals
    if wandb_module is None:
        return

    objective = history.get("objective", [])
    train_loss = history.get("train_loss", [])
    eval_steps = history.get("steps", [])

    n_steps = len(objective)

    eval_keys = [
        key
        for key in history.keys()
        if key not in {"objective", "train_loss", "steps"}
    ]

    eval_map: dict[int, dict[str, float]] = {}

    for i, step_value in enumerate(eval_steps):
        step = int(step_value)
        payload: dict[str, float] = {}

        for key in eval_keys:
            values = history.get(key, [])
            if i < len(values):
                payload[key] = float(values[i])

        eval_map[step] = payload

    for step_idx in range(n_steps):
        step = step_idx + 1
        payload: dict[str, float] = {}

        if step_idx < len(objective):
            payload["objective"] = float(objective[step_idx])

        if step_idx < len(train_loss):
            payload["train_loss"] = float(train_loss[step_idx])

        if step in eval_map:
            payload.update(eval_map[step])

        wandb_module.log(payload, step=step)


def log_final_metrics(wandb_module, metrics: dict, step: int) -> None:
    if wandb_module is None:
        return
    payload = {
        f"final/{k}": v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) or v is None
    }
    wandb_module.log(payload, step=step)


def finish_wandb(wandb_module) -> None:
    if wandb_module is not None:
        wandb_module.finish()