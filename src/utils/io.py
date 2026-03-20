from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import numpy as np


def format_float_for_name(x: float) -> str:
    if float(x).is_integer():
        return f"{x:.1f}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def build_run_name(config: dict) -> str:
    # build a name for each run based on the config
    custom_name = config["experiment"]["run_name"]
    if custom_name is not None:
        return str(custom_name)

    seed = int(config["experiment"]["seed"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    n_train = config["training"].get("n_train")
    if n_train is not None:
        return f"ntrain_{int(n_train)}_seed_{seed}_{timestamp}"

    alpha = format_float_for_name(float(config["training"]["alpha"]))
    return f"alpha_{alpha}_seed_{seed}_{timestamp}"


def create_run_dir(project_root: str | Path, config: dict) -> Path:
    # create the run directory
    project_root = Path(project_root)
    save_root = project_root / config["experiment"]["save_root"]
    save_root.mkdir(parents=True, exist_ok=True)

    run_name = build_run_name(config)
    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_run_arrays(run_dir: str | Path,
    W: np.ndarray,
    S: np.ndarray,
    sigma: np.ndarray,
    eigenvalues: np.ndarray,
    config_suffix: str,
) -> None:
    run_dir = Path(run_dir)

    np.save(run_dir / f"W__{config_suffix}.npy", W)
    np.save(run_dir / f"S__{config_suffix}.npy", S)
    np.save(run_dir / f"sigma__{config_suffix}.npy", sigma)
    np.save(run_dir / f"eigenvalues__{config_suffix}.npy", eigenvalues)