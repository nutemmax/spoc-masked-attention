from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = PROJECT_ROOT / "results" / "sweep_alpha"
OUTPUT_PATH = SWEEP_ROOT / "aggregated_summary.csv"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a top-level JSON object.")
    return data


def flatten_base_config(config: dict) -> dict:
    """
    Extract the main hyperparameters from base_config inside sweep_config.json.
    """
    experiment = config.get("experiment", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})

    return {
        # experiment
        "save_root": experiment.get("save_root"),
        "base_seed": experiment.get("seed"),

        # data
        "T": data.get("T"),
        "d": data.get("d"),
        "covariance_type": data.get("covariance_type"),
        "rho": data.get("rho"),
        "length_scale": data.get("length_scale"),
        "eta": data.get("eta"),
        "mask_value": data.get("mask_value"),

        # model
        "r": model.get("r"),
        "beta": model.get("beta"),
        "normalize_sqrt_d": model.get("normalize_sqrt_d"),
        "dtype": model.get("dtype"),
        "device": model.get("device"),

        # training
        "default_alpha_in_config": training.get("alpha"),
        "n_steps": training.get("n_steps"),
        "learning_rate": training.get("learning_rate"),
        "lambda_reg": training.get("lambda_reg"),

        # evaluation
        "n_population_config": evaluation.get("n_population"),
    }


def process_sweep_folder(folder: Path) -> pd.DataFrame | None:
    """
    Read one sweep folder and return a dataframe with:
    - rows from summary.csv
    - hyperparameters from sweep_config.json appended as columns
    """
    sweep_config_path = folder / "sweep_config.json"
    summary_csv_path = folder / "summary.csv"

    if not sweep_config_path.exists():
        print(f"[skip] Missing sweep_config.json in {folder.name}")
        return None

    if not summary_csv_path.exists():
        print(f"[skip] Missing summary.csv in {folder.name}")
        return None

    try:
        sweep_config = load_json(sweep_config_path)
        summary_df = pd.read_csv(summary_csv_path)
    except Exception as e:
        print(f"[skip] Failed reading {folder.name}: {e}")
        return None

    if summary_df.empty:
        print(f"[skip] Empty summary.csv in {folder.name}")
        return None

    base_config = sweep_config.get("base_config")
    if not isinstance(base_config, dict):
        print(f"[skip] Missing or invalid base_config in {folder.name}")
        return None

    config_fields = flatten_base_config(base_config)

    # Optional metadata from sweep_config.json
    alphas = sweep_config.get("alphas")
    seeds = sweep_config.get("seeds")

    summary_df = summary_df.copy()
    for key, value in config_fields.items():
        summary_df[key] = value

    return summary_df


def aggregate_sweeps(sweep_root: Path) -> pd.DataFrame:
    """
    Aggregate all valid sweep folders under results/sweep_alpha.
    """
    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep root does not exist: {sweep_root}")

    all_dfs: list[pd.DataFrame] = []

    for folder in sorted(sweep_root.iterdir()):
        if not folder.is_dir():
            continue

        df = process_sweep_folder(folder)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No valid sweep folders found in {sweep_root}")

    aggregated = pd.concat(all_dfs, ignore_index=True)

    preferred_order = [
        # folder metadata
        "sweep_alphas",

        # sweep summary row identifiers
        "alpha",
        "seed",

        # hyperparameters
        "T",
        "d",
        "r",
        "beta",
        "lambda_reg",
        "covariance_type",
        "rho",
        "length_scale",
        "eta",
        "mask_value",
        "n_steps",
        "learning_rate",
        # "normalize_sqrt_d",
        # "dtype",
        # "device",
        # "base_seed",
        "n_population_config",
        # "save_root",
        # "default_alpha_in_config",

        # metrics
        "n_train",
        "n_population",
        "train_loss",
        "population_risk",
        "bayes_population_risk",
        "empirical_bayes_risk",
        "weight_norm",
        "trace_s",
        "top_eigenvalue",
        "R1",
    ]

    existing_cols = [col for col in preferred_order if col in aggregated.columns]
    remaining_cols = [col for col in aggregated.columns if col not in existing_cols]
    aggregated = aggregated[existing_cols + remaining_cols]

    return aggregated[existing_cols]


def main() -> None:
    aggregated_df = aggregate_sweeps(SWEEP_ROOT)
    aggregated_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved aggregated summary to: {OUTPUT_PATH}")
    print(f"Number of rows: {len(aggregated_df)}")
    print(f"Number of columns: {len(aggregated_df.columns)}")


if __name__ == "__main__":
    main()