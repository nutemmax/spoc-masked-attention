from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# ensure repository root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.utils.plots_alpha as plots_alpha
from src.utils.plots_alpha import expected_plot_paths, generate_summary_plots

def extract_datetime_from_run_name(run_name: str) -> str | None:
    parts = run_name.split("_")
    if not parts:
        return None
    last = parts[-1]
    if len(last) == 15 and last[8] == "-":
        return last
    return None


def find_metrics_files(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_file() and p.name.startswith("metrics") and p.name.endswith(".json")
    )

def find_config_files(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_file() and p.name.startswith("config") and p.name.endswith(".json")
    )


def collect_runs(sweep_dir: Path) -> list[dict]:
    rows = []

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "plots":
            continue

        metrics_files = find_metrics_files(subdir)
        if not metrics_files:
            continue

        if len(metrics_files) > 1:
            print(
                f"[warn] Multiple metrics files found in {subdir}. "
                f"Using the first one: {metrics_files[0].name}"
            )

        metrics_path = metrics_files[0]

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[skip] Failed to read {metrics_path}: {e}")
            continue

        row = {
            "run_name": subdir.name,
            "experiment_number": None,
            **metrics,
            "datetime": extract_datetime_from_run_name(subdir.name),
        }
        rows.append(row)

    return rows


def collect_run_configs(sweep_dir: Path) -> list[dict]:
    configs = []

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "plots":
            continue

        config_files = find_config_files(subdir)
        if not config_files:
            continue

        if len(config_files) > 1:
            print(
                f"[warn] Multiple config files found in {subdir}. "
                f"Using the first one: {config_files[0].name}"
            )

        config_path = config_files[0]

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[skip] Failed to read {config_path}: {e}")
            continue

        configs.append(config)

    return configs


def detect_sweep_key(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("No rows found.")

    has_alpha = any(row.get("alpha") is not None for row in rows)
    has_ntrain = any(row.get("n_train") is not None for row in rows)

    if has_alpha:
        return "alpha"
    if has_ntrain:
        return "n_train"
    raise ValueError("Could not detect sweep variable (alpha or n_train).")


def build_sweep_metadata(run_configs: list[dict], sweep_key: str) -> dict | None:
    if not run_configs:
        return None

    base_config = copy.deepcopy(run_configs[0])

    if "experiment" in base_config:
        base_config["experiment"]["save_root"] = None
        base_config["experiment"]["run_name"] = None
        base_config["experiment"]["seed"] = None

    if "training" in base_config:
        if sweep_key == "alpha":
            base_config["training"]["alpha"] = None
        elif sweep_key == "n_train":
            base_config["training"]["n_train"] = None

    seeds = sorted({
        int(cfg["experiment"]["seed"])
        for cfg in run_configs
        if cfg.get("experiment", {}).get("seed") is not None
    })

    metadata = {
        "sweep_key": sweep_key,
        "seeds": seeds,
        "base_config": base_config,
    }

    if sweep_key == "alpha":
        alphas = sorted({
            float(cfg["training"]["alpha"])
            for cfg in run_configs
            if cfg.get("training", {}).get("alpha") is not None
        })
        metadata["alphas"] = alphas

    elif sweep_key == "n_train":
        n_trains = sorted({
            int(cfg["training"]["n_train"])
            for cfg in run_configs
            if cfg.get("training", {}).get("n_train") is not None
        })
        metadata["n_trains"] = n_trains

    return metadata


def save_sweep_metadata(sweep_dir: Path, metadata: dict) -> None:
    output_path = sweep_dir / "sweep_config.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "alpha",
        "seed",
        "n_train",
        "n_population",
        "train_loss",
        "population_risk",
        "bayes_population_risk",
        "empirical_bayes_risk",
        "generalization_gap",
        "excess_population_risk",
        "runtime_seconds",
        "runtime_per_step_seconds",
        "initial_objective",
        "final_objective",
        "best_objective",
        "objective_reduction",
        "initial_train_loss_history",
        "final_train_loss_history",
        "best_train_loss_history",
        "train_loss_reduction",
        "weight_norm",
        "trace_s",
        "top_eigenvalue",
        "min_eigenvalue",
        "R1",
        "datetime",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def has_run_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False

    for subdir in path.iterdir():
        if not subdir.is_dir():
            continue
        if find_metrics_files(subdir):
            return True
    return False


def is_aggregated(sweep_dir: Path, sweep_key: str) -> bool:
    summary_path = sweep_dir / "summary.csv"
    metadata_path = sweep_dir / "sweep_config.json"

    if not summary_path.exists():
        return False
    if not metadata_path.exists():
        return False

    expected_paths = expected_plot_paths(sweep_dir, sweep_key)
    return all(path.exists() for path in expected_paths)


def aggregate_one_sweep(sweep_dir: Path, force: bool = False) -> None:
    rows = collect_runs(sweep_dir)
    if not rows:
        print(f"[skip] No runs found in {sweep_dir}")
        return

    sweep_key = detect_sweep_key(rows)

    if not force and is_aggregated(sweep_dir, sweep_key):
        print(f"[skip] Already aggregated: {sweep_dir}")
        return

    rows.sort(key=lambda x: x[sweep_key])

    write_summary_csv(rows, sweep_dir / "summary.csv")

    run_configs = collect_run_configs(sweep_dir)
    metadata = build_sweep_metadata(run_configs, sweep_key)

    if metadata is None:
        print(f"[skip] No run configs found in {sweep_dir}")
        return
    save_sweep_metadata(sweep_dir, metadata)
    generate_summary_plots(sweep_dir, sweep_key)

    print(f"[done] Aggregated {sweep_dir} (sweep key: {sweep_key})")


def find_sweep_dirs(root: Path) -> list[Path]:
    sweep_dirs = []

    if has_run_subdirs(root):
        sweep_dirs.append(root)

    for path in root.rglob("*"):
        if has_run_subdirs(path):
            sweep_dirs.append(path)

    unique = []
    seen = set()
    for path in sweep_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.sweep_dir)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    sweep_dirs = find_sweep_dirs(root)

    if not sweep_dirs:
        print("No sweep directories found.")
        return

    for sweep_dir in sweep_dirs:
        aggregate_one_sweep(sweep_dir, force=args.force)


if __name__ == "__main__":
    main()