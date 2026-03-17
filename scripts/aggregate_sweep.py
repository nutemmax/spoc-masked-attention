from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def collect_runs(sweep_dir: Path) -> list[dict]:
    rows = []

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "plots":
            continue

        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[skip] Failed to read {metrics_path}: {e}")
            continue

        row = {"run_name": subdir.name, **metrics}
        rows.append(row)

    return rows


def collect_run_configs(sweep_dir: Path) -> list[dict]:
    configs = []

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "plots":
            continue

        config_path = subdir / "config.json"
        if not config_path.exists():
            continue

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


def build_sweep_metadata(run_configs: list[dict], sweep_key: str) -> dict:
    if not run_configs:
        raise ValueError("No run configs found.")

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

    fieldnames = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_plot(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=x_col)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        plot_df[x_col],
        plot_df[y_col],
        marker="o",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_train_loss_with_ntrain_plot(df: pd.DataFrame, output_path: Path) -> None:
    required_cols = {"alpha", "train_loss", "n_train"}
    if not required_cols.issubset(df.columns):
        return

    plot_df = df[["alpha", "train_loss", "n_train"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by="alpha")

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

    ax1.plot(
        plot_df["alpha"],
        plot_df["train_loss"],
        marker="o",
        linewidth=2,
        label="Train loss",
    )
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("Train loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["alpha"],
        plot_df["n_train"],
        marker="s",
        linestyle="--",
        label="n_train",
    )

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_summary_plots(sweep_dir: Path, sweep_key: str) -> None:
    summary_csv_path = sweep_dir / "summary.csv"
    if not summary_csv_path.exists():
        print(f"[skip] No summary.csv found in {sweep_dir}")
        return

    df = pd.read_csv(summary_csv_path)
    if df.empty:
        print(f"[skip] summary.csv empty in {sweep_dir}")
        return

    if {"population_risk", "train_loss"}.issubset(df.columns):
        df["generalization_gap"] = df["population_risk"] - df["train_loss"]

    if {"population_risk", "bayes_population_risk"}.issubset(df.columns):
        df["excess_population_risk"] = df["population_risk"] - df["bayes_population_risk"]

    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [
        "train_loss",
        "population_risk",
        "generalization_gap",
        "excess_population_risk",
        "weight_norm",
        "trace_s",
        "top_eigenvalue",
        "R1",
    ]

    for metric in metrics_to_plot:
        save_metric_plot(
            df=df,
            x_col=sweep_key,
            y_col=metric,
            output_path=plots_dir / f"{metric}_vs_{sweep_key}.png",
        )

    if sweep_key == "alpha":
        save_train_loss_with_ntrain_plot(
            df=df,
            output_path=plots_dir / "train_loss_and_ntrain_vs_alpha.png",
        )


def has_run_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False

    for subdir in path.iterdir():
        if not subdir.is_dir():
            continue
        if (subdir / "metrics.json").exists():
            return True
    return False


def expected_plot_paths(sweep_dir: Path, sweep_key: str) -> list[Path]:
    plots_dir = sweep_dir / "plots"
    names = [
        f"train_loss_vs_{sweep_key}.png",
        f"population_risk_vs_{sweep_key}.png",
        f"generalization_gap_vs_{sweep_key}.png",
        f"excess_population_risk_vs_{sweep_key}.png",
        f"weight_norm_vs_{sweep_key}.png",
        f"trace_s_vs_{sweep_key}.png",
        f"top_eigenvalue_vs_{sweep_key}.png",
        f"R1_vs_{sweep_key}.png",
    ]
    if sweep_key == "alpha":
        names.append("train_loss_and_ntrain_vs_alpha.png")
    return [plots_dir / name for name in names]


def is_aggregated(sweep_dir: Path, sweep_key: str) -> bool:
    summary_path = sweep_dir / "summary.csv"
    metadata_path = sweep_dir / "sweep_config.json"

    if not summary_path.exists() or not metadata_path.exists():
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
    generate_summary_plots(sweep_dir, sweep_key)

    run_configs = collect_run_configs(sweep_dir)
    metadata = build_sweep_metadata(run_configs, sweep_key)
    save_sweep_metadata(sweep_dir, metadata)

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