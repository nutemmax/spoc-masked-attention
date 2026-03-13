from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ensure repository root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiment import (
    apply_overrides,
    create_run_dir,
    load_config,
    run_experiment,
    save_experiment_outputs,
)


def parse_alpha_list(alpha_string: str) -> list[float]:
    """Parse a comma-separated list of alphas."""
    try:
        alphas = [float(x.strip()) for x in alpha_string.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"Could not parse alpha list: {alpha_string}") from e

    if not alphas:
        raise ValueError("Alpha list must not be empty.")
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("All alpha values must be positive.")

    return alphas


def parse_seed_list(seed_string: str) -> list[int]:
    """Parse a comma-separated list of seeds."""
    try:
        seeds = [int(x.strip()) for x in seed_string.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"Could not parse seed list: {seed_string}") from e

    if not seeds:
        raise ValueError("Seed list must not be empty.")

    return seeds


def format_float_for_name(x: float) -> str:
    """Format a float compactly for folder names."""
    if float(x).is_integer():
        return f"{x:.1f}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def build_sweep_name(config: dict, alphas: list[float]) -> str:
    """Build the sweep folder name."""
    alpha_min = format_float_for_name(min(alphas))
    alpha_max = format_float_for_name(max(alphas))
    beta = format_float_for_name(float(config["model"]["beta"]))
    lr = format_float_for_name(float(config["training"]["learning_rate"]))
    lambda_reg = format_float_for_name(float(config["training"]["lambda_reg"]))
    covariance_type = str(config["data"]["covariance_type"])
    n_steps = int(config["training"]["n_steps"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    return (
        f"alpha_{alpha_min}-{alpha_max}"
        f"_beta_{beta}"
        f"_lr_{lr}"
        f"_lambda_{lambda_reg}"
        f"_cov_{covariance_type}"
        f"_steps_{n_steps}"
        f"_{timestamp}"
    )


def create_sweep_dir(config: dict, alphas: list[float]) -> Path:
    """Create the sweep directory inside results/sweep_alpha/iter_<n_steps>."""
    n_steps = int(config["training"]["n_steps"])

    sweep_root = PROJECT_ROOT / "results" / "sweep_alpha" / f"iter_{n_steps}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_name = build_sweep_name(config, alphas)
    sweep_dir = sweep_root / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=False)

    return sweep_dir


def write_summary_csv(rows: list[dict], path: Path) -> None:
    """Write summary metrics to CSV."""
    if not rows:
        return

    fieldnames = [
        "run_name",
        "experiment_number",
        "alpha",
        "seed",
        "n_train",
        "n_population",

        # risk / performance
        "train_loss",
        "population_risk",
        "bayes_population_risk",
        "empirical_bayes_risk",
        "generalization_gap",
        "excess_population_risk",

        # runtime
        "runtime_seconds",
        "runtime_per_step_seconds",

        # convergence
        "initial_objective",
        "final_objective",
        "best_objective",
        "objective_reduction",
        "initial_train_loss_history",
        "final_train_loss_history",
        "best_train_loss_history",
        "train_loss_reduction",

        # spectral
        "weight_norm",
        "trace_s",
        "top_eigenvalue",
        "min_eigenvalue",
        "R1",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_sweep_metadata(
    sweep_dir: Path,
    base_config: dict,
    alphas: list[float],
    seeds: list[int],
) -> None:
    """Save metadata for reproducibility."""
    metadata = {
        "alphas": alphas,
        "seeds": seeds,
        "base_config": base_config,
    }
    with open(sweep_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_alpha_sweep(
    base_config: dict,
    alphas: list[float],
    seeds: list[int],
    sweep_dir: Path,
) -> list[dict]:
    """Run experiments over all alpha/seed pairs and collect summary rows."""
    summary_rows: list[dict] = []
    experiment_counter = 1

    for alpha in alphas:
        for seed in seeds:
            config = copy.deepcopy(base_config)

            # Each run is saved inside the current sweep directory
            relative_save_root = str(sweep_dir.relative_to(PROJECT_ROOT))
            config = apply_overrides(
                config=config,
                alpha=alpha,
                seed=seed,
                save_root=relative_save_root,
                run_name=None,
                experiment_number=experiment_counter,
            )

            print(f"\n=== Running experiment {experiment_counter}: alpha={alpha}, seed={seed} ===")

            run_dir = create_run_dir(config)
            results = run_experiment(config)
            save_experiment_outputs(results, run_dir)

            metrics = results["metrics"]
            row = {
                "run_name": run_dir.name,
                "experiment_number": experiment_counter,
                "alpha": metrics["alpha"],
                "seed": metrics["seed"],
                "n_train": metrics["n_train"],
                "n_population": metrics["n_population"],

                # risk / performance
                "train_loss": metrics["train_loss"],
                "population_risk": metrics["population_risk"],
                "bayes_population_risk": metrics["bayes_population_risk"],
                "empirical_bayes_risk": metrics["empirical_bayes_risk"],
                "generalization_gap": metrics["generalization_gap"],
                "excess_population_risk": metrics["excess_population_risk"],

                # spectral
                "weight_norm": metrics["weight_norm"],
                "trace_s": metrics["trace_s"],
                "top_eigenvalue": metrics["top_eigenvalue"],
                "min_eigenvalue": metrics["min_eigenvalue"],
                "R1": metrics["R1"],

                # runtime
                "runtime_seconds": metrics["runtime_seconds"],
                "runtime_per_step_seconds": metrics["runtime_per_step_seconds"],

                # convergence
                "initial_objective": metrics["initial_objective"],
                "final_objective": metrics["final_objective"],
                "best_objective": metrics["best_objective"],
                "objective_reduction": metrics["objective_reduction"],
                "initial_train_loss_history": metrics["initial_train_loss_history"],
                "final_train_loss_history": metrics["final_train_loss_history"],
                "best_train_loss_history": metrics["best_train_loss_history"],
                "train_loss_reduction": metrics["train_loss_reduction"],
            }
            summary_rows.append(row)

            print(f"Saved run to: {run_dir}")
            print(json.dumps(metrics, indent=2))

            experiment_counter += 1

    return summary_rows

def save_metric_plot(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path) -> None:
    """Save a single metric-vs-alpha plot with readable labels and larger text."""
    if y_col not in df.columns:
        return

    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=x_col)

    labels = {
        "alpha": "Alpha",
        "train_loss": "Training loss",
        "population_risk": "Population risk",
        "empirical_bayes_risk": "Empirical Bayes population risk",
        "generalization_gap": "Generalization gap = population risk - training loss",
        "excess_population_risk": "Excess population risk = Population risk - Bayes population risk",
    }

    titles = {
        "train_loss": "Training loss vs alpha",
        "population_risk": "Population risk vs alpha",
        "empirical_bayes_risk": "Empirical Bayes population risk vs alpha",
        "generalization_gap": "Generalization gap vs alpha",
        "excess_population_risk": "Excess population risk (Population risk - Bayes population risk) vs alpha",
        "weight_norm": "Weight norm vs alpha",
        "trace_s": "Trace of S vs alpha",
        "top_eigenvalue": "Largest eigenvalue of S vs alpha",
        "R1": "Spectral concentration R1 vs alpha",
    }

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(
        plot_df[x_col],
        plot_df[y_col],
        marker="o",
        linewidth=2.0,
        markersize=6,
        label=labels.get(y_col, y_col),
    )

    ax.set_xlabel(labels.get(x_col, x_col), fontsize=13)
    ax.set_ylabel(labels.get(y_col, y_col), fontsize=13)
    ax.set_title(titles.get(y_col, f"{y_col} vs {x_col}"), fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_train_loss_with_ntrain_plot(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a combined plot of train loss and training set size versus alpha
    """
    required_cols = {"alpha", "train_loss", "n_train"}
    if not required_cols.issubset(df.columns):
        return

    plot_df = df[["alpha", "train_loss", "n_train"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by="alpha")

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))

    line1 = ax1.plot(
        plot_df["alpha"],
        plot_df["train_loss"],
        marker="o",
        linewidth=2.0,
        markersize=6,
        label="Training loss",
    )
    ax1.set_xlabel("Alpha", fontsize=13)
    ax1.set_ylabel("Training loss", fontsize=13)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        plot_df["alpha"],
        plot_df["n_train"],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=5,
        label="Training set size",
    )
    ax2.set_ylabel("Training set size", fontsize=13)
    ax2.tick_params(axis="y", labelsize=11)

    lines = line1 + line2
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels, fontsize=11, loc="best")

    ax1.set_title("Training loss and training set size vs alpha", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_summary_plots(sweep_dir: Path) -> None:
    """Generate plots from summary.csv and save them in sweep_dir/plots."""
    summary_csv_path = sweep_dir / "summary.csv"
    if not summary_csv_path.exists():
        print(f"[skip] No summary.csv found in {sweep_dir}")
        return

    df = pd.read_csv(summary_csv_path)
    if df.empty:
        print(f"[skip] summary.csv is empty in {sweep_dir}")
        return

    # derived metrics
    if {"population_risk", "train_loss"}.issubset(df.columns):
        df["generalization_gap"] = df["population_risk"] - df["train_loss"]

    if {"population_risk", "bayes_population_risk"}.issubset(df.columns):
        df["excess_population_risk"] = df["population_risk"] - df["bayes_population_risk"]

    if {"population_risk", "empirical_bayes_risk"}.issubset(df.columns):
        df["excess_population_risk_emp"] = df["population_risk"] - df["empirical_bayes_risk"]

    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # main metrics to plot
    metrics_to_plot = [
        "population_risk",
        "generalization_gap",
        "excess_population_risk",
        "trace_s",
        "top_eigenvalue",
        "R1",
    ]

    for metric in metrics_to_plot:
        save_metric_plot(
            df=df,
            x_col="alpha",
            y_col=metric,
            output_path=plots_dir / f"{metric}_vs_alpha.png",
        )

    # separate combined plot for train loss and train set size
    save_train_loss_with_ntrain_plot(
        df=df,
        output_path=plots_dir / "train_loss_and_ntrain_vs_alpha.png",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep SPOC experiments over alpha.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="1.0,1.5,2.0,3.0,5.0",
        help="Comma-separated list of alpha values.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated list of seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_config = load_config(args.config)
    alphas = parse_alpha_list(args.alphas)
    seeds = parse_seed_list(args.seeds)

    sweep_dir = create_sweep_dir(base_config, alphas)
    save_sweep_metadata(sweep_dir, base_config, alphas, seeds)

    summary_rows = run_alpha_sweep(
        base_config=base_config,
        alphas=alphas,
        seeds=seeds,
        sweep_dir=sweep_dir,
    )

    write_summary_csv(summary_rows, sweep_dir / "summary.csv")
    generate_summary_plots(sweep_dir)

    print(f"\nFinished alpha sweep. Summary saved to: {sweep_dir / 'summary.csv'}")
    print(f"Plots saved to: {sweep_dir / 'plots'}")

if __name__ == "__main__":
    main()