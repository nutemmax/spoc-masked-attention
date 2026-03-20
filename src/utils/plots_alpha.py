from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# GLOBAL PLOT STYLE
# =========================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
})


def set_multiline_title(ax, title: str) -> None:
    ax.set_title(title, pad=16)


# =========================
# FORMATTING HELPERS
# =========================
def format_float_for_title(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def format_metric_name(metric: str) -> str:
    mapping = {
        "train_loss": "Train Loss",
        "population_risk": "Population Risk",
        "bayes_population_risk": "Bayes Population Risk",
        "empirical_bayes_risk": "Empirical Bayes Risk",
        "generalization_gap": "Generalization Gap",
        "excess_population_risk": "Excess Population Risk",
        "runtime_seconds": "Runtime",
        "runtime_per_step_seconds": "Runtime per Step",
        "initial_objective": "Initial Objective",
        "final_objective": "Final Objective",
        "best_objective": "Best Objective",
        "objective_reduction": "Objective Reduction",
        "initial_train_loss_history": "Initial Train Loss",
        "final_train_loss_history": "Final Train Loss",
        "best_train_loss_history": "Best Train Loss",
        "train_loss_reduction": "Train Loss Reduction",
        "weight_norm": "Weight Norm",
        "trace_s": "Trace of S",
        "top_eigenvalue": "Top Eigenvalue",
        "min_eigenvalue": "Minimum Eigenvalue",
        "R1": "R1",
        "alpha": "Alpha",
        "n_train": r"$n_{\mathrm{train}}$",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def format_x_label(x_col: str) -> str:
    if x_col == "alpha":
        return "Alpha"
    if x_col == "n_train":
        return r"$n_{\mathrm{train}}$"
    return format_metric_name(x_col)


# =========================
# CONFIG LABEL FOR AGGREGATED PLOTS
# =========================
def build_sweep_plot_config_label(sweep_dir: Path, sweep_key: str) -> str:
    """
    Build a 2-line hyperparameter label for aggregated plots.
    Excludes the sweep variable itself from the displayed hyperparameters.
    """
    sweep_config_path = sweep_dir / "sweep_config.json"
    if not sweep_config_path.exists():
        return ""

    with open(sweep_config_path, "r", encoding="utf-8") as f:
        sweep_cfg = json.load(f)

    base_config = sweep_cfg.get("base_config", {})
    data_cfg = base_config.get("data", {})
    model_cfg = base_config.get("model", {})
    train_cfg = base_config.get("training", {})

    cov = data_cfg.get("covariance_type")
    mask = data_cfg.get("masking_strategy", "random")
    rho = format_float_for_title(data_cfg.get("rho"))
    lam = format_float_for_title(train_cfg.get("lambda_reg"))
    beta = format_float_for_title(model_cfg.get("beta"))
    T = data_cfg.get("T")
    d = data_cfg.get("d")
    lr = format_float_for_title(train_cfg.get("learning_rate"))
    n_steps = train_cfg.get("n_steps")

    line1_parts = [
        f"Cov={cov}",
        f"Mask={mask}",
        rf"$\rho={rho}$",
        rf"$\lambda={lam}$",
        rf"$\beta={beta}$",
    ]

    line2_parts = [
        rf"$d={d}$",
        rf"$T={T}$",
        f"lr={lr}",
        f"iters={n_steps}",
    ]

    if sweep_key != "n_train":
        n_train = train_cfg.get("n_train")
        if n_train is not None:
            line2_parts.append(rf"$n_{{\mathrm{{train}}}}={int(n_train)}$")

    if sweep_key != "alpha":
        alpha = train_cfg.get("alpha")
        if alpha is not None:
            line2_parts.append(rf"$\alpha={format_float_for_title(alpha)}$")

    line1 = ", ".join(line1_parts)
    line2 = ", ".join(line2_parts)

    return line1 + "\n" + line2


# =========================
# INDIVIDUAL PLOT SAVERS
# =========================
def save_metric_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    config_label: str = "",
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=x_col)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(
        plot_df[x_col],
        plot_df[y_col],
        marker="o",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel(format_x_label(x_col))
    ax.set_ylabel(format_metric_name(y_col))

    title_line = f"{format_metric_name(y_col)} vs {format_x_label(x_col)}"
    full_title = title_line if not config_label else title_line + "\n" + config_label

    set_multiline_title(ax, full_title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot_with_horizontal_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    horizontal_value_col: str | None,
    horizontal_value: float | None,
    horizontal_label: str,
    output_path: Path,
    config_label: str = "",
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    needed_cols = [x_col, y_col]
    if horizontal_value_col is not None:
        needed_cols.append(horizontal_value_col)

    plot_df = df[needed_cols].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=x_col)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        plot_df[x_col],
        plot_df[y_col],
        marker="o",
        linewidth=2,
        markersize=6,
        label=format_metric_name(y_col),
    )

    if horizontal_value_col is not None:
        ref_value = plot_df[horizontal_value_col].mean()
        ax.axhline(
            y=ref_value,
            linestyle="--",
            linewidth=2,
            color="gold",
            label=horizontal_label,
        )
    elif horizontal_value is not None:
        ax.axhline(
            y=horizontal_value,
            linestyle="--",
            linewidth=2,
            color="gold",
            label=horizontal_label,
        )

    ax.set_xlabel(format_x_label(x_col))
    ax.set_ylabel(format_metric_name(y_col))

    title_line = f"{format_metric_name(y_col)} vs {format_x_label(x_col)}"
    full_title = title_line if not config_label else title_line + "\n" + config_label

    set_multiline_title(ax, full_title)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_train_loss_with_ntrain_plot(
    df: pd.DataFrame,
    output_path: Path,
    config_label: str = "",
) -> None:
    required_cols = {"alpha", "train_loss", "n_train"}
    if not required_cols.issubset(df.columns):
        return

    plot_df = df[["alpha", "train_loss", "n_train"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by="alpha")

    fig, ax1 = plt.subplots(figsize=(10.5, 7.2))

    ax1.plot(
        plot_df["alpha"],
        plot_df["train_loss"],
        marker="o",
        linewidth=2,
        label="Train Loss",
    )
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        plot_df["alpha"],
        plot_df["n_train"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label=r"$n_{\mathrm{train}}$",
    )
    ax2.set_ylabel(r"$n_{\mathrm{train}}$")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=True)

    title_line = r"Train Loss and $n_{\mathrm{train}}$ vs Alpha"
    full_title = title_line if not config_label else title_line + "\n" + config_label

    set_multiline_title(ax1, full_title)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_train_vs_population_vs_bayes_plot(
    df: pd.DataFrame,
    x_col: str,
    output_path: Path,
    config_label: str = "",
) -> None:
    required_cols = {"train_loss", "population_risk", "bayes_population_risk", x_col}
    if not required_cols.issubset(df.columns):
        return

    plot_df = df[[x_col, "train_loss", "population_risk", "bayes_population_risk"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=x_col)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    ax.plot(
        plot_df[x_col],
        plot_df["train_loss"],
        marker="o",
        linewidth=2,
        label="Train Loss",
    )

    ax.plot(
        plot_df[x_col],
        plot_df["population_risk"],
        marker="s",
        linewidth=2,
        label="Population Risk",
    )

    bayes_level = plot_df["bayes_population_risk"].mean()

    ax.axhline(
        y=bayes_level,
        linestyle="--",
        linewidth=2,
        color="gold",
        label="Bayes Optimal Risk",
    )

    ax.set_xlabel(format_x_label(x_col))
    ax.set_ylabel("Risk")

    title_line = f"Train Loss and Population Risk vs {format_x_label(x_col)}"
    full_title = title_line if not config_label else title_line + "\n" + config_label

    set_multiline_title(ax, full_title)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# HIGH-LEVEL PLOT GENERATION
# =========================
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

    config_label = build_sweep_plot_config_label(sweep_dir, sweep_key)

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
            config_label=config_label,
        )

    save_metric_plot_with_horizontal_line(
        df=df,
        x_col=sweep_key,
        y_col="population_risk",
        horizontal_value_col="bayes_population_risk",
        horizontal_value=None,
        horizontal_label="Bayes Optimal Risk",
        output_path=plots_dir / f"population_risk_vs_{sweep_key}_with_bayes.png",
        config_label=config_label,
    )

    save_metric_plot_with_horizontal_line(
        df=df,
        x_col=sweep_key,
        y_col="train_loss",
        horizontal_value_col="bayes_population_risk",
        horizontal_value=None,
        horizontal_label="Bayes Optimal Risk",
        output_path=plots_dir / f"train_loss_vs_{sweep_key}_with_bayes.png",
        config_label=config_label,
    )

    save_metric_plot_with_horizontal_line(
        df=df,
        x_col=sweep_key,
        y_col="generalization_gap",
        horizontal_value_col=None,
        horizontal_value=0.0,
        horizontal_label="Zero Reference",
        output_path=plots_dir / f"generalization_gap_vs_{sweep_key}_with_zero.png",
        config_label=config_label,
    )

    save_metric_plot_with_horizontal_line(
        df=df,
        x_col=sweep_key,
        y_col="excess_population_risk",
        horizontal_value_col=None,
        horizontal_value=0.0,
        horizontal_label="Zero Reference",
        output_path=plots_dir / f"excess_population_risk_vs_{sweep_key}_with_zero.png",
        config_label=config_label,
    )

    save_train_vs_population_vs_bayes_plot(
        df=df,
        x_col=sweep_key,
        output_path=plots_dir / f"train_vs_population_vs_bayes_{sweep_key}.png",
        config_label=config_label,
    )

    if sweep_key == "alpha":
        save_train_loss_with_ntrain_plot(
            df=df,
            output_path=plots_dir / "train_loss_and_ntrain_vs_alpha.png",
            config_label=config_label,
        )


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
        f"population_risk_vs_{sweep_key}_with_bayes.png",
        f"train_loss_vs_{sweep_key}_with_bayes.png",
        f"generalization_gap_vs_{sweep_key}_with_zero.png",
        f"excess_population_risk_vs_{sweep_key}_with_zero.png",
        f"train_vs_population_vs_bayes_{sweep_key}.png",
    ]
    if sweep_key == "alpha":
        names.append("train_loss_and_ntrain_vs_alpha.png")
    return [plots_dir / name for name in names]