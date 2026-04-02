from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
    "axes.grid": False,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
})


def set_multiline_title(ax, title: str) -> None:
    ax.set_title(title, pad=16)


# =========================
# formatting helpers
# =========================

def format_float_for_title(x) -> str:
    if x is None:
        return "NA"

    x = float(x)

    if x == 0:
        return "0"

    if float(x).is_integer():
        return str(int(x))

    abs_x = abs(x)
    exp = round(math.log10(abs_x))

    if math.isclose(abs_x, 10 ** exp, rel_tol=1e-12, abs_tol=1e-15):
        if x < 0:
            return rf"-10^{{{exp}}}"
        return rf"10^{{{exp}}}"

    if abs_x < 1e-3 or abs_x >= 1e3:
        s = f"{x:.2e}"
        base, exponent = s.split("e")
        exponent = int(exponent)
        base = float(base)
        base_str = f"{base:.2f}".rstrip("0").rstrip(".")
        return rf"{base_str}\cdot 10^{{{exponent}}}"

    return f"{x:.4f}".rstrip("0").rstrip(".")


def format_covariance_label(
    covariance_type: str | None,
    rho: float | None,
    length_scale: float | None = None,
    eta: float | None = None,
) -> str:
    if covariance_type is None:
        return r"$\Sigma = \mathrm{NA}$"

    cov = str(covariance_type).lower()

    if cov == "identity":
        return r"$\Sigma = I$"
    if cov == "toeplitz":
        if rho is None:
            return r"$\Sigma = \mathrm{Toeplitz}$"
        return rf"$\Sigma = \mathrm{{Toeplitz}}(\rho={format_float_for_title(rho)})$"
    if cov == "tridiagonal":
        if rho is None:
            return r"$\Sigma = \mathrm{Tridiag}$"
        return rf"$\Sigma = \mathrm{{Tridiag}}(\rho={format_float_for_title(rho)})$"
    if cov == "circulant_ar1":
        if rho is None:
            return r"$\Sigma = \mathrm{CircToeplitz}$"
        return rf"$\Sigma = \mathrm{{CircToeplitz}}(\rho={format_float_for_title(rho)})$"
    if cov == "matern":
        if length_scale is None:
            return r"$\Sigma = \mathrm{Mat\acute ern}$"
        return rf"$\Sigma = \mathrm{{Mat\acute ern}}(l={format_float_for_title(length_scale)})$"
    if cov == "exponential":
        if length_scale is None:
            return r"$\Sigma = \mathrm{Exp}$"
        return rf"$\Sigma = \mathrm{{Exp}}(l={format_float_for_title(length_scale)})$"
    if cov in ["toeplitz_bump", "toeplitz_nn"]:
        if rho is None or eta is None:
            return r"$\Sigma = \mathrm{ToeplitzNN}$"
        return rf"$\Sigma = \mathrm{{ToeplitzNN}}(\rho={format_float_for_title(rho)}, \eta={format_float_for_title(eta)})$"

    if rho is None:
        return rf"$\Sigma = \mathrm{{{covariance_type}}}$"
    return rf"$\Sigma = \mathrm{{{covariance_type}}}(\rho={format_float_for_title(rho)})$"


def format_masking_label(masking_strategy: str | None) -> str:
    if masking_strategy is None:
        return "Mask=NA"
    return f"Mask={str(masking_strategy).replace('_', '-')}"


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
        "trace_s": r"$\mathrm{Tr}(S)$",
        "top_eigenvalue": "Top Eigenvalue",
        "min_eigenvalue": "Minimum Eigenvalue",
        "R1": r"$R_1$",
        "effective_rank": "Effective Rank",
        "alpha": r"$\alpha$",
        "n_train": r"$n_{\mathrm{train}}$",
        "ridge_population_risk": "Ridge Population Risk",
        "pca_population_risk": "PCA Population Risk",
        "pca_n_components": "PCA Components",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def format_x_label(x_col: str) -> str:
    if x_col == "alpha":
        return r"$\alpha$"
    if x_col == "n_train":
        return r"$n_{\mathrm{train}}$"
    return format_metric_name(x_col)


# =========================
# sweep config label
# =========================

def build_sweep_plot_config_label(sweep_dir: Path, sweep_key: str) -> str:
    sweep_config_path = sweep_dir / "sweep_config.json"
    if not sweep_config_path.exists():
        return ""

    with open(sweep_config_path, "r", encoding="utf-8") as f:
        sweep_cfg = json.load(f)

    base_config = sweep_cfg.get("base_config", {})
    data_cfg = base_config.get("data", {})
    model_cfg = base_config.get("model", {})
    train_cfg = base_config.get("training", {})

    covariance_type = data_cfg.get("covariance_type", "NA")
    masking_strategy = data_cfg.get("masking_strategy", "NA")
    rho = data_cfg.get("rho", None)
    length_scale = data_cfg.get("length_scale", None)
    eta = data_cfg.get("eta", None)

    lambda_reg = train_cfg.get("lambda_reg", None)
    beta = model_cfg.get("beta", None)
    d = data_cfg.get("d", None)
    T = data_cfg.get("T", None)
    lr = train_cfg.get("learning_rate", None)
    n_steps = train_cfg.get("n_steps", None)

    line1_parts = [
        format_covariance_label(covariance_type, rho, length_scale=length_scale, eta=eta),
        format_masking_label(masking_strategy),
        rf"$\lambda = {format_float_for_title(lambda_reg)}$" if lambda_reg is not None else None,
        rf"$\beta = {format_float_for_title(beta)}$" if beta is not None else None,
    ]
    line1 = ", ".join(part for part in line1_parts if part is not None)

    line2_parts = []
    if d is not None:
        line2_parts.append(rf"$d = {d}$")
    if T is not None:
        line2_parts.append(rf"$T = {T}$")
    if lr is not None:
        line2_parts.append(rf"$\eta = {format_float_for_title(lr)}$")
    if n_steps is not None:
        line2_parts.append(rf"$\mathrm{{iters}} = {n_steps}$")

    if sweep_key != "n_train":
        n_train = train_cfg.get("n_train")
        if n_train is not None:
            line2_parts.append(rf"$n_{{\mathrm{{train}}}} = {int(n_train)}$")

    if sweep_key != "alpha":
        alpha = train_cfg.get("alpha")
        if alpha is not None:
            line2_parts.append(rf"$\alpha = {format_float_for_title(alpha)}$")

    line2 = ", ".join(line2_parts)
    return f"{line1}\n{line2}" if line2 else line1


# =========================
# generic plot helpers
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
            color="red",
            label=horizontal_label,
        )
    elif horizontal_value is not None:
        ax.axhline(
            y=horizontal_value,
            linestyle="--",
            linewidth=2,
            color="red",
            label=horizontal_label,
        )

    ax.set_xlabel(format_x_label(x_col))
    ax.set_ylabel(format_metric_name(y_col))

    title_line = f"{format_metric_name(y_col)} vs {format_x_label(x_col)}"
    full_title = title_line if not config_label else title_line + "\n" + config_label

    set_multiline_title(ax, full_title)
    ax.legend(frameon=True)

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
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("Train Loss")

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
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels, frameon=True)

    title_line = r"Train Loss and $n_{\mathrm{train}}$ vs $\alpha$"
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
    required_cols = {x_col, "train_loss", "population_risk", "bayes_population_risk"}
    if not required_cols.issubset(df.columns):
        return

    plot_df = df[[x_col, "train_loss", "population_risk", "bayes_population_risk"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=x_col)
    bayes_level = plot_df["bayes_population_risk"].mean()

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
    ax.axhline(
        y=bayes_level,
        linestyle="--",
        linewidth=2,
        color="red",
        label="Bayes Optimal Risk",
    )

    ax.set_xlabel(format_x_label(x_col))
    ax.set_ylabel("Risk")

    title_line = f"Train Loss and Population Risk vs {format_x_label(x_col)}"
    full_title = title_line if not config_label else title_line + "\n" + config_label

    set_multiline_title(ax, full_title)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_attention_vs_ridge_vs_pca_vs_bayes_plot(
    df: pd.DataFrame,
    x_col: str,
    output_path: Path,
    config_label: str = "",
) -> None:
    required_cols = {
        x_col,
        "train_loss",
        "population_risk",
        "ridge_population_risk",
        "pca_population_risk",
        "bayes_population_risk",
    }
    if not required_cols.issubset(df.columns):
        return

    plot_df = df[
        [x_col, "train_loss", "population_risk", "ridge_population_risk", "pca_population_risk", "bayes_population_risk"]
    ].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=x_col)
    bayes_level = plot_df["bayes_population_risk"].mean()

    fig, ax = plt.subplots(figsize=(10.8, 7.4))

    ax.plot(
        plot_df[x_col],
        plot_df["train_loss"],
        marker="o",
        linewidth=2,
        label="Attention Train Loss",
    )
    ax.plot(
        plot_df[x_col],
        plot_df["population_risk"],
        marker="s",
        linewidth=2,
        label="Attention Population Risk",
    )
    ax.plot(
        plot_df[x_col],
        plot_df["ridge_population_risk"],
        marker="d",
        linewidth=2,
        linestyle="--",
        label="Ridge Population Risk",
    )
    ax.plot(
        plot_df[x_col],
        plot_df["pca_population_risk"],
        marker="^",
        linewidth=2,
        linestyle="--",
        label="PCA Population Risk",
    )
    ax.axhline(
        y=bayes_level,
        linestyle=":",
        linewidth=2,
        color="red",
        label="Bayes Optimal Risk",
    )

    ax.set_xlabel(format_x_label(x_col))
    ax.set_ylabel("Risk")

    title_line = f"Attention, Ridge, PCA, and Bayes vs {format_x_label(x_col)}"
    full_title = title_line if not config_label else title_line + "\n" + config_label
    set_multiline_title(ax, full_title)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# zoomed plots over n_train
# =========================

def get_bayes_risk(df: pd.DataFrame) -> float | None:
    if "bayes_population_risk" not in df.columns:
        return None
    vals = df["bayes_population_risk"].dropna()
    if vals.empty:
        return None
    return float(vals.iloc[0])


def prepare_ntrain_data(
    df: pd.DataFrame,
    x_min: int | None = None,
    x_max: int | None = None,
) -> pd.DataFrame:
    data = df.copy()
    data = data.dropna(subset=["n_train", "train_loss", "population_risk"])
    data["n_train"] = pd.to_numeric(data["n_train"], errors="coerce")
    data["train_loss"] = pd.to_numeric(data["train_loss"], errors="coerce")
    data["population_risk"] = pd.to_numeric(data["population_risk"], errors="coerce")
    data = data.dropna(subset=["n_train", "train_loss", "population_risk"])
    data = data.sort_values("n_train")

    if x_min is not None:
        data = data[data["n_train"] >= x_min]
    if x_max is not None:
        data = data[data["n_train"] <= x_max]

    return data


def build_zoom_plot_filename(
    x_min: int | None,
    x_max: int | None,
) -> str:
    if x_min is None and x_max is None:
        return "train_vs_population_vs_bayes_n_train_full.png"
    left = x_min if x_min is not None else 0
    right = x_max if x_max is not None else "max"
    return f"train_vs_population_vs_bayes_n_train_zoom_{left}_{right}.png"


def build_zoom_plot_ridge_pca_filename(
    x_min: int | None,
    x_max: int | None,
) -> str:
    if x_min is None and x_max is None:
        return "train_vs_population_vs_bayes_vs_ridge_vs_pca_n_train_full.png"
    left = x_min if x_min is not None else 0
    right = x_max if x_max is not None else "max"
    return f"train_vs_population_vs_bayes_vs_ridge_vs_pca_n_train_zoom_{left}_{right}.png"


def save_zoomed_train_vs_population_vs_bayes_plot(
    df: pd.DataFrame,
    output_path: Path,
    config_label: str = "",
    x_min: int | None = None,
    x_max: int | None = None,
) -> None:
    if "n_train" not in df.columns:
        return

    data = prepare_ntrain_data(df, x_min=x_min, x_max=x_max)
    if data.empty:
        return

    bayes_risk = get_bayes_risk(df)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(
        data["n_train"],
        data["train_loss"],
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Train Loss",
    )
    ax.plot(
        data["n_train"],
        data["population_risk"],
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="Population Risk",
    )

    if bayes_risk is not None:
        ax.axhline(
            y=bayes_risk,
            linestyle="--",
            linewidth=2,
            color="red",
            label="Bayes Optimal Risk",
        )

    ax.set_xlabel(r"$n_{\mathrm{train}}$")
    ax.set_ylabel("Risk")

    if x_min is None and x_max is None:
        title_line = r"Train Loss and Population Risk vs $n_{\mathrm{train}}$"
    else:
        left = x_min if x_min is not None else 0
        right = x_max if x_max is not None else "max"
        title_line = rf"Train Loss and Population Risk vs $n_{{\mathrm{{train}}}}$"

    full_title = title_line if not config_label else title_line + "\n" + config_label
    set_multiline_title(ax, full_title)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_zoomed_train_vs_population_vs_bayes_vs_ridge_vs_pca_plot(
    df: pd.DataFrame,
    output_path: Path,
    config_label: str = "",
    x_min: int | None = None,
    x_max: int | None = None,
) -> None:
    if "n_train" not in df.columns:
        return

    base_cols = ["n_train", "train_loss", "population_risk"]
    extra_cols = []
    if "ridge_population_risk" in df.columns:
        extra_cols.append("ridge_population_risk")
    if "pca_population_risk" in df.columns:
        extra_cols.append("pca_population_risk")

    data = df[base_cols + extra_cols].copy()
    data = data.dropna(subset=["n_train", "train_loss", "population_risk"])
    data["n_train"] = pd.to_numeric(data["n_train"], errors="coerce")
    data["train_loss"] = pd.to_numeric(data["train_loss"], errors="coerce")
    data["population_risk"] = pd.to_numeric(data["population_risk"], errors="coerce")

    if "ridge_population_risk" in data.columns:
        data["ridge_population_risk"] = pd.to_numeric(data["ridge_population_risk"], errors="coerce")
    if "pca_population_risk" in data.columns:
        data["pca_population_risk"] = pd.to_numeric(data["pca_population_risk"], errors="coerce")

    data = data.dropna(subset=["n_train", "train_loss", "population_risk"])
    data = data.sort_values("n_train")

    if x_min is not None:
        data = data[data["n_train"] >= x_min]
    if x_max is not None:
        data = data[data["n_train"] <= x_max]

    if data.empty:
        return

    bayes_risk = get_bayes_risk(df)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(
        data["n_train"],
        data["train_loss"],
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Attention Train Loss",
    )
    ax.plot(
        data["n_train"],
        data["population_risk"],
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="Attention Population Risk",
    )

    if "ridge_population_risk" in data.columns:
        ax.plot(
            data["n_train"],
            data["ridge_population_risk"],
            marker="d",
            linewidth=1.8,
            linestyle="--",
            markersize=6,
            label="Ridge Population Risk",
        )

    if "pca_population_risk" in data.columns:
        ax.plot(
            data["n_train"],
            data["pca_population_risk"],
            marker="^",
            linewidth=1.8,
            linestyle="--",
            markersize=6,
            label="PCA Population Risk",
        )

    if bayes_risk is not None:
        ax.axhline(
            y=bayes_risk,
            linestyle=":",
            linewidth=3,
            color="red",
            label="Bayes Optimal Risk",
        )

    ax.set_xlabel(r"$n_{\mathrm{train}}$")
    ax.set_ylabel("Risk")

    if x_min is None and x_max is None:
        title_line = r"Attention, Ridge, PCA, and Bayes vs $n_{\mathrm{train}}$"
    else:
        left = x_min if x_min is not None else 0
        right = x_max if x_max is not None else "max"
        title_line = rf"Attention, Ridge, PCA, and Bayes vs $n_{{\mathrm{{train}}}}$ (zoom: {left} to {right})"

    full_title = title_line if not config_label else title_line + "\n" + config_label
    set_multiline_title(ax, full_title)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# eigenvalue summary stats
# =========================

def effective_rank(eigvals: np.ndarray, eps: float = 1e-12) -> float:
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = eigvals[eigvals > eps]

    if eigvals.size == 0:
        return 0.0

    p = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def find_eigenvalue_file(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("eigenvalues*.npy"))
    if not matches:
        return None
    return matches[0]


def extract_n_train_from_run_name(run_name: str) -> int | None:
    parts = run_name.split("_")
    for i, token in enumerate(parts):
        if token == "ntrain" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def compute_eigenvalue_summary_stats(
    sweep_dir: Path,
) -> pd.DataFrame:
    rows = []

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "plots":
            continue

        eig_path = find_eigenvalue_file(subdir)
        if eig_path is None:
            continue

        n_train = extract_n_train_from_run_name(subdir.name)
        if n_train is None:
            continue

        try:
            eigvals = np.load(eig_path)
            eigvals = np.asarray(eigvals, dtype=float).reshape(-1)
        except Exception as e:
            print(f"[warn] Failed to load eigenvalues from {eig_path}: {e}")
            continue

        eigvals_sorted = np.sort(eigvals)[::-1]
        trace = float(np.sum(eigvals))
        top = float(eigvals_sorted[0]) if eigvals_sorted.size > 0 else np.nan
        r1 = float(top / trace) if trace > 0 else np.nan

        rows.append({
            "n_train": int(n_train),
            "num_eigs": int(eigvals.size),
            "min_eig": float(np.min(eigvals)) if eigvals.size > 0 else np.nan,
            "max_eig": float(np.max(eigvals)) if eigvals.size > 0 else np.nan,
            "mean_eig": float(np.mean(eigvals)) if eigvals.size > 0 else np.nan,
            "median_eig": float(np.median(eigvals)) if eigvals.size > 0 else np.nan,
            "std_eig": float(np.std(eigvals)) if eigvals.size > 0 else np.nan,
            "trace": trace,
            "top_eigenvalue": top,
            "R1": r1,
            "effective_rank": effective_rank(eigvals),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("n_train").reset_index(drop=True)


def save_eigenvalue_summary_csv(
    sweep_dir: Path,
    df_eigs: pd.DataFrame,
) -> None:
    if df_eigs.empty:
        return
    df_eigs.to_csv(sweep_dir / "eigenvalue_summary_stats.csv", index=False)


def save_eigenvalue_metric_vs_ntrain_plot(
    df_stats: pd.DataFrame,
    y_col: str,
    ylabel: str,
    metric_title: str,
    output_path: Path,
    config_label: str = "",
) -> None:
    if df_stats.empty or "n_train" not in df_stats.columns or y_col not in df_stats.columns:
        return

    plot_df = df_stats[["n_train", y_col]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("n_train")

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(
        plot_df["n_train"],
        plot_df[y_col],
        marker="o",
        linewidth=2.2,
        markersize=7,
    )

    ax.set_xlabel(r"$n_{\mathrm{train}}$")
    ax.set_ylabel(ylabel)

    full_title = metric_title if not config_label else metric_title + "\n" + config_label
    set_multiline_title(ax, full_title)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# high-level generation
# =========================

def generate_summary_plots(
    sweep_dir: Path,
    sweep_key: str,
    zoom_ranges: list[tuple[int | None, int | None]] | None = None,
) -> None:
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
        "ridge_population_risk",
        "pca_population_risk",
        "generalization_gap",
        "excess_population_risk",
        "weight_norm",
        "trace_s",
        "top_eigenvalue",
        "R1",
        "effective_rank",
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

    save_attention_vs_ridge_vs_pca_vs_bayes_plot(
        df=df,
        x_col=sweep_key,
        output_path=plots_dir / f"attention_vs_ridge_vs_pca_vs_bayes_{sweep_key}.png",
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

    if sweep_key == "n_train":
        if zoom_ranges is None:
            zoom_ranges = [(None, None), (0, 500), (0, 1000), (0, 2000)]

        for x_min, x_max in zoom_ranges:
            filename_basic = build_zoom_plot_filename(x_min, x_max)
            save_zoomed_train_vs_population_vs_bayes_plot(
                df=df,
                output_path=plots_dir / filename_basic,
                config_label=config_label,
                x_min=x_min,
                x_max=x_max,
            )

            filename_full = build_zoom_plot_ridge_pca_filename(x_min, x_max)
            save_zoomed_train_vs_population_vs_bayes_vs_ridge_vs_pca_plot(
                df=df,
                output_path=plots_dir / filename_full,
                config_label=config_label,
                x_min=x_min,
                x_max=x_max,
            )

        df_eigs = compute_eigenvalue_summary_stats(sweep_dir)
        if not df_eigs.empty:
            save_eigenvalue_summary_csv(sweep_dir, df_eigs)


def expected_plot_paths(
    sweep_dir: Path,
    sweep_key: str,
) -> list[Path]:
    plots_dir = sweep_dir / "plots"

    names = [
        f"train_loss_vs_{sweep_key}.png",
        f"population_risk_vs_{sweep_key}.png",
        f"ridge_population_risk_vs_{sweep_key}.png",
        f"pca_population_risk_vs_{sweep_key}.png",
        f"generalization_gap_vs_{sweep_key}.png",
        f"excess_population_risk_vs_{sweep_key}.png",
        f"weight_norm_vs_{sweep_key}.png",
        f"trace_s_vs_{sweep_key}.png",
        f"top_eigenvalue_vs_{sweep_key}.png",
        f"R1_vs_{sweep_key}.png",
        f"effective_rank_vs_{sweep_key}.png",
        f"population_risk_vs_{sweep_key}_with_bayes.png",
        f"train_loss_vs_{sweep_key}_with_bayes.png",
        f"generalization_gap_vs_{sweep_key}_with_zero.png",
        f"excess_population_risk_vs_{sweep_key}_with_zero.png",
        f"train_vs_population_vs_bayes_{sweep_key}.png",
        f"attention_vs_ridge_vs_pca_vs_bayes_{sweep_key}.png",
    ]

    if sweep_key == "alpha":
        names.append("train_loss_and_ntrain_vs_alpha.png")

    if sweep_key == "n_train":
        names.extend([
            "train_vs_population_vs_bayes_n_train_full.png",
            "train_vs_population_vs_bayes_n_train_zoom_0_500.png",
            "train_vs_population_vs_bayes_n_train_zoom_0_1000.png",
            "train_vs_population_vs_bayes_n_train_zoom_0_2000.png",
            "train_vs_population_vs_bayes_vs_ridge_vs_pca_n_train_full.png",
            "train_vs_population_vs_bayes_vs_ridge_vs_pca_n_train_zoom_0_500.png",
            "train_vs_population_vs_bayes_vs_ridge_vs_pca_n_train_zoom_0_1000.png",
            "train_vs_population_vs_bayes_vs_ridge_vs_pca_n_train_zoom_0_2000.png",
        ])

    return [plots_dir / name for name in names]