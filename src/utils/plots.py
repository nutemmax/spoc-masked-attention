from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
    "axes.grid": False,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
})


# =========================
# formatting helpers
# =========================

def set_multiline_title(ax: Axes, title: str) -> None:
    ax.set_title(title, pad=16)


def sanitize_string_for_filename(s: str | None) -> str:
    if s is None:
        return "NA"
    return str(s).strip().replace(" ", "_").replace("-", "_").lower()


def format_float_for_filename(x: float | int | None) -> str:
    if x is None:
        return "NA"

    x = float(x)

    if x.is_integer():
        return str(int(x))

    s = f"{x:.6g}"
    s = s.replace(".", "p").replace("+", "")
    return s


def format_float_for_title(x: float | int | None) -> str:
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


def format_covariance_label(covariance_type: str | None, rho: float | None, length_scale : float | None = None, eta: float | None = None) -> str:
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
            return r"$\Sigma = \mathrm{Matérn}$"
        return rf"$\Sigma = \mathrm{{Matérn}}(l={format_float_for_title(length_scale)})$"
    if cov == "exponential":
        if length_scale is None:
            return r"$\Sigma = \mathrm{Exp}$"
        return rf"$\Sigma = \mathrm{{Exp}}(l={format_float_for_title(length_scale)})$"
    if covariance_type in ["toeplitz_bump", "toeplitz_nn"] : 
        if rho is None or eta is None :
            return r"$\Sigma = \mathrm{ToeplitzNN}$"
        return rf"$\Sigma = \mathrm{{ToeplitzNN}}(\rho={format_float_for_title(rho)}, \eta = {format_float_for_title(eta)})$"
    if rho is None:
        return rf"$\Sigma = \mathrm{{{covariance_type}}}$"
    return rf"$\Sigma = \mathrm{{{covariance_type}}}(\rho={format_float_for_title(rho)})$"


def format_masking_label(masking_strategy: str | None) -> str:
    if masking_strategy is None:
        return "Mask=NA"

    mask = str(masking_strategy).replace("_", "-")
    return f"Mask={mask}"


def format_config_label(
    config: dict | None,
    actual_n_train: int | None = None,
    include_seed: bool = True,
) -> str:
    if config is None:
        return ""

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    exp_cfg = config.get("experiment", {})

    covariance_type = data_cfg.get("covariance_type", "NA")
    masking_strategy = data_cfg.get("masking_strategy", "NA")
    rho = data_cfg.get("rho", None)
    length_scale = data_cfg.get("length_scale", None)
    eta = data_cfg.get("eta", None)

    lambda_reg = training_cfg.get("lambda_reg", None)
    beta = model_cfg.get("beta", None)
    d = data_cfg.get("d", None)
    T = data_cfg.get("T", None)
    lr = training_cfg.get("learning_rate", None)
    n_steps = training_cfg.get("n_steps", None)
    seed = exp_cfg.get("seed", None)

    line1_parts = [
        format_covariance_label(covariance_type, rho, length_scale=length_scale, eta=eta),
        format_masking_label(masking_strategy),
        rf"$\lambda = {format_float_for_title(lambda_reg)}$" if lambda_reg is not None else None,
        rf"$\beta = {format_float_for_title(beta)}$" if beta is not None else None,
    ]
    line1 = ", ".join(part for part in line1_parts if part is not None)

    if actual_n_train is not None:
        size_part = rf"$n_{{\mathrm{{train}}}} = {actual_n_train}$"
    else:
        n_train_cfg = training_cfg.get("n_train", None)
        if n_train_cfg is not None:
            size_part = rf"$n_{{\mathrm{{train}}}} = {int(n_train_cfg)}$"
        else:
            alpha = training_cfg.get("alpha", None)
            size_part = rf"$\alpha = {format_float_for_title(alpha)}$"

    line2_parts = []
    if d is not None:
        line2_parts.append(rf"$d = {d}$")
    if T is not None:
        line2_parts.append(rf"$T = {T}$")
    if lr is not None:
        line2_parts.append(rf"$\eta = {format_float_for_title(lr)}$")
    if n_steps is not None:
        line2_parts.append(rf"$\mathrm{{iters}} = {n_steps}$")
    line2_parts.append(size_part)
    if include_seed and seed is not None:
        line2_parts.append(rf"$\mathrm{{seed}} = {int(seed)}$")

    line2 = ", ".join(line2_parts)

    return f"{line1}\n{line2}" if line2 else line1


def build_plot_title(metric_title: str, config: dict, actual_n_train: int | None = None) -> str:
    config_label = format_config_label(config, actual_n_train=actual_n_train, include_seed=True)
    if config_label:
        return f"{metric_title}\n{config_label}"
    return metric_title


def build_config_suffix(config: dict, actual_n_train: int | None = None) -> str:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    exp_cfg = config["experiment"]

    cov = sanitize_string_for_filename(data_cfg.get("covariance_type"))
    mask = sanitize_string_for_filename(data_cfg.get("masking_strategy"))
    rho = format_float_for_filename(data_cfg.get("rho"))
    lam = format_float_for_filename(train_cfg.get("lambda_reg"))
    beta = format_float_for_filename(model_cfg.get("beta"))
    d = int(data_cfg.get("d"))
    T = int(data_cfg.get("T"))
    lr = format_float_for_filename(train_cfg.get("learning_rate"))
    n_steps = int(train_cfg.get("n_steps"))
    seed = int(exp_cfg.get("seed"))

    parts = [
        f"cov_{cov}",
        f"mask_{mask}",
        f"rho_{rho}",
        f"lambda_{lam}",
        f"beta_{beta}",
        f"d_{d}",
        f"T_{T}",
        f"lr_{lr}",
        f"iter_{n_steps}",
    ]

    if actual_n_train is not None:
        parts.append(f"ntrain_{int(actual_n_train)}")
    else:
        n_train_cfg = train_cfg.get("n_train")
        if n_train_cfg is not None:
            parts.append(f"ntrain_{int(n_train_cfg)}")
        else:
            alpha = train_cfg.get("alpha")
            parts.append(f"alpha_{format_float_for_filename(alpha)}")

    parts.append(f"seed_{seed}")
    return "_".join(parts)


# =========================
# single-run plot functions
# =========================

def plot_observable_vs_alpha(
    alphas: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(alphas, values, marker="o", linewidth=2.2, markersize=7)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    set_multiline_title(ax, title)
    # ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_eigenvalues(
    eigvals: np.ndarray,
    title: str = r"Sorted eigenvalues of $S$",
    use_log_y: bool = False,
) -> tuple[Figure, Axes]:
    eigvals_sorted = np.sort(np.asarray(eigvals).reshape(-1))[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(
        np.arange(1, len(eigvals_sorted) + 1),
        eigvals_sorted,
        marker="o",
        linewidth=2.2,
        markersize=7,
    )
    ax.set_xlabel(r"Rank $i$")
    ax.set_ylabel(r"$\lambda_i$")
    if use_log_y:
        ax.set_yscale("log")
    set_multiline_title(ax, title)
    fig.tight_layout()
    return fig, ax


def plot_eigenvalue_histogram(
    eigvals: np.ndarray,
    bins: int = 30,
    title: str = r"Eigenvalue histogram of $S$",
    hist_range: tuple[float, float] | None = None,
    density: bool = False,
    use_log_y: bool = False,
) -> tuple[Figure, Axes]:
    vals = np.asarray(eigvals).reshape(-1).copy()

    if hist_range is not None:
        vals = vals[(vals >= hist_range[0]) & (vals <= hist_range[1])]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.hist(vals, bins=bins, density=density, edgecolor="black")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density" if density else "Count")

    if hist_range is not None:
        ax.set_xlim(*hist_range)

    if use_log_y:
        ax.set_yscale("log")

    set_multiline_title(ax, title)
    fig.tight_layout()
    return fig, ax


def plot_matrix_heatmap(matrix: np.ndarray, title: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(matrix, aspect="auto")
    set_multiline_title(ax, title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig, ax


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training convergence",
    bayes_population_risk: float | None = None,
    ridge_population_risk: float | None =None,
    pca_population_risk: float | None =None,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(11, 7.5))

    train_loss = history.get("train_loss", [])
    objective = history.get("objective", [])
    test_loss = history.get("test_loss", [])
    eval_steps = history.get("steps", [])

    if len(train_loss) > 0:
        ax.plot(
            range(1, len(train_loss) + 1),
            train_loss,
            label="Train loss",
            linewidth=2.2,
        )

    if len(objective) > 0:
        ax.plot(
            range(1, len(objective) + 1),
            objective,
            label="Objective",
            linewidth=2.2,
        )

    if len(test_loss) > 0 and len(eval_steps) == len(test_loss):
        ax.plot(
            eval_steps,
            test_loss,
            label="Population risk",
            linewidth=2.2,
        )

    if bayes_population_risk is not None:
        ax.axhline(
            y=bayes_population_risk,
            linestyle="--",
            linewidth=2.5,
            alpha = 0.9,
            color="red",
            label="Bayes optimal risk",
        )
    # Ridge line
    if ridge_population_risk is not None:
        ax.axhline(
            ridge_population_risk,
            linestyle="-.",
            linewidth=2,
            alpha = 0.8,
            color="blue",
            label="Ridge (linear baseline)",
        )

    # PCA line
    if pca_population_risk is not None:
        ax.axhline(
            pca_population_risk,
            linestyle="-.",
            linewidth=2,
            color="deepskyblue",
            label="PCA baseline",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    set_multiline_title(ax, title)
    ax.legend(frameon=True)
    # ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax