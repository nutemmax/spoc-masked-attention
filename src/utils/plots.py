from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# global plot style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 16,
})

def set_multiline_title(ax: Axes, title: str) -> None:
    """Set a title with comfortable spacing for multi-line text."""
    ax.set_title(title, pad=16)


def plot_observable_vs_alpha(alphas: np.ndarray, values: np.ndarray, ylabel: str, title: str) -> tuple[Figure, Axes]:
    """Plot one observable vs alpha."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(alphas, values, marker="o", linewidth=2)
    ax.set_xlabel("Alpha")
    ax.set_ylabel(ylabel)
    set_multiline_title(ax, title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_eigenvalues(eigvals: np.ndarray, title: str = "Eigenvalues of S") -> tuple[Figure, Axes]:
    """Plot sorted eigenvalues."""
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", linewidth=2)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    set_multiline_title(ax, title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_eigenvalue_histogram(eigvals: np.ndarray, bins: int = 30, title: str = "Histogram of eigenvalues") -> tuple[Figure, Axes]:
    """Histogram of eigenvalues."""
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.hist(eigvals, bins=bins)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    set_multiline_title(ax, title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_matrix_heatmap(matrix: np.ndarray, title: str) -> tuple[Figure, Axes]:
    """Plot the matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(matrix, aspect="auto")
    set_multiline_title(ax, title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig, ax


def plot_training_history(history: dict[str, list[float]], title: str = "Training history", bayes_population_risk: float | None = None) -> tuple[Figure, Axes]:
    """Plot training convergence history."""
    fig, ax = plt.subplots(figsize=(11, 7.5))

    train_loss = history.get("train_loss", [])
    objective = history.get("objective", [])
    test_loss = history.get("test_loss", [])
    eval_steps = history.get("steps", [])

    if len(train_loss) > 0:
        ax.plot(range(1, len(train_loss) + 1), train_loss, label="Train loss", linewidth=2)
    if len(objective) > 0:
        ax.plot(range(1, len(objective) + 1), objective, label="Objective", linewidth=2)
    if len(test_loss) > 0 and len(eval_steps) == len(test_loss):
        ax.plot(eval_steps, test_loss, label="Test loss", linewidth=2)
    if bayes_population_risk is not None:
        ax.axhline(y=bayes_population_risk,linestyle="--",linewidth=2,color="gold",label="Bayes optimal risk")
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    set_multiline_title(ax, title)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# =========== PLOT FORMATTING ===========
def format_float_for_title(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.3g}"
    return str(x)


def format_float_for_filename(x) -> str:
    if x is None:
        return "None"
    x = float(x)
    if x.is_integer():
        return str(int(x))
    s = f"{x:.6g}"
    s = s.replace(".", "p")
    s = s.replace("+", "")
    return s


def sanitize_string_for_filename(s: str) -> str:
    return str(s).strip().replace(" ", "_").replace("-", "_").lower()


def build_config_suffix(config: dict, actual_n_train: int | None = None) -> str:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    exp_cfg = config["experiment"]

    cov = sanitize_string_for_filename(data_cfg.get("covariance_type"))
    mask = sanitize_string_for_filename(data_cfg.get("masking_strategy", "random"))
    rho = format_float_for_filename(data_cfg.get("rho"))
    lam = format_float_for_filename(train_cfg.get("lambda_reg"))
    beta = format_float_for_filename(model_cfg.get("beta"))
    d = int(data_cfg.get("d"))
    T = int(data_cfg.get("T"))
    lr = format_float_for_filename(train_cfg.get("learning_rate"))
    n_steps = int(train_cfg.get("n_steps"))
    seed = int(exp_cfg.get("seed"))

    parts = [f"cov_{cov}", f"mask_{mask}", f"rho_{rho}", f"lambda_{lam}", f"beta_{beta}", f"d_{d}", f"T_{T}", f"lr_{lr}", f"iter_{n_steps}"]

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


def build_plot_config_label(config: dict, actual_n_train: int | None = None) -> str:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    exp_cfg = config["experiment"]

    cov = data_cfg.get("covariance_type")
    mask = data_cfg.get("masking_strategy", "random")
    rho = format_float_for_title(data_cfg.get("rho"))
    lam = format_float_for_title(train_cfg.get("lambda_reg"))
    beta = format_float_for_title(model_cfg.get("beta"))
    T = data_cfg.get("T")
    d = data_cfg.get("d")
    lr = format_float_for_title(train_cfg.get("learning_rate"))
    n_steps = int(train_cfg.get("n_steps"))
    seed = int(exp_cfg.get("seed"))

    if actual_n_train is not None:
        size_part = rf"$n_{{\mathrm{{train}}}}={actual_n_train}$"
    else:
        n_train_cfg = train_cfg.get("n_train")
        if n_train_cfg is not None:
            size_part = rf"$n_{{\mathrm{{train}}}}={int(n_train_cfg)}$"
        else:
            alpha = format_float_for_title(train_cfg.get("alpha"))
            size_part = rf"$\alpha={alpha}$"

    return (
        rf"Cov={cov}, Mask={mask}, $\rho={rho}$, $\lambda={lam}$, $\beta={beta}$" "\n"
        rf"$d={d}$, $T={T}$, lr={lr}, iters={n_steps}, "
        rf"{size_part}, seed={seed}"
    )