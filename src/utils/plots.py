from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# global plot style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 16,
})

def plot_observable_vs_alpha(
    alphas: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot one observable vs alpha."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(alphas, values, marker="o", linewidth=2)
    ax.set_xlabel("Alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_eigenvalues(
    eigvals: np.ndarray,
    title: str = "Eigenvalues of S",
) -> tuple[Figure, Axes]:
    """Plot sorted eigenvalues."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", linewidth=2)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_eigenvalue_histogram(
    eigvals: np.ndarray,
    bins: int = 30,
    title: str = "Histogram of eigenvalues",
) -> tuple[Figure, Axes]:
    """Histogram of eigenvalues."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(eigvals, bins=bins)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_matrix_heatmap(
    matrix: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    """Plot the matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    return fig, ax


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training history",
) -> tuple[Figure, Axes]:
    """Plot training convergence history."""
    fig, ax = plt.subplots(figsize=(9, 6))

    train_loss = history.get("train_loss", [])
    objective = history.get("objective", [])
    test_loss = history.get("test_loss", [])
    eval_steps = history.get("steps", [])

    if len(train_loss) > 0:
        ax.plot(
            range(1, len(train_loss) + 1),
            train_loss,
            label="Train loss",
            linewidth=2,
        )
    if len(objective) > 0:
        ax.plot(
            range(1, len(objective) + 1),
            objective,
            label="Objective",
            linewidth=2,
        )
    if len(test_loss) > 0 and len(eval_steps) == len(test_loss):
        ax.plot(
            eval_steps,
            test_loss,
            label="Test loss",
            linewidth=2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax