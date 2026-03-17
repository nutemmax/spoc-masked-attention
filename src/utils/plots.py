from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_observable_vs_alpha(
    alphas: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
) -> tuple[Figure, Axes]:
    
    # plot one observable vs alpha
    fig, ax = plt.subplots()
    ax.plot(alphas, values, marker="o")
    ax.set_xlabel("alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return fig, ax


def plot_eigenvalues(
    eigvals: np.ndarray,
    title: str = "Eigenvalues of S",
) -> tuple[Figure, Axes]:
    # plot sorted eigenvals
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.grid(True)
    return fig, ax


def plot_eigenvalue_histogram(
    eigvals: np.ndarray,
    bins: int = 30,
    title: str = "Histogram of eigenvalues",
) -> tuple[Figure, Axes]:
    # histogram of eigenvals
    fig, ax = plt.subplots()
    ax.hist(eigvals, bins=bins)
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True)
    return fig, ax


def plot_matrix_heatmap(
    matrix: np.ndarray,
    title: str,
) -> tuple[Figure, Axes]:
    # plot the matrix as a heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax

def plot_training_history(history: dict[str, list[float]], title: str = "Training history"):
    # plot full convergence history for train loss and objective
    fig, ax = plt.subplots(figsize=(6, 4))

    train_loss = history.get("train_loss", [])
    objective = history.get("objective", [])

    if len(train_loss) > 0:
        ax.plot(train_loss, label="Train loss")
    if len(objective) > 0:
        ax.plot(objective, label="Objective")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax