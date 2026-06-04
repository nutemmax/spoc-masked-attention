# scripts/plot_risk_analysis.py
"""
Plots exploring the relationship between cosine similarity,
train loss, population risk and generalisation gap, as a function
of n_train, d and kappa_star.

Run after aggregate_teacher_attention_sweep.py.

Usage:
    python scripts/plot_risk_analysis.py \
        --root results/collective \
        --output-dir results/analysis/risk_analysis

    # without titles:
    python scripts/plot_risk_analysis.py \
        --root results/collective \
        --output-dir results/analysis/risk_analysis \
        --no-title
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossing_utils import (
    build_title_metadata,
    format_float_for_title,
    one_or_mixed,
    get_float,
    get_int,
    has_kappa_star_ancestor,
    infer_metadata,
    read_csv_rows,
    sanitize_filename,
)

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 26,
    "axes.grid": False,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
})

ZOOM_RANGES: list[tuple[int | None, int | None]] = [
    (None, None),
    (0, 500),
    (0, 1000),
    (0, 2500),
    (0, 5000),
    (0, 10000),
]

ALPHA_ZOOM_RANGES: list[tuple[float | None, float | None]] = [
    (None, None),
    (0.0, 0.5),
    (0.0, 1.0),
    (0.0, 2.0),
    (0.0, 5.0),
    (0.0, 10.0),
]

ALPHA_LIN_ZOOM_RANGES: list[tuple[float | None, float | None]] = [
    (None, None),
    (0.0, 5.0),
    (0.0, 10.0),
    (0.0, 20.0),
    (0.0, 50.0),
    (0.0, 100.0),
]

KAPPA_COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


COLLECT_KEYS = [
    "train_loss",
    "population_risk",
    "cosine_S_S_star",
    "random_baseline_cosine_S_S_star_mean",
    "random_baseline_cosine_S_S_star",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def zoom_suffix(xlim: tuple[int | float | None, int | float | None]) -> str:
    if xlim == (None, None):
        return "full"
    left = str(xlim[0]).replace(".", "p") if xlim[0] is not None else "min"
    right = str(xlim[1]).replace(".", "p") if xlim[1] is not None else "max"
    return f"zoom_{left}_{right}"


def zoom_title_suffix(
    xlim: tuple[int | float | None, int | float | None],
    variable_latex: str,
) -> str:
    if xlim == (None, None):
        return ""

    lo, hi = xlim
    left = str(lo) if lo is not None else r"-\infty"
    right = str(hi) if hi is not None else r"+\infty"

    return rf", {variable_latex} $\in [{left},{right}]$"


def make_title(first_line: str, title_metadata: str) -> str:
    return first_line + "\n" + title_metadata


def fname(name: str, no_title: bool) -> str:
    if not no_title:
        return name
    stem, ext = name.rsplit(".", 1)
    return f"{stem}_nt.{ext}"


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def grouped_mean_std(
    rows: list[dict],
    sweep_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        x = get_float(row, sweep_key)
        y = get_float(row, metric_key)
        if x is None or y is None:
            continue
        grouped.setdefault(x, []).append(y)
    if not grouped:
        return np.array([]), np.array([]), np.array([])
    xs = np.array(sorted(grouped), dtype=float)
    means = np.array([np.mean(grouped[x]) for x in xs])
    stds = np.array([np.std(grouped[x]) for x in xs])
    return xs, means, stds


def base_config_signature(sig: str) -> str:
    parts = sig.split("__")
    kept = []
    for part in parts:
        if part.startswith("kappa="):
            continue
        if part.startswith("kappa_star="):
            continue
        kept.append(part)
    return "__".join(kept)


def title_field(
    label_latex: str,
    value: object | None,
) -> str | None:
    if value is None:
        return None
    if value == "mixed":
        return None
    return rf"${label_latex} = {format_float_for_title(value)}$"


def build_title_metadata_without_kappa(rows: list[dict]) -> str:
    T = one_or_mixed(rows, "T")
    beta_star = one_or_mixed(rows, "beta_star")
    sigma_star = one_or_mixed(rows, "sigma_star")
    lambda_reg = one_or_mixed(rows, "lambda_reg")
    learning_rate = one_or_mixed(rows, "learning_rate")
    n_steps = one_or_mixed(rows, "n_steps")

    parts = [
        title_field(r"T", T),
        title_field(r"\beta^\star", beta_star),
        title_field(r"\sigma^\star", sigma_star),
        title_field(r"\lambda", lambda_reg),
        title_field(r"\mathrm{iters}", n_steps),
    ]

    return ", ".join(part for part in parts if part is not None)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def collect_rows(root: Path) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.csv")):
        if not has_kappa_star_ancestor(summary_path):
            continue
        rows = read_csv_rows(summary_path)
        if not rows:
            continue
        metadata = infer_metadata(summary_path, rows)
        if metadata is None:
            continue

        d_value = int(metadata["d"])

        for row in rows:
            n_train = get_int(row, "n_train")
            if n_train is None:
                continue

            record: dict[str, Any] = {**metadata, "n_train": n_train}
            record["alpha_lin"] = float(n_train) / float(d_value)
            record["alpha"] = float(n_train) / float(d_value ** 2)

            for key in COLLECT_KEYS:
                record[key] = get_float(row, key)

            train = get_float(row, "train_loss")
            risk = get_float(row, "population_risk")
            record["gen_gap"] = (risk - train) if (train is not None and risk is not None) else None

            all_rows.append(record)

    return all_rows


# ---------------------------------------------------------------------
# Plot 1: stacked panels — cosine (top) + train/risk (bottom) vs n_train
# one plot per (mask, d, kappa_star), with zoom variants
# ---------------------------------------------------------------------

def plot_stacked(
    rows: list[dict],
    mask_label: str,
    d: int,
    kappa_star: float,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
) -> None:
    mask_d_rows = [r for r in rows if r["mask_label"] == mask_label and r["d"] == d]
    if not mask_d_rows:
        return

    xs_cos, means_cos, _ = grouped_mean_std(mask_d_rows, "n_train", "cosine_S_S_star")
    xs_train, means_train, _ = grouped_mean_std(mask_d_rows, "n_train", "train_loss")
    xs_risk, means_risk, _ = grouped_mean_std(mask_d_rows, "n_train", "population_risk")
    xs_base, means_base, _ = grouped_mean_std(mask_d_rows, "n_train", "random_baseline_cosine_S_S_star_mean")
    if xs_base.size == 0:
        xs_base, means_base, _ = grouped_mean_std(mask_d_rows, "n_train", "random_baseline_cosine_S_S_star")

    if xs_cos.size == 0 and xs_train.size == 0:
        return

    baseline_val = float(means_base[0]) if xs_base.size > 0 else None

    for xlim in ZOOM_RANGES:
        zs = zoom_suffix(xlim)
        lo, hi = xlim

        def crop(xs, ys):
            if lo is None and hi is None:
                return xs, ys
            mask = np.ones(len(xs), dtype=bool)
            if lo is not None:
                mask &= xs >= lo
            if hi is not None:
                mask &= xs <= hi
            return xs[mask], ys[mask]

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(11, 10), sharex=True,
            gridspec_kw={"hspace": 0.08},
            constrained_layout=True,
        )

        if xs_cos.size > 0:
            xc, yc = crop(xs_cos, means_cos)
            ax_top.plot(xc, yc, marker="o", linewidth=2, markersize=5, label=r"$\cos(S, S^\star)$")

        if baseline_val is not None:
            ax_top.axhline(
                baseline_val,
                linestyle="--",
                linewidth=1.5,
                color="black",
                alpha=0.5,
                label=r"random PSD baseline",
            )

        ax_top.set_ylabel(r"$\cos(S, S^\star)$")
        ax_top.legend(frameon=True)

        if xs_train.size > 0:
            xt, yt = crop(xs_train, means_train)
            ax_bot.plot(xt, yt, marker="o", linewidth=2, markersize=5, label="train loss")

        if xs_risk.size > 0:
            xr, yr = crop(xs_risk, means_risk)
            ax_bot.plot(
                xr,
                yr,
                marker="s",
                linewidth=2,
                markersize=5,
                linestyle="--",
                label="population risk",
            )

        ax_bot.set_xlabel(r"$n_{\mathrm{train}}$")
        ax_bot.set_ylabel("Train loss / Population risk")
        ax_bot.legend(frameon=True)

        if not no_title:
            zoom_txt = zoom_title_suffix(xlim, r"$n_{\mathrm{train}}$")
            ax_top.set_title(
                make_title(
                    rf"Cosine similarity and losses: {mask_label}, $d={d}${zoom_txt}",
                    title_metadata,
                )
            )

        kstr = str(kappa_star).replace(".", "p")
        save_fig(
            fig,
            output_dir / fname(
                f"stacked_{mask_label}_d{d}_kappa{kstr}_{zs}.png",
                no_title,
            ),
        )


# ---------------------------------------------------------------------
# Parametric trajectory helper
# ---------------------------------------------------------------------

def _parametric_trajectory(
    rows: list[dict],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_train, x_means, y_means) sorted by n_train."""
    xs, xm, _ = grouped_mean_std(rows, "n_train", x_key)
    ys, ym, _ = grouped_mean_std(rows, "n_train", y_key)
    n_shared = sorted(set(xs.tolist()) & set(ys.tolist()))
    if not n_shared:
        return np.array([]), np.array([]), np.array([])
    x_map = dict(zip(xs.tolist(), xm.tolist()))
    y_map = dict(zip(ys.tolist(), ym.tolist()))
    ns = np.array(n_shared)
    xv = np.array([x_map[n] for n in n_shared])
    yv = np.array([y_map[n] for n in n_shared])
    return ns, xv, yv


# ---------------------------------------------------------------------
# Plot B: overlay of parametric curves, one curve per d
# fixed (kappa_star, mask)
# ---------------------------------------------------------------------

def plot_scatter_by_d(
    rows: list[dict],
    mask_label: str,
    kappa_star: float,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
) -> None:
    """Plot B: (cosine, metric) trajectories, one curve per d,
    fixed (kappa_star, mask)."""
    mask_rows = [r for r in rows if r["mask_label"] == mask_label]
    if not mask_rows:
        return

    ds = sorted({r["d"] for r in mask_rows})
    cmap = plt.cm.viridis
    colours = {d: cmap(i / max(len(ds) - 1, 1)) for i, d in enumerate(ds)}
    kstr = str(kappa_star).replace(".", "p")

    for y_key, y_label, plot_tag in [
        ("population_risk", "population risk", "pop_risk"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 7.5))
        any_plotted = False

        for d in ds:
            d_rows = [r for r in mask_rows if r["d"] == d]
            ns, xv, yv = _parametric_trajectory(d_rows, "cosine_S_S_star", y_key)
            if len(ns) == 0:
                continue

            ax.plot(
                xv,
                yv,
                color=colours[d],
                linewidth=1.8,
                marker="o",
                markersize=4,
                alpha=0.85,
                label=rf"$d={d}$",
            )
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(r"$\cos(S, S^\star)$")
        ax.set_ylabel(y_label)
        ax.legend(frameon=True, ncol=2)

        if not no_title:
            ax.set_title(
                make_title(
                    rf"{y_label} vs cosine similarity, curves by $d$: {mask_label}",
                    title_metadata,
                )
            )

        save_fig(
            fig,
            output_dir / fname(
                f"scatter_by_d_{plot_tag}_vs_cosine_{mask_label}_kappa{kstr}.png",
                no_title,
            ),
        )


# ---------------------------------------------------------------------
# Plot C: overlay of parametric curves, one curve per kappa_star
# fixed (d, mask)
# ---------------------------------------------------------------------

def plot_scatter_by_kappa(
    rows: list[dict],
    mask_label: str,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
) -> None:
    """Plot C: (cosine, metric) trajectories, one curve per kappa_star,
    fixed (d, mask)."""
    mask_rows = [r for r in rows if r["mask_label"] == mask_label]
    if not mask_rows:
        return

    ds = sorted({r["d"] for r in mask_rows})
    kappas = sorted({r["kappa_star"] for r in mask_rows})
    colours = {
        k: KAPPA_COLOURS[i % len(KAPPA_COLOURS)]
        for i, k in enumerate(kappas)
    }

    for d in ds:
        d_rows = [r for r in mask_rows if r["d"] == d]

        for y_key, y_label, plot_tag in [
            ("population_risk", "population risk", "pop_risk"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 7.5))
            any_plotted = False

            for kstar in kappas:
                k_rows = [r for r in d_rows if r["kappa_star"] == kstar]
                ns, xv, yv = _parametric_trajectory(k_rows, "cosine_S_S_star", y_key)
                if len(ns) == 0:
                    continue

                ax.plot(
                    xv,
                    yv,
                    color=colours[kstar],
                    linewidth=1.8,
                    marker="o",
                    markersize=4,
                    alpha=0.85,
                    label=rf"$\kappa^\star={kstar}$",
                )
                any_plotted = True

            if not any_plotted:
                plt.close(fig)
                continue

            ax.set_xlabel(r"$\cos(S, S^\star)$")
            ax.set_ylabel(y_label)
            ax.legend(frameon=True)

            if not no_title:
                ax.set_title(
                    make_title(
                        rf"{y_label} vs cosine similarity, curves by $\kappa^\star$: {mask_label}, $d={d}$",
                        title_metadata,
                    )
                )

            save_fig(
                fig,
                output_dir / fname(
                    f"scatter_by_kappa_{plot_tag}_vs_cosine_{mask_label}_d{d}.png",
                    no_title,
                ),
            )


# ---------------------------------------------------------------------
# Plot: cosine similarity and population risk vs alpha = n_train / d^2
# side-by-side panels, curves by d
# one plot per (mask, kappa_star), with alpha zoom variants
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Plot: cosine similarity and population risk vs chosen x-axis
# side-by-side panels, curves by d
# one plot per (mask, kappa_star), with zoom variants
# ---------------------------------------------------------------------

def plot_cosine_and_risk_vs_x_by_d(
    rows: list[dict],
    mask_label: str,
    kappa_star: float,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
    x_key: str,
    x_label: str,
    x_title: str,
    filename_prefix: str,
    xlim_ranges: list[tuple[int | float | None, int | float | None]],
) -> None:
    mask_rows = [r for r in rows if r["mask_label"] == mask_label]
    if not mask_rows:
        return

    ds = sorted({int(r["d"]) for r in mask_rows})
    kstr = str(kappa_star).replace(".", "p")

    default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colours = {
        d: default_colours[i % len(default_colours)]
        for i, d in enumerate(ds)
    }

    for xlim in xlim_ranges:
        zs = zoom_suffix(xlim)
        lo, hi = xlim

        fig, (ax_cos, ax_risk) = plt.subplots(
            1,
            2,
            figsize=(16, 6.5),
            sharex=True,
            constrained_layout=True,
        )

        any_plotted = False

        def crop(
            xs: np.ndarray,
            means: np.ndarray,
            stds: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if xs.size == 0:
                return xs, means, stds

            mask_arr = np.ones(len(xs), dtype=bool)
            if lo is not None:
                mask_arr &= xs >= lo
            if hi is not None:
                mask_arr &= xs <= hi

            return xs[mask_arr], means[mask_arr], stds[mask_arr]

        for d in ds:
            d_rows = [r for r in mask_rows if int(r["d"]) == d]
            colour = colours[d]

            xs_cos, means_cos, stds_cos = grouped_mean_std(
                d_rows,
                x_key,
                "cosine_S_S_star",
            )
            xs_risk, means_risk, stds_risk = grouped_mean_std(
                d_rows,
                x_key,
                "population_risk",
            )

            xs_cos, means_cos, stds_cos = crop(xs_cos, means_cos, stds_cos)
            xs_risk, means_risk, stds_risk = crop(xs_risk, means_risk, stds_risk)

            if xs_cos.size > 0:
                ax_cos.plot(
                    xs_cos,
                    means_cos,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    color=colour,
                    label=rf"$d={d}$",
                )
                if np.any(stds_cos > 0):
                    ax_cos.fill_between(
                        xs_cos,
                        means_cos - stds_cos,
                        means_cos + stds_cos,
                        color=colour,
                        alpha=0.12,
                    )
                any_plotted = True

            if xs_risk.size > 0:
                ax_risk.plot(
                    xs_risk,
                    means_risk,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    color=colour,
                    label=rf"$d={d}$",
                )
                if np.any(stds_risk > 0):
                    ax_risk.fill_between(
                        xs_risk,
                        means_risk - stds_risk,
                        means_risk + stds_risk,
                        color=colour,
                        alpha=0.12,
                    )
                any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax_cos.set_xlabel(x_label)
        ax_risk.set_xlabel(x_label)

        ax_cos.set_ylabel(r"$\cos(S,S^\star)$")
        ax_risk.set_ylabel("population risk")

        ax_cos.legend(frameon=True, ncol=2)
        ax_risk.legend(frameon=True, ncol=2)

        if not no_title:
            zoom_txt = zoom_title_suffix(xlim, x_title)
            fig.suptitle(
                make_title(
                    rf"Cosine similarity and population risk vs {x_title}: "
                    rf"{mask_label}, $\kappa^\star={kappa_star:g}${zoom_txt}",
                    title_metadata,
                )
            )

        save_fig(
            fig,
            output_dir / fname(
                f"{filename_prefix}_{mask_label}_kappa{kstr}_{zs}.png",
                no_title,
            ),
        )


# ---------------------------------------------------------------------
# Plot 3: population risk and train loss vs n_train, curves by d
# one plot per (mask, kappa_star), with zoom variants
# ---------------------------------------------------------------------

def plot_risk_by_d(
    rows: list[dict],
    mask_label: str,
    kappa_star: float,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
) -> None:
    mask_rows = [r for r in rows if r["mask_label"] == mask_label]
    if not mask_rows:
        return

    ds = sorted({r["d"] for r in mask_rows})
    kstr = str(kappa_star).replace(".", "p")

    for metric_key, y_label, plot_tag in [
        ("population_risk", "population risk", "pop_risk"),
        ("train_loss", "train loss", "train_loss"),
    ]:
        for xlim in ZOOM_RANGES:
            zs = zoom_suffix(xlim)
            lo, hi = xlim

            fig, ax = plt.subplots(figsize=(11, 7.5))
            any_plotted = False

            for d in ds:
                d_rows = [r for r in mask_rows if r["d"] == d]
                xs, means, stds = grouped_mean_std(d_rows, "n_train", metric_key)
                if xs.size == 0:
                    continue

                mask_arr = np.ones(len(xs), dtype=bool)
                if lo is not None:
                    mask_arr &= xs >= lo
                if hi is not None:
                    mask_arr &= xs <= hi

                xs, means, stds = xs[mask_arr], means[mask_arr], stds[mask_arr]
                if len(xs) == 0:
                    continue

                ax.plot(xs, means, marker="o", linewidth=2, markersize=5, label=rf"$d={d}$")

                if np.any(stds > 0):
                    ax.fill_between(xs, means - stds, means + stds, alpha=0.12)

                any_plotted = True

            if not any_plotted:
                plt.close(fig)
                continue

            ax.set_xlabel(r"$n_{\mathrm{train}}$")
            ax.set_ylabel(y_label)
            ax.legend(frameon=True, ncol=2)

            if not no_title:
                zoom_txt = zoom_title_suffix(xlim, r"$n_{\mathrm{train}}$")
                ax.set_title(
                    make_title(
                        rf"{y_label} vs $n_{{\mathrm{{train}}}}$: {mask_label}{zoom_txt}",
                        title_metadata,
                    )
                )

            save_fig(
                fig,
                output_dir / fname(
                    f"{plot_tag}_by_d_{mask_label}_kappa{kstr}_{zs}.png",
                    no_title,
                ),
            )


# ---------------------------------------------------------------------
# Plot 4: generalisation gap vs n_train, curves by d
# one plot per (mask, kappa_star), with zoom variants
# ---------------------------------------------------------------------

def plot_gen_gap_by_d(
    rows: list[dict],
    mask_label: str,
    kappa_star: float,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
) -> None:
    mask_rows = [r for r in rows if r["mask_label"] == mask_label]
    if not mask_rows:
        return

    ds = sorted({r["d"] for r in mask_rows})
    kstr = str(kappa_star).replace(".", "p")

    for xlim in ZOOM_RANGES:
        zs = zoom_suffix(xlim)
        lo, hi = xlim

        fig, ax = plt.subplots(figsize=(11, 7.5))
        any_plotted = False

        for d in ds:
            d_rows = [r for r in mask_rows if r["d"] == d]
            xs, means, stds = grouped_mean_std(d_rows, "n_train", "gen_gap")
            if xs.size == 0:
                continue

            mask_arr = np.ones(len(xs), dtype=bool)
            if lo is not None:
                mask_arr &= xs >= lo
            if hi is not None:
                mask_arr &= xs <= hi

            xs, means, stds = xs[mask_arr], means[mask_arr], stds[mask_arr]
            if len(xs) == 0:
                continue

            ax.plot(xs, means, marker="o", linewidth=2, markersize=5, label=rf"$d={d}$")

            if np.any(stds > 0):
                ax.fill_between(xs, means - stds, means + stds, alpha=0.12)

            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.axhline(0, color="black", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"$n_{\mathrm{train}}$")
        ax.set_ylabel("GG = (population risk $-$ train loss)")
        ax.legend(frameon=True, ncol=2)

        if not no_title:
            zoom_txt = zoom_title_suffix(xlim, r"$n_{\mathrm{train}}$")
            ax.set_title(
                make_title(
                    rf"Generalisation gap vs $n_{{\mathrm{{train}}}}$: {mask_label}{zoom_txt}",
                    title_metadata,
                )
            )

        save_fig(
            fig,
            output_dir / fname(
                f"gen_gap_by_d_{mask_label}_kappa{kstr}_{zs}.png",
                no_title,
            ),
        )


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run(root: Path, output_dir: Path, no_title: bool) -> None:
    print("loading data...")
    rows = collect_rows(root)
    print(f"loaded {len(rows)} records")

    if not rows:
        print("no data found — did you run aggregate_teacher_attention_sweep.py first?")
        return

    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        key = (str(r["config_signature"]), float(r["kappa_star"]))
        groups[key].append(r)

    for (sig, kappa_star), group_rows in sorted(groups.items()):
        title_meta = build_title_metadata(group_rows)
        sig_dir = output_dir / sanitize_filename(sig) / f"kappa_star_{str(kappa_star).replace('.', 'p')}"
        sig_dir.mkdir(parents=True, exist_ok=True)

        masks = sorted({str(r["mask_label"]) for r in group_rows})
        ds = sorted({int(r["d"]) for r in group_rows})
        
        over_ntrain_dir = sig_dir / "over_ntrain"
        over_alpha_dir = sig_dir / "over_alpha"

        for mask_label in masks:
            for d in ds:
                plot_stacked(
                    rows=group_rows,
                    mask_label=mask_label,
                    d=d,
                    kappa_star=kappa_star,
                    title_metadata=title_meta,
                    output_dir=over_ntrain_dir / "stacked",
                    no_title=no_title,
                )

            plot_scatter_by_d(
                rows=group_rows,
                mask_label=mask_label,
                kappa_star=kappa_star,
                title_metadata=title_meta,
                output_dir=sig_dir / "scatter_by_d",
                no_title=no_title,
            )

            plot_cosine_and_risk_vs_x_by_d(
                rows=group_rows,
                mask_label=mask_label,
                kappa_star=kappa_star,
                title_metadata=title_meta,
                output_dir=over_ntrain_dir / "cosine_and_risk_vs_ntrain",
                no_title=no_title,
                x_key="n_train",
                x_label=r"$n_{\mathrm{train}}$",
                x_title=r"$n_{\mathrm{train}}$",
                filename_prefix="cosine_and_risk_vs_ntrain",
                xlim_ranges=ZOOM_RANGES,
            )

            plot_cosine_and_risk_vs_x_by_d(
                rows=group_rows,
                mask_label=mask_label,
                kappa_star=kappa_star,
                title_metadata=title_meta,
                output_dir=sig_dir / "over_alpha_lin" / "cosine_and_risk_vs_alpha_lin",
                no_title=no_title,
                x_key="alpha_lin",
                x_label=r"$\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$",
                x_title=r"$\alpha_{\mathrm{lin}}$",
                filename_prefix="cosine_and_risk_vs_alpha_lin",
                xlim_ranges=ALPHA_LIN_ZOOM_RANGES,
            )

            plot_cosine_and_risk_vs_x_by_d(
                rows=group_rows,
                mask_label=mask_label,
                kappa_star=kappa_star,
                title_metadata=title_meta,
                output_dir=over_alpha_dir / "cosine_and_risk_vs_alpha",
                no_title=no_title,
                x_key="alpha",
                x_label=r"$\alpha = n_{\mathrm{train}}/d^2$",
                x_title=r"$\alpha$",
                filename_prefix="cosine_and_risk_vs_alpha",
                xlim_ranges=ALPHA_ZOOM_RANGES,
            )

            plot_risk_by_d(
                rows=group_rows,
                mask_label=mask_label,
                kappa_star=kappa_star,
                title_metadata=title_meta,
                output_dir=over_ntrain_dir / "risk_by_d",
                no_title=no_title,
            )

            plot_gen_gap_by_d(
                rows=group_rows,
                mask_label=mask_label,
                kappa_star=kappa_star,
                title_metadata=title_meta,
                output_dir=over_ntrain_dir / "gen_gap",
                no_title=no_title,
            )

            print(f"  mask={mask_label}, kappa_star={kappa_star} done")


    sig_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        base_sig = base_config_signature(str(r["config_signature"]))
        sig_groups[base_sig].append(r)

    for base_sig, sig_rows in sorted(sig_groups.items()):
        title_meta = build_title_metadata_without_kappa(sig_rows)
        sig_dir = output_dir / sanitize_filename(base_sig)

        for mask_label in sorted({str(r["mask_label"]) for r in sig_rows}):
            plot_scatter_by_kappa(
                rows=sig_rows,
                mask_label=mask_label,
                title_metadata=title_meta,
                output_dir=sig_dir / "scatter_by_kappa",
                no_title=no_title,
            )

    print(f"done -> {output_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing kappa_star_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <root>/analysis/risk_analysis.",
    )
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else root / "analysis" / "risk_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run(root, output_dir, args.no_title)


if __name__ == "__main__":
    main()