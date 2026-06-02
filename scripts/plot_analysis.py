# scripts/plot_analysis.py
"""
Single entry point for all analysis plots.
Run after aggregate_teacher_attention_sweep.py has been run on your results.

Usage:
    python scripts/plot_analysis.py \
        --root results/collective \
        --output-dir results/analysis

    # without titles (e.g. for the report):
    python scripts/plot_analysis.py \
        --root results/collective \
        --output-dir results/analysis \
        --no-title
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# make sure scripts/ is on the path so crossing_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossing_utils import (
    build_kappa_comparison_title,
    build_title_metadata,
    crossing_columns,
    find_config_folder_name,
    first_crossing_vs_baseline,
    first_crossing_vs_constant,
    format_float_for_title,
    get_float,
    get_int,
    grouped_mean_by_ntrain,
    has_kappa_star_ancestor,
    infer_metadata,
    one_or_mixed,
    read_csv_rows,
    sanitize_filename,
    write_csv,
)

# ---------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
    "figure.titlesize": 30,
    "axes.grid": False,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
})

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
RANDOM_PSD_COLOR = "red"
KAPPA_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

# keys read from summary.csv for crossing analysis
CROSSING_KEYS = [
    "cosine_S_S_star",
    "random_baseline_cosine_S_S_star",
    "random_baseline_cosine_S_S_star_mean",
]

# keys read from summary.csv for risk vs cosine analysis
RISK_KEYS = [
    "cosine_S_S_star",
    "population_risk",
    "train_loss",
    "random_baseline_cosine_S_S_star_mean",
    "random_baseline_cosine_S_S_star",
]


# ---------------------------------------------------------------------
# Filename helper
# ---------------------------------------------------------------------

def fname(name: str, no_title: bool) -> str:
    """Append _nt before extension if no_title is set."""
    if not no_title:
        return name
    stem, ext = name.rsplit(".", 1)
    return f"{stem}_nt.{ext}"


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

# ---------------------------------------------------------------------
# Shared low-level plot helpers
# ---------------------------------------------------------------------

def plot_curve_on_ax(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    stds: np.ndarray | None,
    label: str,
    marker: str = "o",
    color: str | None = None,
    alpha: float = 0.9,
    linewidth: float = 2.5,
    markersize: float = 7,
) -> None:
    kwargs: dict[str, Any] = dict(
        marker=marker, linewidth=linewidth, markersize=markersize,
        label=label, alpha=alpha,
    )
    if color:
        kwargs["color"] = color
    ax.plot(xs, ys, **kwargs)
    if stds is not None and np.any(stds > 0):
        fill_kw: dict[str, Any] = dict(alpha=0.12)
        if color:
            fill_kw["color"] = color
        ax.fill_between(xs, ys - stds, ys + stds, **fill_kw)


def save_fig(fig, path: Path, tight: bool = True) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight" if tight else None)
    plt.close(fig)


def _plot_crossing_curve(
    rows: list[dict[str, Any]],
    x_key: str,
    mean_key: str,
    std_key: str,
    label: str,
    ax,
    marker: str = "o",
) -> bool:
    xs: list[float] = []
    ys: list[float] = []
    stds: list[float] = []
    for row in sorted(rows, key=lambda r: float(r.get(x_key, float("nan")))):
        x = row.get(x_key)
        y = row.get(mean_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        std = row.get(std_key)
        stds.append(float(std) if std is not None else 0.0)

    if not xs:
        return False

    plot_curve_on_ax(
        ax,
        np.array(xs, dtype=float),
        np.array(ys, dtype=float),
        np.array(stds, dtype=float),
        label=label,
        marker=marker,
    )
    return True


# ---------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------

def collect_rows(root: Path, extra_keys: list[str]) -> list[dict[str, Any]]:
    """
    Walk all summary.csv files under root, infer metadata from config,
    and return one dict per (summary_path, n_train) row.
    Only summary.csv files under a kappa_star_* ancestor are included.
    """
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

        for row in rows:
            n_train = get_int(row, "n_train")
            if n_train is None:
                continue

            record: dict[str, Any] = {**metadata, "n_train": n_train}
            for key in extra_keys:
                record[key] = get_float(row, key)
            all_rows.append(record)

    return all_rows


# =====================================================================
# SECTION 1: Cosine curves by d
# =====================================================================

def _aggregate_cosine_by_d(
    rows: list[dict[str, Any]],
) -> dict[int, list[tuple[int, float, float]]]:
    """d -> [(n_train, mean_cosine, std_cosine)]"""
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("cosine_S_S_star") is not None:
            grouped[row["d"]][row["n_train"]].append(row["cosine_S_S_star"])
    return {
        d: [
            (n, float(np.mean(vs)), float(np.std(vs)))
            for n, vs in sorted(by_n.items())
        ]
        for d, by_n in grouped.items()
    }


def _plot_cosine_by_d(
    grouped: dict[int, list[tuple[int, float, float]]],
    ax,
    x_scale: float = 1.0,
) -> None:
    for d in sorted(grouped):
        pts = grouped[d]
        xs = np.array([p[0] / x_scale for p in pts]) if x_scale != 1.0 else np.array([p[0] for p in pts], dtype=float)
        means = np.array([p[1] for p in pts])
        stds = np.array([p[2] for p in pts])
        plot_curve_on_ax(ax, xs, means, stds, label=rf"$d={d}$")


def plot_cosine_by_d_for_mask(
    rows: list[dict[str, Any]],
    mask_label: str,
    kappa_star: float,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
) -> None:
    mask_rows = [r for r in rows if r.get("mask_label") == mask_label]
    if not mask_rows:
        return

    grouped = _aggregate_cosine_by_d(mask_rows)
    if not grouped:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = kappa_star / (1.0 + kappa_star)

    x_configs = [
        (1.0,       r"$n_{\mathrm{train}}$",       "ntrain"),
        (None,      r"$n_{\mathrm{train}}/d$",      "ntrain_over_d"),
        (None,      r"$n_{\mathrm{train}}/d^2$",    "ntrain_over_d2"),
    ]

    for show_baseline in [False, True]:
        suffix = "__with_random_psd_baseline" if show_baseline else ""

        # raw n_train
        fig, ax = plt.subplots(figsize=(14, 10))
        _plot_cosine_by_d(grouped, ax, x_scale=1.0)
        if show_baseline:
            ax.axhline(baseline, color=RANDOM_PSD_COLOR, linestyle="dashed",
                       linewidth=2.8, alpha=0.7, label=rf"$\kappa^\star/(1+\kappa^\star)$")
        ax.set_xlabel(r"$n_{\mathrm{train}}$")
        ax.set_ylabel(r"$\cos(S,S^\star)$")
        if not no_title:
            ax.set_title(f"Cosine similarity vs $n_{{\\mathrm{{train}}}}$: {mask_label}{', with random PSD baseline' if show_baseline else ''}\n{title_metadata}")
        ax.legend(frameon=True, ncol=2)
        save_fig(fig, output_dir / fname(f"cosine_vs_ntrain_by_d__{mask_label}{suffix}.png", no_title))

        # n_train / d
        fig, ax = plt.subplots(figsize=(14, 10))
        for d in sorted(grouped):
            pts = grouped[d]
            xs = np.array([p[0] / d for p in pts])
            means = np.array([p[1] for p in pts])
            stds = np.array([p[2] for p in pts])
            plot_curve_on_ax(ax, xs, means, stds, label=rf"$d={d}$")
        if show_baseline:
            ax.axhline(baseline, color=RANDOM_PSD_COLOR, linestyle="dashed",
                       linewidth=2.8, alpha=0.7, label=rf"$\kappa^\star/(1+\kappa^\star)$")
        ax.set_xlabel(r"$n_{\mathrm{train}}/d$")
        ax.set_ylabel(r"$\cos(S,S^\star)$")
        if not no_title:
            ax.set_title(f"Cosine similarity vs $n_{{\\mathrm{{train}}}}/d$: {mask_label}{', with random PSD baseline' if show_baseline else ''}\n{title_metadata}")
        ax.legend(frameon=True, ncol=2)
        save_fig(fig, output_dir / fname(f"cosine_vs_ntrain_over_d_by_d__{mask_label}{suffix}.png", no_title))

        # n_train / d^2 with optional zoom
        for xlim, zoom_tag in [(None, ""), ((0, 10), "__zoom0_10"), ((0, 20), "__zoom0_20")]:
            fig, ax = plt.subplots(figsize=(14, 10))
            for d in sorted(grouped):
                pts = grouped[d]
                xs = np.array([p[0] / d**2 for p in pts])
                means = np.array([p[1] for p in pts])
                stds = np.array([p[2] for p in pts])
                if xlim:
                    keep = (xs >= xlim[0]) & (xs <= xlim[1])
                    xs, means, stds = xs[keep], means[keep], stds[keep]
                plot_curve_on_ax(ax, xs, means, stds, label=rf"$d={d}$")
            if show_baseline:
                ax.axhline(baseline, color=RANDOM_PSD_COLOR, linestyle="dashed",
                           linewidth=2.8, alpha=0.7, label=rf"$\kappa^\star/(1+\kappa^\star)$")
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_xlabel(r"$n_{\mathrm{train}}/d^2$")
            ax.set_ylabel(r"$\cos(S,S^\star)$")
            if not no_title:
                ax.set_title(f"Cosine similarity vs $n_{{\\mathrm{{train}}}}/d^2$: {mask_label}{', with random PSD baseline' if show_baseline else ''}\n{title_metadata}")
            ax.legend(frameon=True, ncol=2)
            save_fig(fig, output_dir / fname(f"cosine_vs_ntrain_over_d2_by_d__{mask_label}{suffix}{zoom_tag}.png", no_title))


def run_cosine_by_d(
    rows: list[dict[str, Any]],
    output_dir: Path,
    no_title: bool,
) -> None:
    print("[cosine_by_d] running...")
    out = output_dir / "cosine_by_d_curves"

    # group by (config_signature, kappa_star) so each group has one kappa
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        key = (str(r["config_signature"]), float(r["kappa_star"]))
        groups[key].append(r)

    for (sig, kappa_star), group_rows in sorted(groups.items()):
        title_meta = build_title_metadata(group_rows)
        sig_dir = out / sanitize_filename(sig) / f"kappa_star_{str(kappa_star).replace('.', 'p')}"

        for mask_label in sorted({str(r["mask_label"]) for r in group_rows}):
            plot_cosine_by_d_for_mask(
                rows=group_rows,
                mask_label=mask_label,
                kappa_star=kappa_star,
                title_metadata=title_meta,
                output_dir=sig_dir,
                no_title=no_title,
            )

    print(f"[cosine_by_d] done -> {out}")


# =====================================================================
# SECTION 2: Mean cosine over d by kappa_star
# =====================================================================

def _aggregate_cosine_mean_over_d(
    rows: list[dict[str, Any]],
    x_normalization: str,
) -> dict[float, list[tuple[float, float, float, int]]]:
    """kappa_star -> [(x, mean, std, count)]"""
    bucket: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        kstar = row.get("kappa_star")
        cosine = row.get("cosine_S_S_star")
        if kstar is None or cosine is None or float(kstar) <= 0:
            continue
        d = row["d"]
        n = row["n_train"]
        x = n / d**2 if x_normalization == "d2" else n / (float(kstar) * d**2)
        bucket[float(kstar)][x].append(cosine)

    return {
        kstar: [
            (x, float(np.mean(vs)), float(np.std(vs)), len(vs))
            for x, vs in sorted(by_x.items())
        ]
        for kstar, by_x in sorted(bucket.items())
    }


def plot_cosine_mean_for_mask(
    rows: list[dict[str, Any]],
    mask_label: str,
    title_metadata: str,
    output_dir: Path,
    no_title: bool,
) -> None:
    mask_rows = [r for r in rows if r.get("mask_label") == mask_label]
    if not mask_rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for x_norm in ["d2", "kappa_d2"]:
        aggregated = _aggregate_cosine_mean_over_d(mask_rows, x_norm)
        if not aggregated:
            continue

        xlabel = (r"$n_{\mathrm{train}} / d^2$" if x_norm == "d2"
                  else r"$n_{\mathrm{train}} / (\kappa^\star d^2)$")
        suffix_base = "" if x_norm == "d2" else "__x_kappa_d2"
        kappas = sorted(aggregated)
        colours = KAPPA_COLOURS[:len(kappas)]

        for xlim, zoom_tag in [(None, ""), ((0, 10), "__zoom0_10"), ((0, 20), "__zoom0_20")]:
            fig, ax = plt.subplots(figsize=(14, 10))
            for kstar, colour in zip(kappas, colours):
                pts = aggregated[kstar]
                xs = np.array([p[0] for p in pts])
                means = np.array([p[1] for p in pts])
                stds = np.array([p[2] for p in pts])
                counts = np.array([p[3] for p in pts])
                if xlim:
                    keep = (xs >= xlim[0]) & (xs <= xlim[1])
                    xs, means, stds, counts = xs[keep], means[keep], stds[keep], counts[keep]
                if len(xs) == 0:
                    continue
                plot_curve_on_ax(ax, xs, means,
                                 stds * (counts > 1),
                                 label=rf"$\kappa^\star = {kstar:g}$",
                                 color=colour)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\cos(S, S^\star)$")
            if not no_title:
                ax.set_title(f"Mean cosine similarity over $d$: {mask_label}\n{title_metadata}")
            if xlim:
                ax.set_xlim(*xlim)
            ax.legend(frameon=True)
            save_fig(fig, output_dir / fname(
                f"cosine_mean_over_d_by_kappa__{mask_label}{suffix_base}{zoom_tag}.png", no_title))


def run_cosine_mean_by_kappa(
    rows: list[dict[str, Any]],
    output_dir: Path,
    no_title: bool,
) -> None:
    print("[cosine_mean_by_kappa] running...")
    out = output_dir / "cosine_mean_by_kappa"
    out.mkdir(parents=True, exist_ok=True)
    title_meta = build_kappa_comparison_title(rows)
    for mask_label in sorted({str(r["mask_label"]) for r in rows}):
        plot_cosine_mean_for_mask(rows, mask_label, title_meta, out, no_title)
        print(f"[cosine_mean_by_kappa] mask={mask_label} done")
    print(f"[cosine_mean_by_kappa] done -> {out}")


# =====================================================================
# SECTION 3: Crossing thresholds over kappa_star
# =====================================================================

def _compute_crossings(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per (mask_label, d, kappa_star) sweep with crossing columns."""
    out = []
    for row in rows:
        d = row["d"]
        kstar = float(row["kappa_star"])
        clt_baseline = 1.0 / math.sqrt(d)

        ntrain_rows = grouped_mean_by_ntrain(
            [row], CROSSING_KEYS  # single row; grouping is a no-op here
        )
        # we need to collect all ntrain rows for this (mask, d, kstar, sig)
        # this function is called with pre-grouped data — see run_crossings_over_kappa
        n_psd, v_psd, b_psd = first_crossing_vs_baseline(
            ntrain_rows, "cosine_S_S_star",
            ["random_baseline_cosine_S_S_star_mean", "random_baseline_cosine_S_S_star"],
        )
        n_clt, v_clt, b_clt = first_crossing_vs_constant(
            ntrain_rows, "cosine_S_S_star", clt_baseline,
        )
        item = {
            **{k: row[k] for k in ["mask_label", "d", "kappa_star", "T",
                                    "lambda_reg", "learning_rate", "n_steps",
                                    "beta_star", "sigma_star", "config_signature",
                                    "summary_path"] if k in row},
            "clt_baseline": clt_baseline,
            **crossing_columns(n_psd, d, kstar, v_psd, b_psd, "random_psd"),
            **crossing_columns(n_clt, d, kstar, v_clt, b_clt, "clt"),
        }
        out.append(item)
    return out

def _aggregate_crossings_by_mask_d_kappa(
    per_sweep: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_sweep:
        key = (str(row.get("config_signature", "")), str(row["mask_label"]), int(row["d"]), float(row["kappa_star"]))
        grouped[key].append(row)

    metric_keys = [
        "random_psd_cross_ntrain", "random_psd_cross_over_d",
        "random_psd_cross_over_d2", "random_psd_cross_over_kappa_d2",
        "clt_cross_ntrain", "clt_cross_over_d",
        "clt_cross_over_d2", "clt_cross_over_kappa_d2",
    ]

    out = []
    for (sig, mask_label, d, kstar), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "config_signature": sig,
            "mask_label": mask_label, "d": d, "kappa_star": kstar,
            "n_sweeps": len(group),
            "T": group[0].get("T"), "lambda_reg": group[0].get("lambda_reg"),
            "learning_rate": group[0].get("learning_rate"),
            "n_steps": group[0].get("n_steps"),
            "beta_star": group[0].get("beta_star"),
            "sigma_star": group[0].get("sigma_star"),
        }
        for key in metric_keys:
            vals = [float(r[key]) for r in group if r.get(key) is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals)) if vals else None
            item[f"{key}_count"] = len(vals)
        out.append(item)
    return out

def _crossing_plots_for_signature(
    aggregated: list[dict[str, Any]],
    sig_dir: Path,
    title_meta: str,
    no_title: bool,
) -> None:
    baseline_specs = [
        ("random_psd", "random PSD baseline"),
        ("clt", r"$1/\sqrt{d}$ baseline"),
    ]
    metric_specs = [
        ("cross_ntrain",        r"$n_{\mathrm{cross}}$",                    "ncross"),
        ("cross_over_d",        r"$n_{\mathrm{cross}}/d$",                  "ncross_over_d"),
        ("cross_over_d2",       r"$n_{\mathrm{cross}}/d^2$",                "ncross_over_d2"),
        ("cross_over_kappa_d2", r"$n_{\mathrm{cross}}/(\kappa^\star d^2)$", "ncross_over_kappa_d2"),
    ]
    masks = sorted({str(r["mask_label"]) for r in aggregated})

    for bp, bl in baseline_specs:
        bd = sig_dir / f"{bp}_crossings"
        bd.mkdir(parents=True, exist_ok=True)

        # all masks on one plot, fixed kappa_star
        kappas = sorted({float(r["kappa_star"]) for r in aggregated if r.get("kappa_star") is not None})

        for ym, yl, ft in metric_specs:
            for kstar in kappas:
                k_rows = [
                    r for r in aggregated
                    if float(r["kappa_star"]) == kstar
                ]

                fig, ax = plt.subplots(figsize=(14, 10))
                plotted = False

                for i, mask in enumerate(masks):
                    mask_rows = [r for r in k_rows if r["mask_label"] == mask]
                    plotted |= _plot_crossing_curve(
                        mask_rows,
                        "d",
                        f"{bp}_{ym}_mean",
                        f"{bp}_{ym}_std",
                        mask,
                        ax,
                        MARKERS[i % len(MARKERS)],
                    )

                if plotted:
                    ax.set_xlabel(r"$d$")
                    ax.set_ylabel(yl)

                    if not no_title:
                        ax.set_title(
                            rf"{yl} vs $d$, all masks, $\kappa^\star={kstar:g}$, {bl}"
                            + "\n"
                            + title_meta
                        )

                    ax.legend(frameon=True, fontsize=18)

                    kstr = str(kstar).replace(".", "p")
                    save_fig(
                        fig,
                        bd / fname(
                            f"{ft}_vs_d__all_masks__kappa{kstr}__{bp}.png",
                            no_title,
                        ),
                    )
                else:
                    plt.close(fig)

        # log-log n_cross vs d, all masks, fixed kappa_star
        for kstar in kappas:
            k_rows = [
                r for r in aggregated
                if float(r["kappa_star"]) == kstar
            ]

            fig, ax = plt.subplots(figsize=(14, 10))
            plotted = False

            for i, mask in enumerate(masks):
                mask_rows = [r for r in k_rows if r["mask_label"] == mask]
                plotted |= _plot_crossing_curve(
                    mask_rows,
                    "d",
                    f"{bp}_cross_ntrain_mean",
                    f"{bp}_cross_ntrain_std",
                    mask,
                    ax,
                    MARKERS[i % len(MARKERS)],
                )

            if plotted:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(r"$d$")
                ax.set_ylabel(r"$n_{\mathrm{cross}}$")

                if not no_title:
                    ax.set_title(
                        rf"$n_{{\mathrm{{cross}}}}$ vs $d$ (log-log), all masks, "
                        rf"$\kappa^\star={kstar:g}$, {bl}"
                        + "\n"
                        + title_meta
                    )

                ax.legend(frameon=True, fontsize=18)

                kstr = str(kstar).replace(".", "p")
                save_fig(
                    fig,
                    bd / fname(
                        f"ncross_vs_d__all_masks_loglog__kappa{kstr}__{bp}.png",
                        no_title,
                    ),
                )
            else:
                plt.close(fig)

        # kappa_star on x-axis, curves per d, one plot per mask
        for ym, yl, ft in metric_specs:
            for mask in masks:
                mask_rows = [r for r in aggregated if r["mask_label"] == mask]
                ds = sorted({int(r["d"]) for r in mask_rows})
                fig, ax = plt.subplots(figsize=(14, 10))
                plotted = False
                for i, d in enumerate(ds):
                    d_rows = [r for r in mask_rows if int(r["d"]) == d]
                    plotted |= _plot_crossing_curve(
                        d_rows, "kappa_star", f"{bp}_{ym}_mean", f"{bp}_{ym}_std",
                        rf"$d={d}$", ax, MARKERS[i % len(MARKERS)])
                if plotted:
                    ax.set_xlabel(r"$\kappa^\star$")
                    ax.set_ylabel(yl)
                    if not no_title:
                        ax.set_title(f"{yl} vs $\\kappa^\\star$, {mask}, {bl}\n{title_meta}")
                    ax.legend(title=r"$d$", title_fontsize=18, fontsize=20, frameon=True, ncol=2)
                    save_fig(fig, bd / fname(f"{ft}_vs_kappa__by_d__{mask}__{bp}.png", no_title))
                else:
                    plt.close(fig)

        # heatmaps over (d, kappa_star) per mask
        for ym, yl, ft in metric_specs:
            for mask in masks:
                mask_rows = [r for r in aggregated if r["mask_label"] == mask]
                ds = sorted({int(r["d"]) for r in mask_rows})
                kappas = sorted({float(r["kappa_star"]) for r in mask_rows})
                if not ds or not kappas:
                    continue
                matrix = np.full((len(ds), len(kappas)), np.nan)
                for r in mask_rows:
                    i = ds.index(int(r["d"]))
                    j = kappas.index(float(r["kappa_star"]))
                    v = r.get(f"{bp}_{ym}_mean")
                    if v is not None:
                        matrix[i, j] = float(v)
                if np.all(np.isnan(matrix)):
                    continue
                fig, ax = plt.subplots(figsize=(14, 10))
                im = ax.imshow(matrix, aspect="auto", origin="lower")
                ax.set_xticks(range(len(kappas)))
                ax.set_xticklabels([f"{k:.3g}" for k in kappas])
                ax.set_yticks(range(len(ds)))
                ax.set_yticklabels([str(d) for d in ds])
                ax.set_xlabel(r"$\kappa^\star$")
                ax.set_ylabel(r"$d$")
                if not no_title:
                    ax.set_title(f"{yl} over $(d,\\kappa^\\star)$, {mask}, {bl}")
                fig.colorbar(im, ax=ax).set_label(yl)
                save_fig(fig, bd / fname(f"heatmap_{ft}_over_d_kappa__{mask}__{bp}.png", no_title))


def run_crossings(
    rows: list[dict[str, Any]],
    output_dir: Path,
    no_title: bool,
) -> None:
    print("[crossings] running...")
    out = output_dir / "kappa_analysis"
    out.mkdir(parents=True, exist_ok=True)

    # collect all per-ntrain rows for crossing computation
    crossing_rows: list[dict[str, Any]] = []
    for summary_path in sorted({r["summary_path"] for r in rows}):
        csv_rows = read_csv_rows(Path(summary_path))
        if not csv_rows:
            continue
        meta = next((r for r in rows if r["summary_path"] == summary_path), None)
        if meta is None:
            continue
        d = meta["d"]
        kstar = float(meta["kappa_star"])
        clt = 1.0 / math.sqrt(d)
        ntrain_rows = grouped_mean_by_ntrain(csv_rows, CROSSING_KEYS)
        if not ntrain_rows:
            continue
        n_psd, v_psd, b_psd = first_crossing_vs_baseline(
            ntrain_rows, "cosine_S_S_star",
            ["random_baseline_cosine_S_S_star_mean", "random_baseline_cosine_S_S_star"])
        n_clt, v_clt, b_clt = first_crossing_vs_constant(
            ntrain_rows, "cosine_S_S_star", clt)
        crossing_rows.append({
            **{k: meta[k] for k in meta if k != "base_config"},
            "clt_baseline": clt,
            **crossing_columns(n_psd, d, kstar, v_psd, b_psd, "random_psd"),
            **crossing_columns(n_clt, d, kstar, v_clt, b_clt, "clt"),
        })

    if not crossing_rows:
        print("[crossings] no crossing rows found, skipping")
        return

    write_csv(crossing_rows, out / "crossings_per_sweep.csv")

    aggregated = _aggregate_crossings_by_mask_d_kappa(crossing_rows)
    write_csv(aggregated, out / "crossings_aggregated.csv")

    base_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in aggregated:
        base_sig = base_config_signature(str(r["config_signature"]))
        base_groups[base_sig].append(r)

    for base_sig, sig_rows in sorted(base_groups.items()):
        title_meta = build_kappa_comparison_title(sig_rows)
        sig_dir = out / sanitize_filename(base_sig)
        _crossing_plots_for_signature(sig_rows, sig_dir, title_meta, no_title)

    print(f"[crossings] done -> {out}")


# =====================================================================
# SECTION 4: Population risk vs cosine similarity
# =====================================================================

def _aggregate_risk_cosine(
    rows: list[dict[str, Any]],
) -> dict[int, list[tuple[int, float, float, float]]]:
    """d -> [(n_train, mean_cosine, mean_risk, baseline_cosine)]"""
    grouped: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        d, n = row["d"], row["n_train"]
        for key in ["cosine_S_S_star", "population_risk",
                    "random_baseline_cosine_S_S_star_mean",
                    "random_baseline_cosine_S_S_star"]:
            v = row.get(key)
            if v is not None:
                grouped[d][n][key].append(v)

    out: dict[int, list[tuple]] = {}
    for d, by_n in grouped.items():
        pts = []
        for n in sorted(by_n):
            vals = by_n[n]
            cosine = float(np.mean(vals["cosine_S_S_star"])) if vals.get("cosine_S_S_star") else None
            risk = float(np.mean(vals["population_risk"])) if vals.get("population_risk") else None
            baseline = None
            for k in ["random_baseline_cosine_S_S_star_mean", "random_baseline_cosine_S_S_star"]:
                if vals.get(k):
                    baseline = float(np.mean(vals[k]))
                    break
            pts.append((n, cosine, risk, baseline))
        out[d] = pts
    return out



# =====================================================================
# SECTION 5: Train loss, population risk, and cosine vs n_train
# =====================================================================

LOSS_ZOOM_RANGES: list[tuple[int | None, int | None]] = [
    (None, None),
    (0, 500),
    (0, 1000),
    (0, 2500),
    (0, 5000),
    (0, 10000),
]


def _zoom_suffix(xlim: tuple[int | None, int | None]) -> str:
    if xlim == (None, None):
        return "full"
    left = "0" if xlim[0] is not None else "min"
    right = str(xlim[1]) if xlim[1] is not None else "max"
    return f"zoom_{left}_{right}"


def _aggregate_losses_and_cosine(
    rows: list[dict[str, Any]],
) -> dict[int, list[tuple[int, float | None, float | None, float | None, float | None]]]:
    """d -> [(n_train, mean_train_loss, mean_pop_risk, mean_cosine, mean_baseline)]"""
    grouped: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        d, n = row["d"], row["n_train"]
        for key in ["train_loss", "population_risk", "cosine_S_S_star",
                    "random_baseline_cosine_S_S_star_mean",
                    "random_baseline_cosine_S_S_star"]:
            v = row.get(key)
            if v is not None:
                grouped[d][n][key].append(v)

    out: dict[int, list[tuple]] = {}
    for d, by_n in grouped.items():
        pts = []
        for n in sorted(by_n):
            vals = by_n[n]
            def mean(k: str) -> float | None:
                return float(np.mean(vals[k])) if vals.get(k) else None
            baseline = mean("random_baseline_cosine_S_S_star_mean") or mean("random_baseline_cosine_S_S_star")
            pts.append((n, mean("train_loss"), mean("population_risk"), mean("cosine_S_S_star"), baseline))
        out[d] = pts
    return out


def _apply_zoom(
    ntrain: list[int],
    *series: list[float | None],
    xlim: tuple[int | None, int | None],
) -> tuple[list[int], list[list[float | None]]]:
    lo, hi = xlim
    mask = [
        (lo is None or n >= lo) and (hi is None or n <= hi)
        for n in ntrain
    ]
    filtered_n = [n for n, m in zip(ntrain, mask) if m]
    filtered_series = [
        [v for v, m in zip(s, mask) if m]
        for s in series
    ]
    return filtered_n, filtered_series


def run_losses(
    rows: list[dict[str, Any]],
    output_dir: Path,
    no_title: bool,
) -> None:
    print("[losses] running...")
    out = output_dir / "train_vs_risk"

    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        key = (str(r["config_signature"]), float(r["kappa_star"]))
        groups[key].append(r)

    print(f"[losses] done -> {out}")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all analysis plots on aggregated results."
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing kappa_star_* folders.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to <root>/analysis.")
    parser.add_argument("--no-title", action="store_true",
                        help="Omit plot titles. Output filenames get _nt appended.")
    parser.add_argument("--skip-cosine-by-d", action="store_true")
    parser.add_argument("--skip-cosine-mean", action="store_true")
    parser.add_argument("--skip-crossings", action="store_true")
    parser.add_argument("--skip-losses", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    no_title = args.no_title

    print(f"Loading data from {root}...")
    all_keys = list(set(CROSSING_KEYS + RISK_KEYS))
    rows = collect_rows(root, all_keys)
    print(f"Loaded {len(rows)} records.")

    if not rows:
        print("No data found. Did you run aggregate_teacher_attention_sweep.py first?")
        return

    if not args.skip_cosine_by_d:
        run_cosine_by_d(rows, output_dir, no_title)

    if not args.skip_cosine_mean:
        run_cosine_mean_by_kappa(rows, output_dir, no_title)

    if not args.skip_crossings:
        run_crossings(rows, output_dir, no_title)

    if not args.skip_losses:
        run_losses(rows, output_dir, no_title)

    print(f"\nAll done. Output in: {output_dir}")


if __name__ == "__main__":
    main()