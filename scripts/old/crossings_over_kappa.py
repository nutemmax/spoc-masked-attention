from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crossing_utils_old import (
    first_unique,
    format_mask_label,
    get_float,
    get_int,
    grouped_mean_by_ntrain,
    parse_float_token,
    parse_from_name,
    parse_mask_from_name,
    read_csv_rows,
    read_json,
    resolve_r_star,
    find_config_folder_name,
    infer_kappa_from_parent_dirs,
    has_kappa_star_ancestor
)

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 28,
    "axes.grid": False,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
})

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
CROSSING_KEYS = [
    "cosine_S_S_star",
    "random_baseline_cosine_S_S_star",
    "random_baseline_cosine_S_S_star_mean",
]

# ---------------------------------------------------------------------
# metadata extraction
# ---------------------------------------------------------------------

def extract_metadata(summary_path: Path, rows: list[dict[str, str]]) -> dict[str, Any] | None:
    sweep_dir = summary_path.parent
    config_name = find_config_folder_name(summary_path)

    sweep_config = read_json(sweep_dir / "sweep_config.json")
    base_config = sweep_config.get("base_config") if sweep_config is not None else None

    d = None
    T = None
    r = None
    r_star = None
    kappa = None
    kappa_star = None
    lambda_reg = None
    learning_rate = None
    n_steps = None
    beta = None
    beta_star = None
    sigma_star = None
    masking_strategy = None
    masks_per_sample = None

    if base_config is not None:
        data_cfg = base_config.get("data", {})
        model_cfg = base_config.get("model", {})
        teacher_cfg = base_config.get("teacher", {})
        training_cfg = base_config.get("training", {})

        d = data_cfg.get("d")
        T = data_cfg.get("T")
        masking_strategy = data_cfg.get("masking_strategy")
        masks_per_sample = data_cfg.get("masks_per_sample")

        r = model_cfg.get("r")
        beta = model_cfg.get("beta")

        r_star = resolve_r_star(teacher_cfg.get("r_star"), int(d) if d is not None else None)
        beta_star = teacher_cfg.get("beta_star")
        sigma_star = teacher_cfg.get("sigma_star")

        lambda_reg = training_cfg.get("lambda_reg")
        learning_rate = training_cfg.get("learning_rate")
        n_steps = training_cfg.get("n_steps")

    if rows:
        first = rows[0]

        if d is None:
            d = get_int(first, "d")
        if T is None:
            T = get_int(first, "T")
        if r is None:
            r = get_int(first, "r")
        if r_star is None:
            r_star = get_int(first, "r_star")
        if kappa is None:
            kappa = get_float(first, "kappa")
        if kappa_star is None:
            kappa_star = get_float(first, "kappa_star")
        if lambda_reg is None:
            lambda_reg = get_float(first, "lambda_reg")
        if learning_rate is None:
            learning_rate = get_float(first, "learning_rate")
        if n_steps is None:
            n_steps = get_int(first, "n_steps")
        if beta_star is None:
            beta_star = get_float(first, "beta_star")
        if sigma_star is None:
            sigma_star = get_float(first, "sigma_star")

    # Folder-name fallback.
    if d is None:
        d_token = parse_from_name(config_name, r"_d(\d+)")
        d = int(d_token) if d_token is not None else None

    if r is None:
        r_token = parse_from_name(config_name, r"_r_(\d+)")
        r = int(r_token) if r_token is not None else None

    if r_star is None:
        rstar_token = parse_from_name(config_name, r"_rstar_(\d+)")
        r_star = int(rstar_token) if rstar_token is not None else None

    if lambda_reg is None:
        lambda_token = parse_from_name(config_name, r"_lambda([0-9p]+)")
        lambda_reg = parse_float_token(lambda_token)

    if beta_star is None:
        beta_star_token = parse_from_name(config_name, r"_bstar_([0-9p]+)")
        beta_star = parse_float_token(beta_star_token)

    if beta is None:
        beta_token = parse_from_name(config_name, r"_beta_([0-9p]+)")
        beta = parse_float_token(beta_token)

    if sigma_star is None:
        sigma_token = parse_from_name(config_name, r"_sigstar_([0-9p]+)")
        sigma_star = parse_float_token(sigma_token)

    if n_steps is None:
        steps_token = parse_from_name(config_name, r"_iter(\d+)")
        n_steps = int(steps_token) if steps_token is not None else None

    if T is None:
        T_token = parse_from_name(config_name, r"_T(\d+)")
        T = int(T_token) if T_token is not None else None

    if d is None:
        print(f"[skip] could not infer d for {summary_path}")
        return None

    d = int(d)

    if r is not None:
        r = int(r)
    if r_star is not None:
        r_star = int(r_star)

    if kappa is None and r is not None:
        kappa = float(r) / float(d)

    if kappa_star is None and r_star is not None:
        kappa_star = float(r_star) / float(d)

    if kappa_star is None:
        kappa_star = infer_kappa_from_parent_dirs(summary_path)

    if kappa_star is None:
        print(f"[skip] could not infer kappa_star for {summary_path}")
        return None

    mask_label = format_mask_label(masking_strategy, masks_per_sample)
    if mask_label is None:
        mask_label = parse_mask_from_name(config_name)

    if mask_label is None:
        print(f"[skip] could not infer mask label for {summary_path}")
        return None

    return {
        "summary_path": str(summary_path),
        "config_name": config_name,
        "job_name": summary_path.parent.name,
        "d": int(d),
        "T": int(T) if T is not None else None,
        "r": int(r) if r is not None else None,
        "r_star": int(r_star) if r_star is not None else None,
        "kappa": float(kappa) if kappa is not None else None,
        "kappa_star": float(kappa_star),
        "lambda_reg": float(lambda_reg) if lambda_reg is not None else None,
        "learning_rate": float(learning_rate) if learning_rate is not None else None,
        "n_steps": int(n_steps) if n_steps is not None else None,
        "beta": float(beta) if beta is not None else None,
        "beta_star": float(beta_star) if beta_star is not None else None,
        "sigma_star": float(sigma_star) if sigma_star is not None else None,
        "mask_label": mask_label,
    }

# ---------------------------------------------------------------------
# Crossing computation
# ---------------------------------------------------------------------
def first_crossing_random_psd(rows_by_ntrain: list[dict]) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_ntrain:
        learned = row.get("cosine_S_S_star")
        baseline = row.get("random_baseline_cosine_S_S_star_mean")
        if baseline is None:
            baseline = row.get("random_baseline_cosine_S_S_star")

        if learned is None or baseline is None:
            continue

        if float(learned) >= float(baseline):
            return int(row["n_train"]), float(learned), float(baseline)

    return None, None, None


def first_crossing_constant(rows_by_ntrain: list[dict], baseline: float) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_ntrain:
        learned = row.get("cosine_S_S_star")

        if learned is None:
            continue

        if float(learned) >= float(baseline):
            return int(row["n_train"]), float(learned), float(baseline)

    return None, None, None


def add_crossing_columns(out: dict[str, Any], prefix: str, n_cross: int | None, value_at_crossing: float | None, baseline_at_crossing: float | None) -> None:
    d = float(out["d"])
    kappa_star = float(out["kappa_star"])

    out[f"{prefix}_cross_ntrain"] = n_cross
    out[f"{prefix}_cross_value"] = value_at_crossing
    out[f"{prefix}_cross_baseline"] = baseline_at_crossing

    out[f"{prefix}_cross_over_d"] = n_cross / d if n_cross is not None else None
    out[f"{prefix}_cross_over_d2"] = n_cross / (d ** 2) if n_cross is not None else None
    out[f"{prefix}_cross_over_kappa_d2"] = (
        n_cross / (kappa_star * (d ** 2))
        if n_cross is not None and kappa_star > 0
        else None
    )


def analyze_summary(summary_path: Path) -> dict[str, Any] | None:
    rows = read_csv_rows(summary_path)
    if not rows:
        return None

    metadata = extract_metadata(summary_path, rows)
    if metadata is None:
        return None

    rows_by_ntrain = grouped_mean_by_ntrain(rows, CROSSING_KEYS)
    if not rows_by_ntrain:
        return None

    d = int(metadata["d"])
    clt_baseline = 1.0 / math.sqrt(d)

    out: dict[str, Any] = {
        **metadata,
        "clt_baseline": clt_baseline,
        "min_ntrain_available": min(int(row["n_train"]) for row in rows_by_ntrain),
        "max_ntrain_available": max(int(row["n_train"]) for row in rows_by_ntrain),
        "n_points": len(rows_by_ntrain),
    }

    n_cross, value, baseline = first_crossing_random_psd(rows_by_ntrain)
    add_crossing_columns(out=out, prefix="random_psd", n_cross=n_cross, value_at_crossing=value, baseline_at_crossing=baseline)

    n_cross, value, baseline = first_crossing_constant(rows_by_ntrain=rows_by_ntrain, baseline=clt_baseline)
    add_crossing_columns(out=out, prefix="clt", n_cross=n_cross, value_at_crossing=value, baseline_at_crossing=baseline)

    return out


def aggregate_crossings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}

    for row in rows:
        key = (str(row["mask_label"]), int(row["d"]), float(row["kappa_star"]))
        grouped.setdefault(key, []).append(row)

    metric_keys = [
        "random_psd_cross_ntrain",
        "random_psd_cross_over_d",
        "random_psd_cross_over_d2",
        "random_psd_cross_over_kappa_d2",

        "clt_cross_ntrain",
        "clt_cross_over_d",
        "clt_cross_over_d2",
        "clt_cross_over_kappa_d2",
    ]

    out = []

    for (mask_label, d, kappa_star), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        item: dict[str, Any] = {
            "mask_label": mask_label,
            "d": d,
            "kappa_star": kappa_star,
            "n_sweeps": len(group),
            "T": first_unique(group, "T"),
            "lambda_reg": first_unique(group, "lambda_reg"),
            "learning_rate": first_unique(group, "learning_rate"),
            "n_steps": first_unique(group, "n_steps"),
            "beta_star": first_unique(group, "beta_star"),
            "sigma_star": first_unique(group, "sigma_star"),
        }

        for key in metric_keys:
            values = [float(row[key]) for row in group if row.get(key) is not None]
            if values:
                item[f"{key}_mean"] = float(np.mean(values))
                item[f"{key}_std"] = float(np.std(values))
                item[f"{key}_count"] = int(len(values))
            else:
                item[f"{key}_mean"] = None
                item[f"{key}_std"] = None
                item[f"{key}_count"] = 0

        out.append(item)

    return out

# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    preferred = [
        "mask_label",
        "d",
        "kappa_star",
        "n_sweeps",
        "T",
        "lambda_reg",
        "learning_rate",
        "n_steps",
        "beta_star",
        "sigma_star",

        "random_psd_cross_ntrain",
        "random_psd_cross_over_d",
        "random_psd_cross_over_d2",
        "random_psd_cross_over_kappa_d2",
        "random_psd_cross_value",
        "random_psd_cross_baseline",

        "clt_cross_ntrain",
        "clt_cross_over_d",
        "clt_cross_over_d2",
        "clt_cross_over_kappa_d2",
        "clt_cross_value",
        "clt_cross_baseline",

        "min_ntrain_available",
        "max_ntrain_available",
        "n_points",
        "summary_path",
    ]

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames += sorted(key for key in all_keys if key not in fieldnames)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_curve(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    y_std_key: str,
    label_key: str,
    label_value: Any,
    ax,
    marker: str,
) -> bool:
    sub = [row for row in rows if row.get(label_key) == label_value]

    xs = []
    ys = []
    stds = []

    for row in sub:
        x = row.get(x_key)
        y = row.get(y_key)

        if x is None or y is None:
            continue

        x = float(x)
        y = float(y)

        if x <= 0 or y <= 0:
            continue

        xs.append(x)
        ys.append(y)
        stds.append(float(row.get(y_std_key) or 0.0))

    if not xs:
        return False

    xs_np = np.asarray(xs, dtype=float)
    ys_np = np.asarray(ys, dtype=float)
    stds_np = np.asarray(stds, dtype=float)

    order = np.argsort(xs_np)
    xs_np = xs_np[order]
    ys_np = ys_np[order]
    stds_np = stds_np[order]

    ax.plot(xs_np, ys_np, marker=marker, linewidth=3, markersize=16, label=str(label_value), alpha=0.7)

    if np.any(stds_np > 0):
        lower = np.maximum(ys_np - stds_np, 1e-12)
        upper = ys_np + stds_np
        ax.fill_between(xs_np, lower, upper, alpha=0.08)

    return True


def plot_over_kappa_by_d(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_prefix: str,
    baseline_label: str,
    y_metric: str,
    y_label: str,
    filename_tag: str,
) -> None:
    """
    For each masking scheme:
        x-axis: kappa_star
        curves: one per d
    """
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    for mask in masks:
        mask_rows = [row for row in rows if row["mask_label"] == mask]
        ds = sorted({int(row["d"]) for row in mask_rows if row.get("d") is not None})

        fig, ax = plt.subplots(figsize=(14, 10))

        plotted = False
        for i, d in enumerate(ds):
            marker = MARKERS[i % len(MARKERS)]
            plotted |= plot_curve(
                rows=mask_rows,
                x_key="kappa_star",
                y_key=f"{baseline_prefix}_{y_metric}_mean",
                y_std_key=f"{baseline_prefix}_{y_metric}_std",
                label_key="d",
                label_value=d,
                ax=ax,
                marker=marker,
            )

        if plotted:
            ax.set_xlabel(r"$\kappa^\star$")
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} vs $\\kappa^\\star$\n{mask}, {baseline_label}")
            ax.legend(title=r"$d$", frameon=True, ncol=2)
            fig.tight_layout()

            out_path = output_dir / f"{filename_tag}_vs_kappa__by_d__{mask}__{baseline_prefix}.png"
            fig.savefig(out_path, bbox_inches="tight")

        plt.close(fig)


def plot_over_kappa_by_d_scaled(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_prefix: str,
    baseline_label: str,
    y_metric: str,
    y_label: str,
    filename_tag: str,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """
    Diagnostic version of plot_over_kappa_by_d with optional log axes.
    For each masking scheme:
        x-axis: kappa_star
        curves: one per d
    """
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    for mask in masks:
        mask_rows = [row for row in rows if row["mask_label"] == mask]
        ds = sorted({int(row["d"]) for row in mask_rows if row.get("d") is not None})

        fig, ax = plt.subplots(figsize=(14, 10))

        plotted = False
        for i, d in enumerate(ds):
            marker = MARKERS[i % len(MARKERS)]
            plotted |= plot_curve(
                rows=mask_rows,
                x_key="kappa_star",
                y_key=f"{baseline_prefix}_{y_metric}_mean",
                y_std_key=f"{baseline_prefix}_{y_metric}_std",
                label_key="d",
                label_value=d,
                ax=ax,
                marker=marker,
            )

        if plotted:
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

            ax.set_xlabel(r"$\kappa^\star$")
            ax.set_ylabel(y_label)

            scale_text = []
            if log_x:
                scale_text.append("log x")
            if log_y:
                scale_text.append("log y")
            scale_suffix = ", ".join(scale_text)

            title = f"{y_label} vs $\\kappa^\\star$\n{mask}, {baseline_label}"
            if scale_suffix:
                title += f" ({scale_suffix})"

            ax.set_title(title)
            ax.legend(title=r"$d$", frameon=True, ncol=2)
            fig.tight_layout()

            out_path = output_dir / f"{filename_tag}_vs_kappa__by_d__{mask}__{baseline_prefix}.png"
            fig.savefig(out_path, bbox_inches="tight")

        plt.close(fig)


def plot_over_kappa_all_masks(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_prefix: str,
    baseline_label: str,
    y_metric: str,
    y_label: str,
    filename_tag: str,
) -> None:
    """
    For each fixed d:
        x-axis: kappa_star
        curves: one per mask
    """
    ds = sorted({int(row["d"]) for row in rows if row.get("d") is not None})
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    for d in ds:
        d_rows = [row for row in rows if int(row["d"]) == d]

        fig, ax = plt.subplots(figsize=(14, 10))

        plotted = False
        for i, mask in enumerate(masks):
            marker = MARKERS[i % len(MARKERS)]
            plotted |= plot_curve(
                rows=d_rows,
                x_key="kappa_star",
                y_key=f"{baseline_prefix}_{y_metric}_mean",
                y_std_key=f"{baseline_prefix}_{y_metric}_std",
                label_key="mask_label",
                label_value=mask,
                ax=ax,
                marker=marker,
            )

        if plotted:
            ax.set_xlabel(r"$\kappa^\star$")
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} vs $\\kappa^\\star$\n$d={d}$, {baseline_label}")
            ax.legend(frameon=True)
            fig.tight_layout()

            out_path = output_dir / f"{filename_tag}_vs_kappa__all_masks__d{d}__{baseline_prefix}.png"
            fig.savefig(out_path, bbox_inches="tight")

        plt.close(fig)


def plot_heatmap_over_d_kappa(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_prefix: str,
    baseline_label: str,
    y_metric: str,
    colorbar_label: str,
    filename_tag: str,
) -> None:
    """
    For each masking scheme:
        x-axis: kappa_star
        y-axis: d
        color: chosen crossing metric
    """
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    for mask in masks:
        mask_rows = [row for row in rows if row["mask_label"] == mask]
        ds = sorted({int(row["d"]) for row in mask_rows if row.get("d") is not None})
        kappas = sorted({float(row["kappa_star"]) for row in mask_rows if row.get("kappa_star") is not None})

        if not ds or not kappas:
            continue

        matrix = np.full((len(ds), len(kappas)), np.nan, dtype=float)
        d_to_i = {d: i for i, d in enumerate(ds)}
        k_to_j = {kappa: j for j, kappa in enumerate(kappas)}

        for row in mask_rows:
            d = int(row["d"])
            kappa = float(row["kappa_star"])
            value = row.get(f"{baseline_prefix}_{y_metric}_mean")

            if value is None:
                continue

            matrix[d_to_i[d], k_to_j[kappa]] = float(value)

        if np.all(np.isnan(matrix)):
            continue

        fig, ax = plt.subplots(figsize=(14, 10))

        im = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_xticks(np.arange(len(kappas)))
        ax.set_xticklabels([f"{k:.3g}" for k in kappas])
        ax.set_yticks(np.arange(len(ds)))
        ax.set_yticklabels([str(d) for d in ds])

        ax.set_xlabel(r"$\kappa^\star$")
        ax.set_ylabel(r"$d$")
        ax.set_title(f"{colorbar_label} over $(d, \\kappa^\\star)$\n{mask}, {baseline_label}")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

        fig.tight_layout()
        out_path = output_dir / f"heatmap_{filename_tag}_over_d_kappa__{mask}__{baseline_prefix}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def make_plots(aggregated_rows: list[dict[str, Any]], output_dir: Path) -> None:
    plots_dir = output_dir / "plots_over_kappa"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_specs = [
        ("random_psd", "random PSD baseline"),
        ("clt", r"$1/\sqrt{d}$ baseline"),
    ]

    metric_specs = [
        ("cross_ntrain", r"$n_{\mathrm{cross}}$", "ncross"),
        ("cross_over_d", r"$n_{\mathrm{cross}}/d$", "ncross_over_d"),
        ("cross_over_d2", r"$n_{\mathrm{cross}}/d^2$", "ncross_over_d2"),
        ("cross_over_kappa_d2", r"$n_{\mathrm{cross}}/(\kappa^\star d^2)$", "ncross_over_kappa_d2"),
    ]

    for baseline_prefix, baseline_label in baseline_specs:
        baseline_dir = plots_dir / baseline_prefix
        baseline_dir.mkdir(parents=True, exist_ok=True)

        for y_metric, y_label, filename_tag in metric_specs:
            plot_over_kappa_by_d(
                rows=aggregated_rows,
                output_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                y_label=y_label,
                filename_tag=filename_tag,
            )

            plot_over_kappa_all_masks(
                rows=aggregated_rows,
                output_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                y_label=y_label,
                filename_tag=filename_tag,
            )

        diagnostic_dir = baseline_dir # / "scaling_diagnostics"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

        # 1. log-log n_cross vs kappa_star, fixed d, one plot per mask.
        plot_over_kappa_by_d_scaled(
            rows=aggregated_rows,
            output_dir=diagnostic_dir,
            baseline_prefix=baseline_prefix,
            baseline_label=baseline_label,
            y_metric="cross_ntrain",
            y_label=r"$n_{\mathrm{cross}}$",
            filename_tag="loglog_ncross",
            log_x=True,
            log_y=True,
        )

        # 2.1. log-log n_cross/d^2 vs kappa_star, fixed d, one plot per mask
        plot_over_kappa_by_d_scaled(
            rows=aggregated_rows,
            output_dir=diagnostic_dir,
            baseline_prefix=baseline_prefix,
            baseline_label=baseline_label,
            y_metric="cross_over_d2",
            y_label=r"$n_{\mathrm{cross}}/d^2$",
            filename_tag="loglog_ncross_over_d2",
            log_x=True,
            log_y=True,
        )

        # 2.2. log-log n_cross/d vs kappa_star, fixed d, one plot per mask
        plot_over_kappa_by_d_scaled(
            rows=aggregated_rows,
            output_dir=diagnostic_dir,
            baseline_prefix=baseline_prefix,
            baseline_label=baseline_label,
            y_metric="cross_over_d",
            y_label=r"$n_{\mathrm{cross}}/d$",
            filename_tag="loglog_ncross_over_d",
            log_x=True,
            log_y=True,
        )

        # 3. n_cross/(kappa_star d^2) vs kappa_star, log y-axis, fixed d, one plot per mask.
        plot_over_kappa_by_d_scaled(
            rows=aggregated_rows,
            output_dir=diagnostic_dir,
            baseline_prefix=baseline_prefix,
            baseline_label=baseline_label,
            y_metric="cross_over_kappa_d2",
            y_label=r"$n_{\mathrm{cross}}/(\kappa^\star d^2)$",
            filename_tag="logy_ncross_over_kappa_d2",
            log_x=False,
            log_y=True,
        )

        # 4. heatmap of n_cross/(kappa_star d^2) over (d, kappa_star), one heatmap per mask.
        heatmap_specs = [
            ("cross_ntrain", r"$n_{\mathrm{cross}}$", "ncross"),
            ("cross_over_d", r"$n_{\mathrm{cross}}/d$", "ncross_over_d"),
            ("cross_over_d2", r"$n_{\mathrm{cross}}/d^2$", "ncross_over_d2"),
            ("cross_over_kappa_d2", r"$n_{\mathrm{cross}}/(\kappa^\star d^2)$", "ncross_over_kappa_d2"),
        ]

        for y_metric, colorbar_label, filename_tag in heatmap_specs:
            plot_heatmap_over_d_kappa(
                rows=aggregated_rows,
                output_dir=diagnostic_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                colorbar_label=colorbar_label,
                filename_tag=filename_tag,
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=("Root directory containing kappa_star_* folders, eg results/teacher-attention/iter_5000. "),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <root>/kappa_analysis.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else root / "kappa_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(p for p in root.rglob("summary.csv") if has_kappa_star_ancestor(p))

    if not summary_paths:
        print(f"No summary.csv files found under {root}")
        return

    per_sweep_rows = []

    for summary_path in summary_paths:
        row = analyze_summary(summary_path)
        if row is not None:
            per_sweep_rows.append(row)

    if not per_sweep_rows:
        print("No valid crossing rows found.")
        return

    per_sweep_rows = sorted(
        per_sweep_rows,
        key=lambda row: (
            str(row["mask_label"]),
            int(row["d"]),
            float(row["kappa_star"]),
            str(row["summary_path"]),
        ),
    )

    per_sweep_path = output_dir / "crossings_over_kappa_per_sweep.csv"
    write_csv(per_sweep_rows, per_sweep_path)

    aggregated_rows = aggregate_crossings(per_sweep_rows)
    aggregated_rows = sorted(
        aggregated_rows,
        key=lambda row: (
            str(row["mask_label"]),
            int(row["d"]),
            float(row["kappa_star"]),
        ),
    )

    aggregated_path = output_dir / "crossings_over_kappa_by_mask_d_kappa.csv"
    write_csv(aggregated_rows, aggregated_path)

    make_plots(aggregated_rows, output_dir)

    print(f"[done] Wrote per-sweep crossings to: {per_sweep_path}")
    print(f"[done] Wrote aggregated crossings to: {aggregated_path}")
    print(f"[done] Wrote plots to: {output_dir / 'plots_over_kappa'}")


if __name__ == "__main__":
    main()