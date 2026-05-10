from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crossing_utils import (
    format_float_for_title,
    format_mask_label,
    get_float,
    get_int,
    grouped_mean_by_ntrain,
    one_or_mixed,
    parse_mask_from_name,
    read_csv_rows,
    read_json,
    resolve_r_star,
    sanitize_filename,
)


plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 34,
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



def group_teacher_signature(
    teacher_init: str | None,
    sigma_star: float | None,
) -> tuple[str, float]:
    """
    Group standard_gaussian together with scaled_gaussian at sigma_star = 1.0.
    """
    init = "NA" if teacher_init is None else str(teacher_init)
    sigma = 1.0 if sigma_star is None else float(sigma_star)

    if init == "standard_gaussian":
        return "scaled_gaussian", 1.0

    if init == "scaled_gaussian" and np.isclose(sigma, 1.0):
        return "scaled_gaussian", 1.0

    return init, sigma




def unique_non_none(rows: list[dict], key: str) -> list[object]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and value not in values:
            values.append(value)
    return values



def build_title_metadata(rows: list[dict]) -> str:
    """
    Build compact title metadata from one config-signature group.
    Assumes these values are constant within the group, except d.
    """
    kappa = one_or_mixed(rows, "kappa")
    kappa_star = one_or_mixed(rows, "kappa_star")
    T = one_or_mixed(rows, "T")
    lambda_reg = one_or_mixed(rows, "lambda_reg")
    learning_rate = one_or_mixed(rows, "learning_rate")
    n_steps = one_or_mixed(rows, "n_steps")

    return (
        rf"$\kappa^\star = {format_float_for_title(kappa_star)}$, "
        rf"$\kappa = {format_float_for_title(kappa)}$, "
        rf"$T = {format_float_for_title(T)}$, "
        rf"$\lambda = {format_float_for_title(lambda_reg)}$, "
        rf"$\eta = {format_float_for_title(learning_rate)}$, "
        rf"iters $= {format_float_for_title(n_steps)}$"
    )




def build_config_signature(base_config: dict | None) -> str:
    """
    Build a signature for grouping experiments that differ only by d.
    Intentionally excluded:
    - data.d
    - model.r
    - teacher.r_star
    - evaluation.pca_n_components

    Also excluded because they do not influence the learning curve shape:
    - training.alpha, since n_train is varied explicitly
    - experiment/logging fields
    - device
    - n_population
    - eval_every
    - attention_metric_subset_size
    - n_random_baselines
    - use_wandb

    Also groups together:
    - standard_gaussian
    - scaled_gaussian with sigma_star = 1
    """
    if base_config is None:
        return "unknown_config"

    data_cfg = base_config.get("data", {})
    model_cfg = base_config.get("model", {})
    teacher_cfg = base_config.get("teacher", {})
    training_cfg = base_config.get("training", {})

    teacher_init_canon, sigma_star_canon = group_teacher_signature(
        teacher_init=teacher_cfg.get("init"),
        sigma_star=teacher_cfg.get("sigma_star"),
    )

    parts = [
        f"data_model={data_cfg.get('data_model')}",
        f"T={data_cfg.get('T')}",
        f"mask_value={data_cfg.get('mask_value')}",
        f"teacher_init={teacher_init_canon}",
        f"sigma_star={sigma_star_canon}",
        f"beta_star={teacher_cfg.get('beta_star')}",
        f"beta={model_cfg.get('beta')}",
        f"normalize_sqrt_d={model_cfg.get('normalize_sqrt_d')}",
        f"dtype={model_cfg.get('dtype')}",
        f"lambda={training_cfg.get('lambda_reg')}",
        f"lr={training_cfg.get('learning_rate')}",
        f"n_steps={training_cfg.get('n_steps')}",
    ]

    return "__".join(parts)


def find_first_run_config(sweep_dir: Path) -> dict | None:
    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir():
            continue

        config_files = sorted(
            p for p in subdir.iterdir()
            if p.is_file() and p.name.startswith("config") and p.name.endswith(".json")
        )

        if config_files:
            return read_json(config_files[0])

    return None


def extract_d_from_rows(rows: list[dict]) -> int | None:
    """
    Extract d from summary.csv rows if the column exists.
    """
    for row in rows:
        d = get_int(row, "d")
        if d is not None:
            return d
    return None


def extract_metadata(summary_path: Path, rows: list[dict]) -> dict:
    sweep_dir = summary_path.parent
    metadata_path = sweep_dir / "sweep_config.json"
    metadata = read_json(metadata_path)

    base_config = None
    d = None
    T = None
    r = None
    r_star = None
    kappa = None
    kappa_star = None
    lambda_reg = None
    learning_rate = None
    n_steps = None
    masking_strategy = None
    masks_per_sample = None

    config_name = sweep_dir.parent.name
    sweep_name = sweep_dir.name

    if metadata is not None:
        base_config = metadata.get("base_config", None)

    if base_config is None:
        base_config = find_first_run_config(sweep_dir)

    if base_config is not None:
        data_cfg = base_config.get("data", {})
        model_cfg = base_config.get("model", {})
        teacher_cfg = base_config.get("teacher", {})
        training_cfg = base_config.get("training", {})

        d = data_cfg.get("d", None)
        T = data_cfg.get("T", None)
        masking_strategy = data_cfg.get("masking_strategy", None)
        masks_per_sample = data_cfg.get("masks_per_sample", None)

        r = model_cfg.get("r", None)
        raw_r_star = teacher_cfg.get("r_star", None)
        r_star = resolve_r_star(raw_r_star, int(d) if d is not None else None)

        lambda_reg = training_cfg.get("lambda_reg", None)
        learning_rate = training_cfg.get("learning_rate", None)
        n_steps = training_cfg.get("n_steps", None)

    if d is None:
        d = extract_d_from_rows(rows)

    if rows:
        first = rows[0]

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

    if d is None:
        print(
            f"[skip] Could not infer d from sweep_config.json, run config*.json, "
            f"or summary.csv for {summary_path}"
        )

    if d is not None:
        d = int(d)

    if T is not None:
        T = int(T)

    if r is not None:
        r = int(r)

    if r_star is not None:
        r_star = resolve_r_star(r_star, d)

    if kappa is None and r is not None and d is not None:
        kappa = float(r) / float(d)

    if kappa_star is None and r_star is not None and d is not None:
        kappa_star = float(r_star) / float(d)

    mask_label = format_mask_label(masking_strategy, masks_per_sample)
    if mask_label is None:
        mask_label = parse_mask_from_name(config_name)

    seed = None
    if rows:
        seed = get_int(rows[0], "master_seed")
        if seed is None:
            seed = get_int(rows[0], "seed")

    config_signature = build_config_signature(base_config)

    return {
        "config_signature": config_signature,
        "config_signature_safe": sanitize_filename(config_signature),
        "d": int(d) if d is not None else None,
        "T": int(T) if T is not None else None,
        "r": int(r) if r is not None else None,
        "r_star": int(r_star) if r_star is not None else None,
        "kappa": float(kappa) if kappa is not None else None,
        "kappa_star": float(kappa_star) if kappa_star is not None else None,
        "lambda_reg": float(lambda_reg) if lambda_reg is not None else None,
        "learning_rate": float(learning_rate) if learning_rate is not None else None,
        "n_steps": int(n_steps) if n_steps is not None else None,
        "mask_label": mask_label,
        "config_name": config_name,
        "sweep_name": sweep_name,
        "seed": seed,
        "summary_path": str(summary_path),
    }


def baseline_value(row: dict, mean_keys: list[str]) -> float | None:
    for key in mean_keys:
        value = row.get(key)
        if value is not None:
            return float(value)

    return None


def first_crossing_against_baseline(
    rows_by_ntrain: list[dict],
    learned_key: str,
    baseline_mean_keys: list[str],
) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_ntrain:
        learned = row.get(learned_key)
        baseline = baseline_value(
            row=row,
            mean_keys=baseline_mean_keys,
        )

        if learned is None or baseline is None:
            continue

        if float(learned) >= float(baseline):
            return int(row["n_train"]), float(learned), float(baseline)

    return None, None, None


def first_crossing_against_constant_baseline(
    rows_by_ntrain: list[dict],
    learned_key: str,
    baseline: float,
) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_ntrain:
        learned = row.get(learned_key)

        if learned is None:
            continue

        if float(learned) >= float(baseline):
            return int(row["n_train"]), float(learned), float(baseline)

    return None, None, None


def add_crossing(
    out: dict[str, object],
    rows_by_ntrain: list[dict],
    d: int,
    prefix: str,
    learned_key: str,
    baseline_mean_keys: list[str],
) -> None:
    ntrain, learned_value, baseline_threshold = first_crossing_against_baseline(
        rows_by_ntrain=rows_by_ntrain,
        learned_key=learned_key,
        baseline_mean_keys=baseline_mean_keys,
    )

    kappa_star = out.get("kappa_star")
    if kappa_star is not None:
        kappa_star = float(kappa_star)

    out[f"{prefix}_cross_ntrain"] = ntrain
    out[f"{prefix}_cross_alpha"] = ntrain / (d ** 2) if ntrain is not None else None
    out[f"{prefix}_cross_alpha_linear"] = ntrain / d if ntrain is not None else None
    out[f"{prefix}_cross_alpha_kappa"] = (
        ntrain / (kappa_star * (d ** 2))
        if ntrain is not None and kappa_star is not None and kappa_star > 0
        else None
    )
    out[f"{prefix}_cross_value"] = learned_value
    out[f"{prefix}_cross_baseline"] = baseline_threshold


def add_constant_crossing(
    out: dict[str, object],
    rows_by_ntrain: list[dict],
    d: int,
    prefix: str,
    learned_key: str,
    baseline: float,
) -> None:
    ntrain, learned_value, baseline_threshold = first_crossing_against_constant_baseline(
        rows_by_ntrain=rows_by_ntrain,
        learned_key=learned_key,
        baseline=baseline,
    )

    kappa_star = out.get("kappa_star")
    if kappa_star is not None:
        kappa_star = float(kappa_star)

    out[f"{prefix}_cross_ntrain"] = ntrain
    out[f"{prefix}_cross_alpha"] = ntrain / (d ** 2) if ntrain is not None else None
    out[f"{prefix}_cross_alpha_linear"] = ntrain / d if ntrain is not None else None
    out[f"{prefix}_cross_alpha_kappa"] = (
        ntrain / (kappa_star * (d ** 2))
        if ntrain is not None and kappa_star is not None and kappa_star > 0
        else None
    )
    out[f"{prefix}_cross_value"] = learned_value
    out[f"{prefix}_cross_baseline"] = baseline_threshold


def analyze_summary(summary_path: Path) -> dict | None:
    rows = read_csv_rows(summary_path)
    if not rows:
        return None

    metadata = extract_metadata(summary_path, rows)

    if metadata["d"] is None or metadata["mask_label"] is None:
        print(f"[skip] Could not infer d/mask for {summary_path}")
        return None

    rows_by_ntrain = grouped_mean_by_ntrain(rows, CROSSING_KEYS)
    if not rows_by_ntrain:
        return None

    d = int(metadata["d"])
    clt_baseline = 1.0 / math.sqrt(d)

    out: dict[str, object] = {
        **metadata,
        "clt_baseline": clt_baseline,
        "min_ntrain_available": min(row["n_train"] for row in rows_by_ntrain),
        "max_ntrain_available": max(row["n_train"] for row in rows_by_ntrain),
        "n_points": len(rows_by_ntrain),
    }

    # random PSD baseline
    add_crossing(
        out=out,
        rows_by_ntrain=rows_by_ntrain,
        d=d,
        prefix="random_psd",
        learned_key="cosine_S_S_star",
        baseline_mean_keys=[
            "random_baseline_cosine_S_S_star_mean",
            "random_baseline_cosine_S_S_star",
        ],
    )

    # CLT baseline: 1/sqrt(d)
    add_constant_crossing(
        out=out,
        rows_by_ntrain=rows_by_ntrain,
        d=d,
        prefix="clt",
        learned_key="cosine_S_S_star",
        baseline=clt_baseline,
    )

    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    preferred = [
        "config_signature",
        "config_signature_safe",
        "mask_label",
        "d",
        "T",
        "r",
        "r_star",
        "kappa",
        "kappa_star",
        "lambda_reg",
        "learning_rate",
        "n_steps",
        "seed",
        "clt_baseline",

        "random_psd_cross_ntrain",
        "random_psd_cross_alpha",
        "random_psd_cross_alpha_linear",
        "random_psd_cross_alpha_kappa",
        "random_psd_cross_value",
        "random_psd_cross_baseline",

        "clt_cross_ntrain",
        "clt_cross_alpha",
        "clt_cross_alpha_linear",
        "clt_cross_alpha_kappa",
        "clt_cross_value",
        "clt_cross_baseline",

        "min_ntrain_available",
        "max_ntrain_available",
        "n_points",
        "config_name",
        "sweep_name",
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


def aggregate_crossings(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = {}

    for row in rows:
        config_signature = row.get("config_signature")
        mask_label = row.get("mask_label")
        d = row.get("d")

        if config_signature is None or mask_label is None or d is None:
            continue

        grouped.setdefault((str(config_signature), str(mask_label), int(d)), []).append(row)

    metric_keys = [
        "random_psd_cross_ntrain",
        "random_psd_cross_alpha",
        "random_psd_cross_alpha_linear",
        "random_psd_cross_alpha_kappa",

        "clt_cross_ntrain",
        "clt_cross_alpha",
        "clt_cross_alpha_linear",
        "clt_cross_alpha_kappa",
    ]

    out = []

    for (config_signature, mask_label, d), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        item: dict[str, object] = {
            "config_signature": config_signature,
            "config_signature_safe": sanitize_filename(config_signature),
            "mask_label": mask_label,
            "d": d,
            "n_sweeps": len(group),

            "T": one_or_mixed(group, "T"),
            "r": one_or_mixed(group, "r"),
            "r_star": one_or_mixed(group, "r_star"),
            "kappa": one_or_mixed(group, "kappa"),
            "kappa_star": one_or_mixed(group, "kappa_star"),
            "lambda_reg": one_or_mixed(group, "lambda_reg"),
            "learning_rate": one_or_mixed(group, "learning_rate"),
            "n_steps": one_or_mixed(group, "n_steps"),
        }

        for key in metric_keys:
            values = [row[key] for row in group if row.get(key) is not None]
            if values:
                item[f"{key}_mean"] = float(np.mean(values))
                item[f"{key}_std"] = float(np.std(values))
                item[f"{key}_min"] = float(np.min(values))
                item[f"{key}_max"] = float(np.max(values))
                item[f"{key}_count"] = int(len(values))
            else:
                item[f"{key}_mean"] = None
                item[f"{key}_std"] = None
                item[f"{key}_min"] = None
                item[f"{key}_max"] = None
                item[f"{key}_count"] = 0

        out.append(item)

    return out


def plot_curve(
    rows: list[dict],
    x_key: str,
    y_mean_key: str,
    y_std_key: str,
    label: str,
    ax,
    line_alpha: float = 0.7,
    marker: str = "o",
) -> bool:
    xs = []
    means = []
    stds = []

    for row in rows:
        mean = row.get(y_mean_key)
        if mean is None:
            continue

        xs.append(float(row[x_key]))
        means.append(float(mean))
        stds.append(float(row.get(y_std_key) or 0.0))

    if not xs:
        return False

    xs_np = np.asarray(xs, dtype=float)
    means_np = np.asarray(means, dtype=float)
    stds_np = np.asarray(stds, dtype=float)

    order = np.argsort(xs_np)
    xs_np = xs_np[order]
    means_np = means_np[order]
    stds_np = stds_np[order]

    ax.plot(
        xs_np,
        means_np,
        marker=marker,
        linewidth=3,
        markersize=13,
        label=label,
        alpha=line_alpha,
    )

    if np.any(stds_np > 0):
        ax.fill_between(
            xs_np,
            means_np - stds_np,
            means_np + stds_np,
            alpha=0.12,
        )

    return True


def plot_crossings_for_mask(
    rows: list[dict],
    mask_label: str,
    output_dir: Path,
    prefix: str,
    baseline_label: str,
    filename_tag: str,
    title_metadata: str,
) -> None:
    mask_rows = [row for row in rows if row.get("mask_label") == mask_label]
    if not mask_rows:
        return

    # n_cross over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = plot_curve(
        rows=mask_rows,
        x_key="d",
        y_mean_key=f"{prefix}_cross_ntrain_mean",
        y_std_key=f"{prefix}_cross_ntrain_std",
        label=mask_label,
        ax=ax,
    )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}$")
        ax.set_title(
            f"Minimum sample size to beat the {baseline_label}\n"
            f"{mask_label}\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_ntrain_vs_d__{filename_tag}__{mask_label}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

    # n_cross / d^2 over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = plot_curve(
        rows=mask_rows,
        x_key="d",
        y_mean_key=f"{prefix}_cross_alpha_mean",
        y_std_key=f"{prefix}_cross_alpha_std",
        label=mask_label,
        ax=ax,
    )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha = n_{\mathrm{train}}/d^2$")
        ax.set_title(
            f"Minimum quadratically normalized sample size to beat the {baseline_label}\n"
            f"{mask_label}, with " + r"$\alpha = n_{\mathrm{train}}/d^2$" + "\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_vs_d__{filename_tag}__{mask_label}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

    # n_cross / d over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = plot_curve(
        rows=mask_rows,
        x_key="d",
        y_mean_key=f"{prefix}_cross_alpha_linear_mean",
        y_std_key=f"{prefix}_cross_alpha_linear_std",
        label=mask_label,
        ax=ax,
    )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$")
        ax.set_title(
            f"Minimum linearly normalized sample size to beat the {baseline_label}\n"
            f"{mask_label}, with " + r"$\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$" + "\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_linear_vs_d__{filename_tag}__{mask_label}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

    # n_cross/(kappa_star d^2) over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = plot_curve(
        rows=mask_rows,
        x_key="d",
        y_mean_key=f"{prefix}_cross_alpha_kappa_mean",
        y_std_key=f"{prefix}_cross_alpha_kappa_std",
        label=mask_label,
        ax=ax,
    )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}/(\kappa^\star d^2)$")
        ax.set_title(
            f"Minimum rank-normalized sample size to beat the {baseline_label}\n"
            f"{mask_label}, with " + r"$n_{\mathrm{train}}/(\kappa^\star d^2)$" + "\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_kappa_vs_d__{filename_tag}__{mask_label}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

def plot_all_masks_crossing(
    rows: list[dict],
    output_dir: Path,
    prefix: str,
    baseline_label: str,
    filename_tag: str,
    title_metadata: str,
) -> None:
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    # n_cross over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for i, mask in enumerate(masks):
        marker = MARKERS[i % len(MARKERS)]
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_ntrain_mean",
            y_std_key=f"{prefix}_cross_ntrain_std",
            label=mask,
            ax=ax,
            marker=marker,
        )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}$")
        ax.set_title(
            f"Minimum sample size to beat the {baseline_label}\n"
            "comparison across masking strategies\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_ntrain_vs_d__all_masks__{filename_tag}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

    # n_cross / d^2 over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for i, mask in enumerate(masks):
        marker = MARKERS[i % len(MARKERS)]
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_alpha_mean",
            y_std_key=f"{prefix}_cross_alpha_std",
            label=mask,
            ax=ax,
            marker=marker,
        )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha = n_{\mathrm{train}}/d^2$")
        ax.set_title(
            f"Minimum quadratically normalized sample size to beat the {baseline_label}\n"
            r"comparison across masking strategies, $\alpha = n_{\mathrm{train}}/d^2$"
            "\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_vs_d__all_masks__{filename_tag}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

    # n_cross / d over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for i, mask in enumerate(masks):
        marker = MARKERS[i % len(MARKERS)]
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_alpha_linear_mean",
            y_std_key=f"{prefix}_cross_alpha_linear_std",
            label=mask,
            ax=ax,
            marker=marker,
        )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$")
        ax.set_title(
            f"Minimum linearly normalized sample size to beat the {baseline_label}\n"
            r"comparison across masking strategies, $\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$"
            "\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_linear_vs_d__all_masks__{filename_tag}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

    # n_cross / (kappa_star d^2) over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for i, mask in enumerate(masks):
        marker = MARKERS[i % len(MARKERS)]
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_alpha_kappa_mean",
            y_std_key=f"{prefix}_cross_alpha_kappa_std",
            label=mask,
            ax=ax,
            marker=marker,
        )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}/(\kappa^\star d^2)$")
        ax.set_title(
            f"Minimum rank-normalized sample size to beat the {baseline_label}\n"
            r"comparison across masking strategies, $n_{\mathrm{train}}/(\kappa^\star d^2)$"
            "\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_kappa_vs_d__all_masks__{filename_tag}.png",
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_all_masks_loglog_ntrain(
    rows: list[dict],
    output_dir: Path,
    prefix: str,
    baseline_label: str,
    filename_tag: str,
    title_metadata: str,
) -> None:
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for i, mask in enumerate(masks):
        marker = MARKERS[i % len(MARKERS)]
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_ntrain_mean",
            y_std_key=f"{prefix}_cross_ntrain_std",
            label=mask,
            ax=ax,
            marker=marker,
        )

    if plotted:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}$")
        ax.set_title(
            f"Minimum sample size to beat the {baseline_label}\n"
            "comparison across masking strategies, log-log scale\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_ntrain_vs_d__all_masks_loglog__{filename_tag}.png",
            bbox_inches="tight",
        )

    plt.close(fig)


def write_group_report(aggregated_rows: list[dict], output_path: Path) -> None:
    signatures = sorted({row["config_signature"] for row in aggregated_rows})

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Baseline comparison grouping report\n")
        f.write("===================================\n\n")
        f.write(
            "Rows are grouped by config_signature, mask_label, and d.\n"
            "For each group, we report crossing thresholds for two baselines:\n"
            "1. the prior/random PSD baseline, computed from an independent random PSD matrix;\n"
            "2. the CLT baseline 1/sqrt(d).\n"
            "The config_signature excludes d, r, r_star, pca_n_components, alpha, "
            "logging fields, device, and evaluation-only fields.\n\n"
        )

        for signature in signatures:
            rows = [row for row in aggregated_rows if row["config_signature"] == signature]
            masks = sorted({row["mask_label"] for row in rows})
            ds = sorted({int(row["d"]) for row in rows})
            title_metadata = build_title_metadata(rows)

            f.write(f"config_signature:\n{signature}\n")
            f.write(f"metadata: {title_metadata}\n")
            f.write(f"masks: {', '.join(masks)}\n")
            f.write(f"d values: {ds}\n")
            f.write(f"number of grouped rows: {len(rows)}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="results/teacher-attention/iter_5000/cosine-sim",
        help="Root directory containing config folders and aggregated summary.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <root>/crossing_analysis.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else root / "crossing_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(root.rglob("summary.csv"))

    if not summary_paths:
        print(f"No summary.csv files found under {root}")
        return

    crossing_rows = []

    for summary_path in summary_paths:
        row = analyze_summary(summary_path)
        if row is not None:
            crossing_rows.append(row)

    if not crossing_rows:
        print("No valid crossing rows found.")
        return

    crossing_rows = sorted(
        crossing_rows,
        key=lambda row: (
            str(row["config_signature"]),
            str(row["mask_label"]),
            int(row["d"]),
            str(row.get("sweep_name")),
        ),
    )

    crossing_path = output_dir / "baseline_comparison_per_sweep.csv"
    write_csv(crossing_rows, crossing_path)

    aggregated_rows = aggregate_crossings(crossing_rows)
    aggregated_path = output_dir / "baseline_comparison_by_config_mask_and_d.csv"
    write_csv(aggregated_rows, aggregated_path)

    write_group_report(
        aggregated_rows=aggregated_rows,
        output_path=output_dir / "grouping_report.txt",
    )

    signatures = sorted({row["config_signature"] for row in aggregated_rows})

    for signature in signatures:
        signature_rows = [
            row for row in aggregated_rows
            if row["config_signature"] == signature
        ]

        title_metadata = build_title_metadata(signature_rows)

        signature_dir = output_dir / sanitize_filename(signature)
        signature_dir.mkdir(parents=True, exist_ok=True)

        mask_labels = sorted({
            str(row["mask_label"])
            for row in signature_rows
            if row.get("mask_label") is not None
        })

        baseline_specs = [
            {
                "prefix": "random_psd",
                "baseline_label": "random PSD baseline",
                "filename_tag": "random_psd",
            },
            {
                "prefix": "clt",
                "baseline_label": r"$1/\sqrt{d}$ baseline",
                "filename_tag": "clt",
            },
        ]

        for spec in baseline_specs:
            baseline_dir = signature_dir / f"{spec['filename_tag']}_crossings"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            for mask_label in mask_labels:
                plot_crossings_for_mask(
                    rows=signature_rows,
                    mask_label=mask_label,
                    output_dir=baseline_dir,
                    prefix=spec["prefix"],
                    baseline_label=spec["baseline_label"],
                    filename_tag=spec["filename_tag"],
                    title_metadata=title_metadata,
                )

            plot_all_masks_crossing(
                rows=signature_rows,
                output_dir=baseline_dir,
                prefix=spec["prefix"],
                baseline_label=spec["baseline_label"],
                filename_tag=spec["filename_tag"],
                title_metadata=title_metadata,
            )

            plot_all_masks_loglog_ntrain(
                rows=signature_rows,
                output_dir=baseline_dir,
                prefix=spec["prefix"],
                baseline_label=spec["baseline_label"],
                filename_tag=spec["filename_tag"],
                title_metadata=title_metadata,
            )

    print(f"[done] Wrote per-sweep baseline comparisons to: {crossing_path}")
    print(f"[done] Wrote aggregated baseline comparisons to: {aggregated_path}")
    print(f"[done] Wrote grouping report to: {output_dir / 'grouping_report.txt'}")
    print(f"[done] Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()