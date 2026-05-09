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


def get_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(out) or math.isinf(out):
        return None

    return out


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


def get_int(row: dict, key: str) -> int | None:
    value = get_float(row, key)
    if value is None:
        return None
    return int(round(value))


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def sanitize_filename(s: str, max_len: int = 80) -> str:
    cleaned = (
        str(s)
        .replace("=", "-")
        .replace(".", "p")
        .replace("/", "-")
        .replace(" ", "_")
        .replace(":", "-")
        .replace(",", "_")
        .replace("__", "_")
    )

    digest = hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:10]

    if len(cleaned) <= max_len:
        return f"{cleaned}__{digest}"

    return f"{cleaned[:max_len]}__{digest}"


def parse_mask_from_name(name: str) -> str | None:
    known_masks = [
        "maskrandom_k",
        "maskmulti_k",
        "maskrandom",
        "maskall",
        "masklast",
    ]

    for mask in known_masks:
        if name.startswith(mask):
            if mask in {"maskrandom_k", "maskmulti_k"}:
                match = re.match(r"(mask(?:random|multi)_k\d+)", name)
                if match is not None:
                    return match.group(1)
            return mask

    return None


def format_mask_label(masking_strategy: str | None, masks_per_sample: int | None) -> str | None:
    if masking_strategy is None:
        return None

    masking_strategy = str(masking_strategy)

    if masks_per_sample is None:
        masks_per_sample_int = 1
    else:
        masks_per_sample_int = int(masks_per_sample)

    if masking_strategy == "random":
        if masks_per_sample_int == 1:
            return "maskrandom"
        return f"maskrandom_k{masks_per_sample_int}"

    if masking_strategy == "k_random":
        return f"maskrandom_k{masks_per_sample_int}"

    if masking_strategy == "multi_random":
        return f"maskmulti_k{masks_per_sample_int}"

    if masking_strategy == "all":
        return "maskall"

    if masking_strategy == "last":
        return "masklast"

    return masking_strategy


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
        d = data_cfg.get("d", None)
        masking_strategy = data_cfg.get("masking_strategy", None)
        masks_per_sample = data_cfg.get("masks_per_sample", None)

    if d is None:
        d = extract_d_from_rows(rows)

    if d is None:
        print(
            f"[skip] Could not infer d from sweep_config.json, run config*.json, "
            f"or summary.csv for {summary_path}"
        )

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
        "mask_label": mask_label,
        "config_name": config_name,
        "sweep_name": sweep_name,
        "seed": seed,
        "summary_path": str(summary_path),
    }


def grouped_mean_by_ntrain(rows: list[dict]) -> list[dict]:
    grouped: dict[int, dict[str, list[float]]] = {}

    keys = [
        "cosine_S_S_star",
        "random_baseline_cosine_S_S_star",
        "random_baseline_cosine_S_S_star_mean",
    ]

    for row in rows:
        n_train = get_int(row, "n_train")
        if n_train is None:
            continue

        grouped.setdefault(n_train, {key: [] for key in keys})

        for key in keys:
            value = get_float(row, key)
            if value is not None:
                grouped[n_train][key].append(value)

    out = []

    for n_train in sorted(grouped):
        item = {"n_train": n_train}

        for key, values in grouped[n_train].items():
            if values:
                item[key] = float(np.mean(values))
            else:
                item[key] = None

        out.append(item)

    return out


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

    out[f"{prefix}_cross_ntrain"] = ntrain
    out[f"{prefix}_cross_alpha"] = ntrain / (d ** 2) if ntrain is not None else None
    out[f"{prefix}_cross_alpha_linear"] = ntrain / d if ntrain is not None else None
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

    out[f"{prefix}_cross_ntrain"] = ntrain
    out[f"{prefix}_cross_alpha"] = ntrain / (d ** 2) if ntrain is not None else None
    out[f"{prefix}_cross_alpha_linear"] = ntrain / d if ntrain is not None else None
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

    rows_by_ntrain = grouped_mean_by_ntrain(rows)
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
        "seed",
        "clt_baseline",

        "random_psd_cross_ntrain",
        "random_psd_cross_alpha",
        "random_psd_cross_alpha_linear",
        "random_psd_cross_value",
        "random_psd_cross_baseline",

        "clt_cross_ntrain",
        "clt_cross_alpha",
        "clt_cross_alpha_linear",
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

        "clt_cross_ntrain",
        "clt_cross_alpha",
        "clt_cross_alpha_linear",
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

    ax.plot(xs_np, means_np, marker="o", linewidth=3, markersize=11, label=label)

    if np.any(stds_np > 0):
        ax.fill_between(xs_np, means_np - stds_np, means_np + stds_np, alpha=0.15)

    return True


def plot_crossings_for_mask(
    rows: list[dict],
    mask_label: str,
    output_dir: Path,
    prefix: str,
    baseline_label: str,
    filename_tag: str,
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
        label=baseline_label,
        ax=ax,
    )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}$")
        ax.set_title(
            f"Minimum sample size to beat the {baseline_label}\n"
            f"{mask_label}"
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
        label=baseline_label,
        ax=ax,
    )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha = n_{\mathrm{train}}/d^2$")
        ax.set_title(
            f"Minimum quadratically normalized sample size to beat the {baseline_label}\n"
            f"{mask_label}, with " + r"$\alpha = n_{\mathrm{train}}/d^2$"
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
        label=baseline_label,
        ax=ax,
    )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$")
        ax.set_title(
            f"Minimum linearly normalized sample size to beat the {baseline_label}\n"
            f"{mask_label}, with " + r"$\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_linear_vs_d__{filename_tag}__{mask_label}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

def plot_all_masks_crossing(
    rows: list[dict],
    output_dir: Path,
    prefix: str,
    baseline_label: str,
    filename_tag: str,
) -> None:
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    # n_cross over d
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for mask in masks:
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_ntrain_mean",
            y_std_key=f"{prefix}_cross_ntrain_std",
            label=mask,
            ax=ax,
        )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}$")
        ax.set_title(
            f"Minimum sample size to beat the {baseline_label}\n"
            "comparison across masking strategies"
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
    for mask in masks:
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_alpha_mean",
            y_std_key=f"{prefix}_cross_alpha_std",
            label=mask,
            ax=ax,
        )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha = n_{\mathrm{train}}/d^2$")
        ax.set_title(
            f"Minimum quadratically normalized sample size to beat the {baseline_label}\n"
            r"comparison across masking strategies, $\alpha = n_{\mathrm{train}}/d^2$"
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
    for mask in masks:
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_alpha_linear_mean",
            y_std_key=f"{prefix}_cross_alpha_linear_std",
            label=mask,
            ax=ax,
        )

    if plotted:
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$")
        ax.set_title(
            f"Minimum linearly normalized sample size to beat the {baseline_label}\n"
            r"comparison across masking strategies, $\alpha_{\mathrm{lin}} = n_{\mathrm{train}}/d$"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"min_alpha_linear_vs_d__all_masks__{filename_tag}.png",
            bbox_inches="tight",
        )
    plt.close(fig)

def plot_all_masks_loglog_ntrain(
    rows: list[dict],
    output_dir: Path,
    prefix: str,
    baseline_label: str,
    filename_tag: str,
) -> None:
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for mask in masks:
        mask_rows = [row for row in rows if row.get("mask_label") == mask]
        plotted |= plot_curve(
            rows=mask_rows,
            x_key="d",
            y_mean_key=f"{prefix}_cross_ntrain_mean",
            y_std_key=f"{prefix}_cross_ntrain_std",
            label=mask,
            ax=ax,
        )

    if plotted:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$d$")
        ax.set_ylabel(r"min $n_{\mathrm{train}}$")
        ax.set_title(
            f"Minimum sample size to beat the {baseline_label}\n"
            "comparison across masking strategies, log-log scale"
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

            f.write(f"config_signature:\n{signature}\n")
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
                )

            plot_all_masks_crossing(
                rows=signature_rows,
                output_dir=baseline_dir,
                prefix=spec["prefix"],
                baseline_label=spec["baseline_label"],
                filename_tag=spec["filename_tag"],
            )

            plot_all_masks_loglog_ntrain(
                rows=signature_rows,
                output_dir=baseline_dir,
                prefix=spec["prefix"],
                baseline_label=spec["baseline_label"],
                filename_tag=spec["filename_tag"],
            )

    print(f"[done] Wrote per-sweep baseline comparisons to: {crossing_path}")
    print(f"[done] Wrote aggregated baseline comparisons to: {aggregated_path}")
    print(f"[done] Wrote grouping report to: {output_dir / 'grouping_report.txt'}")
    print(f"[done] Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()