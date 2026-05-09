from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 28,
    "axes.grid": False,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
})


RANDOM_PSD_BASELINE_COLOR = "red"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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


def get_int(row: dict, key: str) -> int | None:
    value = get_float(row, key)
    if value is None:
        return None
    return int(round(value))


def sanitize_filename(s: str, max_len: int = 100) -> str:
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


def format_float_for_title(x: float | int | str | None) -> str:
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    return f"{float(x):.6g}"


def build_title_metadata(rows: list[dict[str, Any]]) -> str:
    """
    Build compact title metadata from one config-signature group.
    Assumes these values are constant within the group.
    """

    def unique_non_none(key: str) -> list[Any]:
        values = []
        for row in rows:
            value = row.get(key)
            if value is not None and value not in values:
                values.append(value)
        return values

    def one_or_mixed(key: str) -> Any:
        values = unique_non_none(key)
        if len(values) == 1:
            return values[0]
        if len(values) == 0:
            return None
        return "mixed"

    kappa_star = one_or_mixed("kappa_star")
    T = one_or_mixed("T")
    beta_star = one_or_mixed("beta_star")
    sigma_star = one_or_mixed("sigma_star")
    lambda_reg = one_or_mixed("lambda_reg")
    n_steps = one_or_mixed("n_steps")

    return (
        rf"$\kappa^\star = {format_float_for_title(kappa_star)}$, "
        rf"$T = {format_float_for_title(T)}$, "
        rf"$\beta^\star = {format_float_for_title(beta_star)}$, "
        rf"$\sigma^\star = {format_float_for_title(sigma_star)}$, "
        rf"$\lambda = {format_float_for_title(lambda_reg)}$, "
        rf"iters $= {format_float_for_title(n_steps)}$"
    )


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


def resolve_r_star(raw_r_star: Any, d: int | None) -> int | None:
    if raw_r_star is None:
        if d is None:
            return None
        return int(d)

    if isinstance(raw_r_star, str):
        if raw_r_star.lower() == "d":
            if d is None:
                return None
            return int(d)
        return int(float(raw_r_star))

    return int(raw_r_star)


def parse_int_from_folder(name: str, pattern: str) -> int | None:
    match = re.search(pattern, name)
    if match is None:
        return None
    return int(match.group(1))


def parse_float_from_folder(name: str, pattern: str) -> float | None:
    match = re.search(pattern, name)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def find_first_run_config(sweep_dir: Path) -> dict[str, Any] | None:
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


def infer_metadata(summary_path: Path, rows: list[dict[str, str]]) -> dict[str, Any] | None:
    """
    Infer d, r, r_star, kappa_star, mask_label, and config signature for one summary.csv.
    """
    sweep_dir = summary_path.parent
    metadata_path = sweep_dir / "sweep_config.json"
    metadata = read_json(metadata_path)

    base_config = None
    if metadata is not None:
        base_config = metadata.get("base_config")

    if base_config is None:
        base_config = find_first_run_config(sweep_dir)

    config_name = sweep_dir.parent.name
    sweep_name = sweep_dir.name

    d = None
    r = None
    r_star = None
    kappa = None
    kappa_star = None
    masking_strategy = None
    masks_per_sample = None
    T = None
    beta = None
    beta_star = None
    lambda_reg = None
    learning_rate = None
    n_steps = None
    teacher_init = None
    sigma_star = None
    normalize_sqrt_d = None
    dtype = None
    data_model = None
    mask_value = None

    # 1. Prefer base_config.
    if base_config is not None:
        data_cfg = base_config.get("data", {})
        model_cfg = base_config.get("model", {})
        teacher_cfg = base_config.get("teacher", {})
        training_cfg = base_config.get("training", {})

        d = data_cfg.get("d")
        T = data_cfg.get("T")
        masking_strategy = data_cfg.get("masking_strategy")
        masks_per_sample = data_cfg.get("masks_per_sample")
        data_model = data_cfg.get("data_model")
        mask_value = data_cfg.get("mask_value")

        r = model_cfg.get("r")
        beta = model_cfg.get("beta")
        normalize_sqrt_d = model_cfg.get("normalize_sqrt_d")
        dtype = model_cfg.get("dtype")

        r_star = resolve_r_star(teacher_cfg.get("r_star"), int(d) if d is not None else None)
        beta_star = teacher_cfg.get("beta_star")
        teacher_init = teacher_cfg.get("init")
        sigma_star = teacher_cfg.get("sigma_star")

        lambda_reg = training_cfg.get("lambda_reg")
        learning_rate = training_cfg.get("learning_rate")
        n_steps = training_cfg.get("n_steps")

    # 2. Fall back to summary rows.
    if rows:
        first = rows[0]

        if d is None:
            d = get_int(first, "d")
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

    # 3. Fall back to folder name.
    if d is None:
        d = parse_int_from_folder(config_name, r"_d(\d+)")
    if r is None:
        r = parse_int_from_folder(config_name, r"_r_(\d+)")
    if r_star is None:
        r_star = parse_int_from_folder(config_name, r"_rstar_(\d+)")
    if lambda_reg is None:
        lambda_reg = parse_float_from_folder(config_name, r"_lambda([0-9p]+)")

    if d is not None:
        d = int(d)
    if r is not None:
        r = int(r)
    if r_star is not None:
        r_star = int(r_star)

    if d is None:
        print(f"[skip] Could not infer d for {summary_path}")
        return None

    if r is not None:
        kappa = float(r) / float(d)

    if r_star is not None:
        kappa_star = float(r_star) / float(d)

    if kappa_star is None:
        print(f"[skip] Could not infer kappa_star for {summary_path}")
        return None

    mask_label = format_mask_label(masking_strategy, masks_per_sample)
    if mask_label is None:
        mask_label = parse_mask_from_name(config_name)

    if mask_label is None:
        print(f"[skip] Could not infer mask label for {summary_path}")
        return None

    teacher_init_canon, sigma_star_canon = group_teacher_signature(
        teacher_init=teacher_init,
        sigma_star=float(sigma_star) if sigma_star is not None else None,
    )

    # This signature groups dimensions together but separates rank regimes.
    config_signature = "__".join([
        f"data_model={data_model}",
        f"T={T}",
        f"mask_value={mask_value}",
        f"teacher_init={teacher_init_canon}",
        f"sigma_star={sigma_star_canon}",
        f"beta_star={beta_star}",
        f"beta={beta}",
        f"normalize_sqrt_d={normalize_sqrt_d}",
        f"dtype={dtype}",
        f"lambda={lambda_reg}",
        f"lr={learning_rate}",
        f"n_steps={n_steps}",
        f"kappa={kappa}",
        f"kappa_star={kappa_star}",
    ])

    return {
        "config_signature": config_signature,
        "config_signature_safe": sanitize_filename(config_signature),
        "d": d,
        "T": int(T) if T is not None else None,
        "r": r,
        "r_star": r_star,
        "kappa": kappa,
        "kappa_star": kappa_star,
        "beta": float(beta) if beta is not None else None,
        "beta_star": float(beta_star) if beta_star is not None else None,
        "sigma_star": float(sigma_star) if sigma_star is not None else None,
        "lambda_reg": float(lambda_reg) if lambda_reg is not None else None,
        "learning_rate": float(learning_rate) if learning_rate is not None else None,
        "n_steps": int(n_steps) if n_steps is not None else None,
        "mask_label": mask_label,
        "config_name": config_name,
        "sweep_name": sweep_name,
        "summary_path": str(summary_path),
    }


# ---------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------

def collect_cosine_rows(root: Path) -> list[dict[str, Any]]:
    summary_paths = sorted(root.rglob("summary.csv"))
    all_rows: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        rows = read_csv_rows(summary_path)
        if not rows:
            continue

        metadata = infer_metadata(summary_path, rows)
        if metadata is None:
            continue

        for row in rows:
            n_train = get_int(row, "n_train")
            cosine = get_float(row, "cosine_S_S_star")

            if n_train is None or cosine is None:
                continue

            all_rows.append({
                **metadata,
                "n_train": n_train,
                "cosine_S_S_star": cosine,
            })

    return all_rows


def aggregate_by_d_and_ntrain(rows: list[dict[str, Any]]) -> dict[int, list[tuple[int, float, float]]]:
    """
    Return:
        d -> [(n_train, mean_cosine, std_cosine), ...]
    """
    grouped: dict[int, dict[int, list[float]]] = {}

    for row in rows:
        d = int(row["d"])
        n_train = int(row["n_train"])
        cosine = float(row["cosine_S_S_star"])

        grouped.setdefault(d, {})
        grouped[d].setdefault(n_train, [])
        grouped[d][n_train].append(cosine)

    out: dict[int, list[tuple[int, float, float]]] = {}

    for d, by_ntrain in grouped.items():
        points = []
        for n_train in sorted(by_ntrain):
            values = np.asarray(by_ntrain[n_train], dtype=float)
            points.append((
                int(n_train),
                float(np.mean(values)),
                float(np.std(values)),
            ))
        out[d] = points

    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_cosine_by_d_for_mask(
    rows: list[dict[str, Any]],
    mask_label: str,
    output_dir: Path,
    kappa_star: float,
    title_metadata: str,
) -> None:
    mask_rows = [row for row in rows if row.get("mask_label") == mask_label]
    if not mask_rows:
        return

    grouped = aggregate_by_d_and_ntrain(mask_rows)
    if not grouped:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Plot 1: no baseline
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for d in sorted(grouped):
        points = grouped[d]
        xs = np.asarray([p[0] for p in points], dtype=float)
        means = np.asarray([p[1] for p in points], dtype=float)
        stds = np.asarray([p[2] for p in points], dtype=float)

        ax.plot(
            xs,
            means,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=fr"$d={d}$",
        )

        if np.any(stds > 0):
            ax.fill_between(xs, means - stds, means + stds, alpha=0.12)

        plotted = True

    if plotted:
        ax.set_xlabel(r"$n_{\mathrm{train}}$")
        ax.set_ylabel(r"$\cos(S,S^\star)$")
        ax.set_title(
            f"Cosine similarity vs $n_{{\\mathrm{{train}}}}$: "
            f"{mask_label}\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True, ncol=2)
        fig.tight_layout()

        fig.savefig(
            output_dir / f"cosine_vs_ntrain_by_d__{mask_label}.png",
            bbox_inches="tight",
        )

    plt.close(fig)

    # ------------------------------------------------------------
    # Plot 2: with random PSD formula baseline
    # ------------------------------------------------------------
    random_psd_baseline = float(kappa_star) / (1.0 + float(kappa_star))

    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for d in sorted(grouped):
        points = grouped[d]
        xs = np.asarray([p[0] for p in points], dtype=float)
        means = np.asarray([p[1] for p in points], dtype=float)
        stds = np.asarray([p[2] for p in points], dtype=float)

        ax.plot(
            xs,
            means,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=fr"$d={d}$",
        )

        if np.any(stds > 0):
            ax.fill_between(xs, means - stds, means + stds, alpha=0.12)

        plotted = True

    if plotted:
        ax.axhline(
            random_psd_baseline,
            color=RANDOM_PSD_BASELINE_COLOR,
            linestyle="dashed",
            linewidth=2.8,
            alpha = 0.7,
            label=(
                rf"$\kappa^\star/(1+\kappa^\star)$"
            ),
        )

        ax.set_xlabel(r"$n_{\mathrm{train}}$")
        ax.set_ylabel(r"$\cos(S,S^\star)$")
        ax.set_title(
            f"Cosine similarity vs $n_{{\\mathrm{{train}}}}$: "
            f"{mask_label}, with random PSD baseline\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True, ncol=2)
        fig.tight_layout()

        fig.savefig(
            output_dir / f"cosine_vs_ntrain_by_d__{mask_label}__with_random_psd_baseline.png",
            bbox_inches="tight",
        )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing summary.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <root>/cosine_by_d_curves.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else root / "cosine_by_d_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_cosine_rows(root)

    if not rows:
        print(f"No valid cosine rows found under {root}")
        return

    signatures = sorted({str(row["config_signature"]) for row in rows})

    for signature in signatures:
        signature_rows = [row for row in rows if str(row["config_signature"]) == signature]
        signature_safe = sanitize_filename(signature)

        signature_dir = output_dir / signature_safe
        signature_dir.mkdir(parents=True, exist_ok=True)

        kappas_star = sorted({
            float(row["kappa_star"])
            for row in signature_rows
            if row.get("kappa_star") is not None
        })

        if len(kappas_star) != 1:
            print(
                f"[skip] Expected one kappa_star in signature group, got {kappas_star} "
                f"for signature {signature}"
            )
            continue

        kappa_star = kappas_star[0]

        mask_labels = sorted({
            str(row["mask_label"])
            for row in signature_rows
            if row.get("mask_label") is not None
        })

        title_metadata = build_title_metadata(signature_rows)

        for mask_label in mask_labels:
            plot_cosine_by_d_for_mask(
                rows=signature_rows,
                mask_label=mask_label,
                output_dir=signature_dir,
                kappa_star=kappa_star,
                title_metadata=title_metadata,
            )

    print(f"[done] Read {len(rows)} cosine rows")
    print(f"[done] Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()