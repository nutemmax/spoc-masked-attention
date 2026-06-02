# scripts/crossing_utils.py NEW
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------

def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_csv(rows: list[dict[str, Any]], path: Path, preferred_keys: list[str] | None = None) -> None:
    if not rows:
        return
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    if preferred_keys:
        fieldnames = [k for k in preferred_keys if k in all_keys]
        fieldnames += sorted(k for k in all_keys if k not in fieldnames)
    else:
        fieldnames = sorted(all_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_float_token(x: float) -> str:
    """Format a float for use in filenames, replacing . with p."""
    return f"{float(x):.6g}".replace(".", "p").replace("-", "m")


def unique_non_none(rows: list[dict], key: str) -> list[Any]:
    values = []
    for row in rows:
        v = row.get(key)
        if v is not None and v not in values:
            values.append(v)
    return values


def one_or_mixed(rows: list[dict], key: str) -> object | None:
    values = unique_non_none(rows, key)
    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        return None
    return "mixed"

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


# ---------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------

def get_int(row: dict, key: str) -> int | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def get_float(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def parse_float_token(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        return float(token.replace("p", "."))
    except (TypeError, ValueError):
        return None


def parse_from_name(name: str, pattern: str) -> str | None:
    m = re.search(pattern, name)
    return m.group(1) if m else None


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------

def grouped_mean_by_ntrain(
    rows: list[dict],
    keys_to_average: list[str],
) -> list[dict]:
    """Average numeric fields over rows sharing the same n_train."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        n = get_int(row, "n_train")
        if n is not None:
            buckets[n].append(row)

    out = []
    for n_train in sorted(buckets):
        group = buckets[n_train]
        merged: dict[str, Any] = {"n_train": n_train}
        for key in keys_to_average:
            vals = [get_float(r, key) for r in group if get_float(r, key) is not None]
            merged[key] = float(sum(vals) / len(vals)) if vals else None
        out.append(merged)
    return out


def first_unique(rows: list[dict], key: str) -> Any:
    for row in rows:
        v = row.get(key)
        if v is not None:
            return v
    return None


def one_or_mixed(rows: list[dict], key: str) -> Any:
    values = {row[key] for row in rows if row.get(key) is not None}
    if len(values) == 1:
        return next(iter(values))
    if len(values) == 0:
        return None
    return "mixed"


# ---------------------------------------------------------------------
# Metadata inference from folder structure
# ---------------------------------------------------------------------

def resolve_r_star(raw: Any, d: int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.lower() == "d":
        return d
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def format_mask_label(masking_strategy: str | None, masks_per_sample: Any) -> str | None:
    if masking_strategy is None:
        return None

    s = str(masking_strategy).lower().strip()
    k = 1
    try:
        k = int(float(masks_per_sample)) if masks_per_sample is not None else 1
    except (TypeError, ValueError):
        pass

    # canonical strategy names from config
    if s in ("maskall", "all"):
        return "maskall"
    if s in ("masklast", "last"):
        return "masklast"
    if s in ("maskrandom", "random"):
        return "maskrandom" if k <= 1 else f"maskrandom_k{k}"
    if s in ("k_random", "maskrandom_k2"):
        return f"maskrandom_k{k}"
    if s in ("multi_random", "maskmulti", "maskmulti_k2", "multi"):
        return f"maskmulti_k{k}"

    # fallback: try to parse from the string itself
    return parse_mask_from_name(s)


def parse_mask_from_name(name: str) -> str | None:
    name_lower = name.lower()
    # order matters: more specific patterns first
    for mask in ["maskmulti_k2", "maskrandom_k2", "maskall", "masklast", "maskrandom"]:
        # match with word boundary: mask name followed by _ or end
        if re.search(rf"(?<![a-z]){re.escape(mask)}(?:_|$)", name_lower):
            return mask
    return None


def infer_kappa_from_parent_dirs(path: Path) -> float | None:
    for part in path.parts:
        m = re.search(r"kappa_star_(\d+p\d+|\d+)", part)
        if m:
            return float(m.group(1).replace("p", "."))
    return None


def has_kappa_star_ancestor(path: Path) -> bool:
    return any("kappa_star" in part for part in path.parts)


def find_config_folder_name(summary_path: Path) -> str:
    """Return the config-level folder name (two levels above summary.csv)."""
    return summary_path.parent.parent.name


def find_metrics_files(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_file() and p.name.startswith("metrics") and p.name.endswith(".json")
    )


def find_config_files(run_dir: Path) -> list[Path]:
    return sorted(
        p for p in run_dir.iterdir()
        if p.is_file() and p.name.startswith("config") and p.name.endswith(".json")
    )


def has_run_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False
    for subdir in path.iterdir():
        if subdir.is_dir() and find_metrics_files(subdir):
            return True
    return False


def find_sweep_dirs(root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in [root, *root.rglob("*")]:
        if has_run_subdirs(path):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(path)
    return out


# ---------------------------------------------------------------------
# Title and formatting helpers
# ---------------------------------------------------------------------

def format_float_for_title(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, str):
        return value
    try:
        f = float(value)
        if f == int(f):
            return str(int(f))
        return f"{f:g}"
    except (TypeError, ValueError):
        return str(value)


def sanitize_filename(s: str, max_len: int = 120) -> str:
    cleaned = (
        str(s)
        .replace("=", "-")
        .replace(".", "p")
        .replace("/", "-")
        .replace(" ", "_")
        .replace(":", "-")
        .replace(",", "_")
    )
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def group_teacher_signature(
    teacher_init: str | None,
    sigma_star: Any,
) -> tuple[str, str]:
    if teacher_init in (None, "standard_gaussian"):
        return "scaled_gaussian", "1.0"
    if teacher_init == "scaled_gaussian":
        try:
            s = float(sigma_star)
            return "scaled_gaussian", f"{s:g}"
        except (TypeError, ValueError):
            return "scaled_gaussian", str(sigma_star)
    return str(teacher_init), str(sigma_star)


def build_title_metadata(rows: list[dict]) -> str:
    T = one_or_mixed(rows, "T")
    kappa_star = one_or_mixed(rows, "kappa_star")
    beta_star = one_or_mixed(rows, "beta_star")
    sigma_star = one_or_mixed(rows, "sigma_star")
    lambda_reg = one_or_mixed(rows, "lambda_reg")
    learning_rate = one_or_mixed(rows, "learning_rate")
    n_steps = one_or_mixed(rows, "n_steps")
    return (
        rf"$\kappa^\star = {format_float_for_title(kappa_star)}$, "
        rf"$T = {format_float_for_title(T)}$, "
        rf"$\beta^\star = {format_float_for_title(beta_star)}$, "
        rf"$\sigma^\star = {format_float_for_title(sigma_star)}$, "
        rf"$\lambda = {format_float_for_title(lambda_reg)}$, "
        rf"iters $= {format_float_for_title(n_steps)}$"
    )


def build_kappa_comparison_title(rows: list[dict]) -> str:
    T = one_or_mixed(rows, "T")
    beta_star = one_or_mixed(rows, "beta_star")
    sigma_star = one_or_mixed(rows, "sigma_star")
    lambda_reg = one_or_mixed(rows, "lambda_reg")
    n_steps = one_or_mixed(rows, "n_steps")
    return (
        r"$"
        rf"T = {format_float_for_title(T)},\ "
        rf"\beta^\star = {format_float_for_title(beta_star)},\ "
        rf"\sigma^\star = {format_float_for_title(sigma_star)},\ "
        rf"\lambda = {format_float_for_title(lambda_reg)},\ "
        rf"\mathrm{{iters}} = {format_float_for_title(n_steps)}"
        r"$"
    )


def build_sweep_title(metric_title: str, base_config: dict | None) -> str:
    """Build a multi-line plot title from a base_config dict."""
    if base_config is None:
        return metric_title

    data_cfg = base_config.get("data", {})
    model_cfg = base_config.get("model", {})
    teacher_cfg = base_config.get("teacher", {})
    training_cfg = base_config.get("training", {})

    teacher_init = str(teacher_cfg.get("init", "NA")).replace("_", "-")
    r_star_raw = teacher_cfg.get("r_star")
    r_star = "d" if r_star_raw is None else str(r_star_raw)
    beta_star = teacher_cfg.get("beta_star")
    sigma_star = teacher_cfg.get("sigma_star")
    masking_strategy = data_cfg.get("masking_strategy", "NA")
    d = data_cfg.get("d")
    T = data_cfg.get("T")
    r = model_cfg.get("r")
    beta = model_cfg.get("beta")
    lambda_reg = training_cfg.get("lambda_reg")
    learning_rate = training_cfg.get("learning_rate")
    n_steps = training_cfg.get("n_steps")

    line1 = ", ".join(filter(None, [
        rf"$W^\star$: {teacher_init}",
        rf"$r^\star = {r_star}$",
        rf"$\beta^\star = {format_float_for_title(beta_star)}$" if beta_star is not None else None,
        rf"$\sigma^\star = {format_float_for_title(sigma_star)}$" if sigma_star is not None else None,
        f"Mask={masking_strategy}",
        rf"$\lambda = {format_float_for_title(lambda_reg)}$" if lambda_reg is not None else None,
        rf"$\beta = {format_float_for_title(beta)}$" if beta is not None else None,
    ]))

    line2 = ", ".join(filter(None, [
        rf"$d = {d}$" if d is not None else None,
        rf"$T = {T}$" if T is not None else None,
        rf"$r = {r}$" if r is not None else None,
        rf"$\eta = {format_float_for_title(learning_rate)}$" if learning_rate is not None else None,
        rf"$\mathrm{{iters}} = {n_steps}$" if n_steps is not None else None,
    ]))

    parts = [metric_title, line1]
    if line2:
        parts.append(line2)
    return "\n".join(parts)


# ---------------------------------------------------------------------
# Metadata extraction from summary.csv + sweep_config.json
# ---------------------------------------------------------------------

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


def _build_config_signature(base_config: dict | None) -> str:
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


# ---------------------------------------------------------------------
# Crossing computation helpers
# ---------------------------------------------------------------------

def first_crossing_vs_baseline(
    rows_by_ntrain: list[dict],
    learned_key: str,
    baseline_keys: list[str],
) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_ntrain:
        learned = get_float(row, learned_key)
        baseline = None
        for k in baseline_keys:
            v = get_float(row, k)
            if v is not None:
                baseline = v
                break
        if learned is None or baseline is None:
            continue
        if learned >= baseline:
            return int(row["n_train"]), learned, baseline
    return None, None, None


def first_crossing_vs_constant(
    rows_by_ntrain: list[dict],
    learned_key: str,
    baseline: float,
) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_ntrain:
        learned = get_float(row, learned_key)
        if learned is not None and learned >= baseline:
            return int(row["n_train"]), learned, baseline
    return None, None, None


def crossing_columns(
    n_cross: int | None,
    d: int,
    kappa_star: float,
    value: float | None,
    baseline: float | None,
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}_cross_ntrain": n_cross,
        f"{prefix}_cross_over_d": n_cross / d if n_cross else None,
        f"{prefix}_cross_over_d2": n_cross / d**2 if n_cross else None,
        f"{prefix}_cross_over_kappa_d2": (
            n_cross / (kappa_star * d**2)
            if n_cross and kappa_star > 0 else None
        ),
        f"{prefix}_cross_value": value,
        f"{prefix}_cross_baseline": baseline,
    }