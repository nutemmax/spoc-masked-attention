from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


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


def parse_float_token(token: str | None) -> float | None:
    if token is None:
        return None
    return float(token.replace("p", "."))


def parse_from_name(name: str, pattern: str) -> str | None:
    match = re.search(pattern, name)
    if match is None:
        return None
    return match.group(1)


def format_float_token(x: float) -> str:
    return f"{float(x):.6g}".replace(".", "p").replace("-", "m")


def format_float_for_title(x: float | int | str | None) -> str:
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    return f"{float(x):.6g}"


def sanitize_filename(s: str, max_len: int = 120) -> str:
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
    # digest = hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:10]
    # if len(cleaned) <= max_len:
    #     return f"{cleaned}__{digest}"
    # return f"{cleaned[:max_len]}__{digest}"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len]

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

def resolve_r_star(raw_r_star: Any, d: int | None) -> int | None:
    if raw_r_star is None:
        return int(d) if d is not None else None

    if isinstance(raw_r_star, str):
        if raw_r_star.lower() == "d":
            return int(d) if d is not None else None
        return int(float(raw_r_star))

    return int(raw_r_star)

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



def parse_mask_from_name(name: str) -> str | None:
    known_masks = ["maskrandom_k", "maskmulti_k", "maskrandom", "maskall", "masklast"]

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
    k = 1 if masks_per_sample is None else int(masks_per_sample)

    if masking_strategy == "random":
        return "maskrandom" if k == 1 else f"maskrandom_k{k}"
    if masking_strategy == "k_random":
        return f"maskrandom_k{k}"
    if masking_strategy == "multi_random":
        return f"maskmulti_k{k}"
    if masking_strategy == "all":
        return "maskall"
    if masking_strategy == "last":
        return "masklast"

    return masking_strategy


def unique_non_none(rows: list[dict], key: str) -> list[object]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and value not in values:
            values.append(value)
    return values


def one_or_mixed(rows: list[dict], key: str) -> object | None:
    values = unique_non_none(rows, key)
    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        return None
    return "mixed"


def first_unique(rows: list[dict[str, Any]], key: str) -> Any:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and value not in values:
            values.append(value)

    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        return None
    return "mixed"


def grouped_mean_by_ntrain(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, float | int | None]]:
    grouped: dict[int, dict[str, list[float]]] = {}

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
        item: dict[str, float | int | None] = {"n_train": n_train}
        for key, values in grouped[n_train].items():
            item[key] = float(np.mean(values)) if values else None
        out.append(item)

    return out


def find_config_folder_name(summary_path: Path) -> str:
    """
    Expected structure:
    root/kappa_star_x/config_folder/job_folder/summary.csv
    Here summary_path.parent is the job folder.
    summary_path.parent.parent is the config folder.
    """
    return summary_path.parent.parent.name


def infer_kappa_from_parent_dirs(summary_path: Path) -> float | None:
    for part in summary_path.parts:
        match = re.match(r"kappa_star_([0-9p]+)$", part)
        if match is not None:
            return parse_float_token(match.group(1))
    return None

def has_kappa_star_ancestor(path: Path) -> bool:
    return any(re.match(r"^kappa_star_[0-9p]+$", part) is not None for part in path.parts)
