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
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len]


def resolve_r_star(raw_r_star: Any, d: int | None) -> int | None:
    if raw_r_star is None:
        return int(d) if d is not None else None

    if isinstance(raw_r_star, str):
        if raw_r_star.lower() == "d":
            return int(d) if d is not None else None
        return int(float(raw_r_star))

    return int(raw_r_star)


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
