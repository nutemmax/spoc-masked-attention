# scripts/correlation_cosine_risk.py
"""
Compute correlations between teacher recovery and functional generalisation.

For each masking scheme, this script saves one CSV file containing, for every
(d, kappa_star) pair:

    Spearman corr(cos(S,S_star), -population_risk)
    Pearson  corr(cos(S,S_star), -population_risk)

Run after aggregate_teacher_attention_sweep.py.

Usage:
    python scripts/correlation_cosine_risk.py \
        --root /home/anastass/spoc-masked-attention/results/collective \
        --output-dir /home/anastass/spoc-masked-attention/results/analysis/risk_analysis/correlations
"""

from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossing_utils import (
    get_float,
    get_int,
    has_kappa_star_ancestor,
    infer_metadata,
    read_csv_rows,
    sanitize_filename,
)


COLLECT_KEYS = [
    "cosine_S_S_star",
    "population_risk",
]


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    """
    Return ranks with average rank for ties.
    Equivalent to scipy.stats.rankdata(method='average'), but avoids scipy dependency.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)

    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1

        # ranks are 1-indexed
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return None

    x_std = float(np.std(x))
    y_std = float(np.std(y))

    if x_std == 0.0 or y_std == 0.0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None

    rx = rankdata_average_ties(np.asarray(x, dtype=float))
    ry = rankdata_average_ties(np.asarray(y, dtype=float))

    return pearson_corr(rx, ry)



def aggregate_by_n_train(group_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in group_rows:
        n = int(row["n_train"])
        grouped[n]["cosine_S_S_star"].append(float(row["cosine_S_S_star"]))
        grouped[n]["population_risk"].append(float(row["population_risk"]))

    out = []
    for n in sorted(grouped):
        out.append({
            "n_train": n,
            "cosine_S_S_star": float(np.mean(grouped[n]["cosine_S_S_star"])),
            "population_risk": float(np.mean(grouped[n]["population_risk"])),
        })

    return out



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

        for row in rows:
            n_train = get_int(row, "n_train")
            if n_train is None:
                continue

            record: dict[str, Any] = {
                **metadata,
                "n_train": n_train,
                "summary_path": str(summary_path),
            }

            for key in COLLECT_KEYS:
                record[key] = get_float(row, key)

            if (
                record.get("cosine_S_S_star") is None
                or record.get("population_risk") is None
            ):
                continue

            all_rows.append(record)

    return all_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_correlations(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Returns:
        mask_label -> list of rows, one row per (config_signature, d, kappa_star)
    """
    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (
            str(row["config_signature"]),
            str(row["mask_label"]),
            int(row["d"]),
            float(row["kappa_star"]),
        )
        grouped[key].append(row)

    by_mask: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for (config_signature, mask_label, d, kappa_star), group_rows in sorted(grouped.items()):
        trajectory = aggregate_by_n_train(group_rows)

        cosine = np.array([float(r["cosine_S_S_star"]) for r in trajectory], dtype=float)
        neg_pop_risk = np.array([-float(r["population_risk"]) for r in trajectory], dtype=float)
        pop_risk = np.array([float(r["population_risk"]) for r in trajectory], dtype=float)

        spearman = spearman_corr(cosine, neg_pop_risk)
        pearson = pearson_corr(cosine, neg_pop_risk)

        n_train_values = [int(r["n_train"]) for r in trajectory]

        out_row: dict[str, Any] = {
            "config_signature": config_signature,
            "mask_label": mask_label,
            "d": d,
            "kappa_star": kappa_star,
            "n_points": len(trajectory),
            "n_raw_rows": len(group_rows),
            "n_train_min": min(n_train_values),
            "n_train_max": max(n_train_values),
            "cosine_min": float(np.min(cosine)),
            "cosine_max": float(np.max(cosine)),
            "population_risk_min": float(np.min(pop_risk)),
            "population_risk_max": float(np.max(pop_risk)),
            "spearman_corr_cosine_neg_pop_risk": spearman,
            "pearson_corr_cosine_neg_pop_risk": pearson,
        }

        by_mask[mask_label].append(out_row)

    return by_mask


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
        help="Output directory. Defaults to <root>/analysis/risk_analysis/correlations.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else root / "analysis" / "risk_analysis" / "correlations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(root)
    print(f"Loaded {len(rows)} valid rows.")

    if not rows:
        print("No valid rows found.")
        return

    by_mask = compute_correlations(rows)

    for mask_label, mask_rows in sorted(by_mask.items()):
        safe_mask = sanitize_filename(mask_label)
        out_path = output_dir / f"correlations_cosine_pop_risk__{safe_mask}.csv"
        write_csv(mask_rows, out_path)
        print(f"Saved {len(mask_rows)} rows -> {out_path}")

    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()