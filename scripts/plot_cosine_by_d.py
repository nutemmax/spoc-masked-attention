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

from crossing_utils import (
    build_title_metadata,
    get_float,
    get_int,
    read_csv_rows,
    sanitize_filename,
    infer_metadata
)


RANDOM_PSD_BASELINE_COLOR = "red"

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
    random_psd_baseline = float(kappa_star) / (1.0 + float(kappa_star))
    
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
    # ------------------------------------------------------------
    # Plot 3: x-axis = n_train / d
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for d in sorted(grouped):
        points = grouped[d]
        xs = np.asarray([p[0] / d for p in points], dtype=float)
        means = np.asarray([p[1] for p in points], dtype=float)
        stds = np.asarray([p[2] for p in points], dtype=float)

        ax.plot(xs, means, marker="o", linewidth=2.5, markersize=7, label=fr"$d={d}$")
        if np.any(stds > 0):
            ax.fill_between(xs, means - stds, means + stds, alpha=0.12)
        plotted = True

    if plotted:
        ax.axhline(
            random_psd_baseline,
            color=RANDOM_PSD_BASELINE_COLOR,
            linestyle="dashed",
            linewidth=2.8,
            alpha=0.7,
            label=rf"$\kappa^\star/(1+\kappa^\star)$",
        )
        ax.set_xlabel(r"$n_{\mathrm{train}} / d$")
        ax.set_ylabel(r"$\cos(S,S^\star)$")
        ax.set_title(
            f"Cosine similarity vs $n_{{\\mathrm{{train}}}}/d$: "
            f"{mask_label}, with random PSD baseline\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True, ncol=2)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"cosine_vs_ntrain_over_d_by_d__{mask_label}__with_random_psd_baseline.png",
            bbox_inches="tight",
        )

    plt.close(fig)

    # ------------------------------------------------------------
    # Plot 4: x-axis = n_train / d^2
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))

    plotted = False
    for d in sorted(grouped):
        points = grouped[d]
        xs = np.asarray([p[0] / (d ** 2) for p in points], dtype=float)
        means = np.asarray([p[1] for p in points], dtype=float)
        stds = np.asarray([p[2] for p in points], dtype=float)

        ax.plot(xs, means, marker="o", linewidth=2.5, markersize=7, label=fr"$d={d}$")
        if np.any(stds > 0):
            ax.fill_between(xs, means - stds, means + stds, alpha=0.12)
        plotted = True

    if plotted:
        ax.axhline(
            random_psd_baseline,
            color=RANDOM_PSD_BASELINE_COLOR,
            linestyle="dashed",
            linewidth=2.8,
            alpha=0.7,
            label=rf"$\kappa^\star/(1+\kappa^\star)$",
        )
        ax.set_xlabel(r"$n_{\mathrm{train}} / d^2$")
        ax.set_ylabel(r"$\cos(S,S^\star)$")
        ax.set_title(
            f"Cosine similarity vs $n_{{\\mathrm{{train}}}}/d^2$: "
            f"{mask_label}, with random PSD baseline\n"
            f"{title_metadata}"
        )
        ax.legend(frameon=True, ncol=2)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"cosine_vs_ntrain_over_d2_by_d__{mask_label}__with_random_psd_baseline.png",
            bbox_inches="tight",
        )
        # Zoomed version: x in [0, 10]
        ax.set_xlim(0, 10)
        fig.savefig(
            output_dir / f"cosine_vs_ntrain_over_d2_by_d__{mask_label}__with_random_psd_baseline__zoom0_10.png",
            bbox_inches="tight",
        )

        # Zoomed version: x in [0, 20]
        ax.set_xlim(0, 20)
        fig.savefig(
            output_dir / f"cosine_vs_ntrain_over_d2_by_d__{mask_label}__with_random_psd_baseline__zoom0_20.png",
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