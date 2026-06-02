"""
plot_cosine_mean_over_d_by_kappa.py

For each masking scheme, plot the mean cosine similarity over dimensions as a
function of either:

    n_train / d^2

or:

    n_train / (kappa_star d^2)

Each curve corresponds to one kappa_star value.

For a fixed kappa_star and masking scheme, each dimension d gives one cosine
curve against the chosen normalized sample size. At each x-value, the script
averages over the dimensions that have data available at that x-value. Therefore,
at small x the mean usually uses all dimensions, while at larger x it may use
fewer dimensions.

Outputs:
    cosine_mean_over_d_by_kappa__{mask_label}.png
    cosine_mean_over_d_by_kappa__{mask_label}__zoom0_10.png
    cosine_mean_over_d_by_kappa__{mask_label}__zoom0_20.png

    cosine_mean_over_d_by_kappa__{mask_label}__x_kappa_d2.png
    cosine_mean_over_d_by_kappa__{mask_label}__x_kappa_d2__zoom0_10.png
    cosine_mean_over_d_by_kappa__{mask_label}__x_kappa_d2__zoom0_20.png

Usage:
    PYTHONPATH=/home/anastass/spoc-masked-attention/scripts \
    python /home/anastass/spoc-masked-attention/scripts/plot_cosine_mean_by_d_over_kappa.py \
        --root /home/anastass/spoc-masked-attention/results/collective \
        --output-dir /home/anastass/spoc-masked-attention/results/collective/cosine_mean_by_kappa
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crossing_utils_old import (
    format_float_for_title,
    get_float,
    get_int,
    infer_metadata,
    one_or_mixed,
    read_csv_rows,
)

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

KAPPA_COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def build_kappa_comparison_title_metadata(rows: list[dict[str, Any]]) -> str:
    """
    Build compact title metadata for plots comparing several kappa_star values.
    kappa_star is omitted because it is represented by the plotted curves.
    """

    T = one_or_mixed(rows, "T")
    beta_star = one_or_mixed(rows, "beta_star")
    sigma_star = one_or_mixed(rows, "sigma_star")
    # lambda_reg = one_or_mixed(rows, "lambda_reg")
    lambda_reg = 0.05
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


def collect_cosine_rows(root: Path) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []

    for summary_path in sorted(root.rglob("summary.csv")):
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


def aggregate_mean_over_d(
    rows: list[dict[str, Any]],
    x_normalization: str,
) -> dict[float, list[tuple[float, float, float, int]]]:
    """
    Returns:
        kappa_star -> sorted list of (x, mean_cosine, std_cosine, n_dims)

    x_normalization:
        "d2"        -> x = n_train / d^2
        "kappa_d2" -> x = n_train / (kappa_star d^2)
    """

    bucket: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        kappa_star = float(row["kappa_star"])
        d = int(row["d"])
        n_train = int(row["n_train"])
        cosine = float(row["cosine_S_S_star"])

        if kappa_star <= 0:
            continue

        if x_normalization == "d2":
            x = n_train / (d ** 2)
        elif x_normalization == "kappa_d2":
            x = n_train / (kappa_star * d ** 2)
        else:
            raise ValueError(f"Unknown x_normalization: {x_normalization}")

        bucket[kappa_star][x].append(cosine)

    out: dict[float, list[tuple[float, float, float, int]]] = {}

    for kappa_star in sorted(bucket):
        points = []

        for x in sorted(bucket[kappa_star]):
            vals = np.asarray(bucket[kappa_star][x], dtype=float)

            points.append((
                float(x),
                float(np.mean(vals)),
                float(np.std(vals)),
                int(len(vals)),
            ))

        out[kappa_star] = points

    return out


def plot_for_mask(
    rows: list[dict[str, Any]],
    mask_label: str,
    output_dir: Path,
    title_metadata: str,
    x_normalization: str,
) -> None:
    mask_rows = [
        row for row in rows
        if row.get("mask_label") == mask_label
    ]

    if not mask_rows:
        return

    aggregated = aggregate_mean_over_d(
        rows=mask_rows,
        x_normalization=x_normalization,
    )

    if not aggregated:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    kappas = sorted(aggregated.keys())
    colours = KAPPA_COLOURS[: len(kappas)]

    if x_normalization == "d2":
        xlabel = r"$n_{\mathrm{train}} / d^2$"
        zoom_xlabel = r"$n_{\mathrm{train}}/d^2$"
        filename_suffix_base = ""
    elif x_normalization == "kappa_d2":
        xlabel = r"$n_{\mathrm{train}} / (\kappa^\star d^2)$"
        zoom_xlabel = r"$n_{\mathrm{train}}/(\kappa^\star d^2)$"
        filename_suffix_base = "__x_kappa_d2"
    else:
        raise ValueError(f"Unknown x_normalization: {x_normalization}")

    def draw_and_save(
        xlim: tuple[float, float] | None,
        suffix: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 10))

        for kappa_star, colour in zip(kappas, colours):
            points = aggregated[kappa_star]

            xs = np.asarray([p[0] for p in points], dtype=float)
            means = np.asarray([p[1] for p in points], dtype=float)
            stds = np.asarray([p[2] for p in points], dtype=float)
            counts = np.asarray([p[3] for p in points], dtype=int)

            if xlim is not None:
                keep = (xs >= xlim[0]) & (xs <= xlim[1])
                xs = xs[keep]
                means = means[keep]
                stds = stds[keep]
                counts = counts[keep]

            if len(xs) == 0:
                continue

            label = rf"$\kappa^\star = {kappa_star:g}$"

            ax.plot(
                xs,
                means,
                marker="o",
                linewidth=2.5,
                markersize=7,
                color=colour,
                label=label,
                alpha=0.9,
            )

            shade_mask = counts > 1
            if np.any(shade_mask):
                ax.fill_between(
                    xs[shade_mask],
                    (means - stds)[shade_mask],
                    (means + stds)[shade_mask],
                    color=colour,
                    alpha=0.15,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\cos(S, S^\star)$")

        if xlim is None:
            ax.set_title(
                rf"Mean cosine similarity over $d$: {mask_label}"
                + "\n"
                + title_metadata
            )
        else:
            ax.set_title(
                rf"Mean cosine similarity over $d$: {mask_label}, "
                rf"{zoom_xlabel} $\in [{xlim[0]:g},{xlim[1]:g}]$"
                + "\n"
                + title_metadata
            )
            ax.set_xlim(*xlim)

        ax.legend(frameon=True, ncol=1)
        fig.tight_layout()

        output_path = (
            output_dir
            / f"cosine_mean_over_d_by_kappa__{mask_label}{filename_suffix_base}{suffix}.png"
        )

        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    draw_and_save(xlim=None, suffix="")
    draw_and_save(xlim=(0.0, 10.0), suffix="__zoom0_10")
    draw_and_save(xlim=(0.0, 20.0), suffix="__zoom0_20")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean cosine over dimensions, with one curve per kappa_star."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing the kappa_star_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <root>/cosine_mean_by_kappa.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else root / "cosine_mean_by_kappa"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_cosine_rows(root)
    if not rows:
        print(f"No valid cosine rows found under {root}")
        return

    title_metadata = build_kappa_comparison_title_metadata(rows)

    mask_labels = sorted({
        str(row["mask_label"])
        for row in rows
        if row.get("mask_label") is not None
    })

    for mask_label in mask_labels:
        for x_normalization in ["d2", "kappa_d2"]:
            plot_for_mask(
                rows=rows,
                mask_label=mask_label,
                output_dir=output_dir,
                title_metadata=title_metadata,
                x_normalization=x_normalization,
            )

        print(f"[done] mask={mask_label}")

    print(f"\n[done] Read {len(rows)} cosine rows")
    print(f"[done] Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()