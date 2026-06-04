"""
Run with:
python scripts/plot_loss_crossing_analysis.py \
    --root results/collective \
    --output-dir results/analysis/loss_crossing_analysis
    
"""
from __future__ import annotations
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossing_utils import (
    build_kappa_comparison_title,
    get_float,
    get_int,
    grouped_mean_by_ntrain,
    has_kappa_star_ancestor,
    infer_metadata,
    read_csv_rows,
    sanitize_filename,
    write_csv,
)

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 18,
    "figure.titlesize": 30,
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


def fname(name: str, no_title: bool) -> str:
    if not no_title:
        return name
    stem, ext = name.rsplit(".", 1)
    return f"{stem}_nt.{ext}"


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def base_config_signature(sig: str) -> str:
    parts = sig.split("__")
    kept = []
    for part in parts:
        if part.startswith("kappa="):
            continue
        if part.startswith("kappa_star="):
            continue
        kept.append(part)
    return "__".join(kept)


def grouped_mean_by_loss_terms(
    rows: list[dict[str, str]],
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        x = get_int(row, "n_train_loss_terms")
        if x is None:
            x = get_int(row, "n_train")
        if x is None:
            continue

        grouped[x]["n_train_loss_terms"].append(float(x))

        n_train = get_int(row, "n_train")
        if n_train is not None:
            grouped[x]["n_train"].append(float(n_train))

        for key in metric_keys:
            value = get_float(row, key)
            if value is not None:
                grouped[x][key].append(float(value))

    out = []
    for x in sorted(grouped):
        item: dict[str, Any] = {"n_train_loss_terms": int(x)}

        if grouped[x].get("n_train"):
            item["n_train"] = int(round(float(np.mean(grouped[x]["n_train"]))))

        for key in metric_keys:
            values = grouped[x].get(key, [])
            item[key] = float(np.mean(values)) if values else None

        out.append(item)

    return out


def first_crossing_vs_baseline_loss_terms(
    rows_by_loss_terms: list[dict[str, Any]],
    learned_key: str,
    baseline_keys: list[str],
) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_loss_terms:
        learned = row.get(learned_key)
        if learned is None:
            continue

        baseline = None
        for key in baseline_keys:
            if row.get(key) is not None:
                baseline = row[key]
                break

        if baseline is None:
            continue

        if float(learned) >= float(baseline):
            return (
                int(row["n_train_loss_terms"]),
                float(learned),
                float(baseline),
            )

    return None, None, None


def first_crossing_vs_constant_loss_terms(
    rows_by_loss_terms: list[dict[str, Any]],
    learned_key: str,
    baseline: float,
) -> tuple[int | None, float | None, float | None]:
    for row in rows_by_loss_terms:
        learned = row.get(learned_key)
        if learned is None:
            continue

        if float(learned) >= float(baseline):
            return (
                int(row["n_train_loss_terms"]),
                float(learned),
                float(baseline),
            )

    return None, None, None


def loss_crossing_columns(
    n_cross_loss: int | None,
    d: int,
    kappa_star: float,
    value_at_crossing: float | None,
    baseline_at_crossing: float | None,
    prefix: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    out[f"{prefix}_loss_cross_ntrain_loss_terms"] = n_cross_loss
    out[f"{prefix}_loss_cross_value"] = value_at_crossing
    out[f"{prefix}_loss_cross_baseline"] = baseline_at_crossing

    if n_cross_loss is None:
        out[f"{prefix}_loss_cross_over_d"] = None
        out[f"{prefix}_loss_cross_over_d2"] = None
        out[f"{prefix}_loss_cross_over_kappa_d2"] = None
        return out

    d_float = float(d)
    kappa_float = float(kappa_star)

    out[f"{prefix}_loss_cross_over_d"] = float(n_cross_loss) / d_float
    out[f"{prefix}_loss_cross_over_d2"] = float(n_cross_loss) / (d_float ** 2)
    out[f"{prefix}_loss_cross_over_kappa_d2"] = (
        float(n_cross_loss) / (kappa_float * d_float ** 2)
        if kappa_float > 0
        else None
    )

    return out


def analyze_summary(summary_path: Path) -> dict[str, Any] | None:
    rows = read_csv_rows(summary_path)
    if not rows:
        return None

    metadata = infer_metadata(summary_path, rows)
    if metadata is None:
        return None

    rows_by_loss_terms = grouped_mean_by_loss_terms(rows, CROSSING_KEYS)
    if not rows_by_loss_terms:
        return None

    d = int(metadata["d"])
    kappa_star = float(metadata["kappa_star"])
    clt_baseline = 1.0 / math.sqrt(d)

    n_random_psd, v_random_psd, b_random_psd = first_crossing_vs_baseline_loss_terms(
        rows_by_loss_terms=rows_by_loss_terms,
        learned_key="cosine_S_S_star",
        baseline_keys=[
            "random_baseline_cosine_S_S_star_mean",
            "random_baseline_cosine_S_S_star",
        ],
    )

    n_clt, v_clt, b_clt = first_crossing_vs_constant_loss_terms(
        rows_by_loss_terms=rows_by_loss_terms,
        learned_key="cosine_S_S_star",
        baseline=clt_baseline,
    )

    n_train_values = [
        row.get("n_train")
        for row in rows_by_loss_terms
        if row.get("n_train") is not None
    ]

    out: dict[str, Any] = {
        **metadata,
        "summary_path": str(summary_path),
        "clt_baseline": clt_baseline,
        "min_n_train_loss_terms_available": min(
            int(row["n_train_loss_terms"]) for row in rows_by_loss_terms
        ),
        "max_n_train_loss_terms_available": max(
            int(row["n_train_loss_terms"]) for row in rows_by_loss_terms
        ),
        "n_points": len(rows_by_loss_terms),
    }

    if n_train_values:
        out["min_n_train_unique_available"] = min(int(x) for x in n_train_values)
        out["max_n_train_unique_available"] = max(int(x) for x in n_train_values)

    out.update(
        loss_crossing_columns(
            n_cross_loss=n_random_psd,
            d=d,
            kappa_star=kappa_star,
            value_at_crossing=v_random_psd,
            baseline_at_crossing=b_random_psd,
            prefix="random_psd",
        )
    )

    out.update(
        loss_crossing_columns(
            n_cross_loss=n_clt,
            d=d,
            kappa_star=kappa_star,
            value_at_crossing=v_clt,
            baseline_at_crossing=b_clt,
            prefix="clt",
        )
    )

    return out


def aggregate_crossings(per_sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)

    for row in per_sweep_rows:
        key = (
            str(row.get("config_signature", "")),
            str(row["mask_label"]),
            int(row["d"]),
            float(row["kappa_star"]),
        )
        grouped[key].append(row)

    metric_keys = [
        "random_psd_loss_cross_ntrain_loss_terms",
        "random_psd_loss_cross_over_d",
        "random_psd_loss_cross_over_d2",
        "random_psd_loss_cross_over_kappa_d2",
        "clt_loss_cross_ntrain_loss_terms",
        "clt_loss_cross_over_d",
        "clt_loss_cross_over_d2",
        "clt_loss_cross_over_kappa_d2",
    ]

    out = []

    for (sig, mask_label, d, kappa_star), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "config_signature": sig,
            "mask_label": mask_label,
            "d": d,
            "kappa_star": kappa_star,
            "n_sweeps": len(group),
            "T": group[0].get("T"),
            "lambda_reg": group[0].get("lambda_reg"),
            "learning_rate": group[0].get("learning_rate"),
            "n_steps": group[0].get("n_steps"),
            "beta_star": group[0].get("beta_star"),
            "sigma_star": group[0].get("sigma_star"),
        }

        for key in metric_keys:
            vals = [float(row[key]) for row in group if row.get(key) is not None]
            item[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            item[f"{key}_std"] = float(np.std(vals)) if vals else None
            item[f"{key}_count"] = int(len(vals))

        out.append(item)

    return out


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

    ax.plot(
        xs_np,
        ys_np,
        marker=marker,
        linewidth=3,
        markersize=16,
        label=str(label_value),
        alpha=0.9,
    )

    if np.any(stds_np > 0):
        lower = np.maximum(ys_np - stds_np, 1e-12)
        upper = ys_np + stds_np
        ax.fill_between(xs_np, lower, upper, alpha=0.08)

    return True


def plot_all_masks_vs_d(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_prefix: str,
    baseline_label: str,
    y_metric: str,
    y_label: str,
    filename_tag: str,
    no_title: bool,
    loglog: bool = False,
) -> None:
    masks = sorted({
        str(row["mask_label"])
        for row in rows
        if row.get("mask_label") is not None
    })

    kappas = sorted({
        float(row["kappa_star"])
        for row in rows
        if row.get("kappa_star") is not None
    })

    for kappa_star in kappas:
        kappa_rows = [
            row for row in rows
            if float(row["kappa_star"]) == kappa_star
        ]

        fig, ax = plt.subplots(figsize=(14, 10))
        plotted = False

        for i, mask in enumerate(masks):
            mask_rows = [
                row for row in kappa_rows
                if row["mask_label"] == mask
            ]

            plotted |= plot_curve(
                rows=mask_rows,
                x_key="d",
                y_key=f"{baseline_prefix}_{y_metric}_mean",
                y_std_key=f"{baseline_prefix}_{y_metric}_std",
                label_key="mask_label",
                label_value=mask,
                ax=ax,
                marker=MARKERS[i % len(MARKERS)],
            )

        if not plotted:
            plt.close(fig)
            continue

        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel(r"$d$")
        ax.set_ylabel(y_label)

        if not no_title:
            suffix = " (log-log)" if loglog else ""
            ax.set_title(
                rf"{y_label} vs $d$, all masks, "
                rf"$\kappa^\star={kappa_star:g}$, {baseline_label}{suffix}"
            )

        ax.legend(frameon=True, fontsize=24)

        kstr = str(kappa_star).replace(".", "p")

        save_fig(
            fig,
            output_dir / fname(
                f"loss_{filename_tag}_vs_d__all_masks__kappa{kstr}"
                f"{'_loglog' if loglog else ''}__{baseline_prefix}.png",
                no_title,
            ),
        )

        

def plot_over_kappa_by_d(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_prefix: str,
    baseline_label: str,
    y_metric: str,
    y_label: str,
    filename_tag: str,
    no_title: bool,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    masks = sorted({str(row["mask_label"]) for row in rows if row.get("mask_label") is not None})

    for mask in masks:
        mask_rows = [row for row in rows if row["mask_label"] == mask]
        ds = sorted({int(row["d"]) for row in mask_rows if row.get("d") is not None})

        fig, ax = plt.subplots(figsize=(14, 10))
        plotted = False

        for i, d in enumerate(ds):
            d_rows = [row for row in mask_rows if int(row["d"]) == d]
            plotted |= plot_curve(
                rows=d_rows,
                x_key="kappa_star",
                y_key=f"{baseline_prefix}_{y_metric}_mean",
                y_std_key=f"{baseline_prefix}_{y_metric}_std",
                label_key="d",
                label_value=d,
                ax=ax,
                marker=MARKERS[i % len(MARKERS)],
            )

        if not plotted:
            plt.close(fig)
            continue

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel(r"$\kappa^\star$")
        ax.set_ylabel(y_label)

        if not no_title:
            scale = []
            if log_x:
                scale.append("log x")
            if log_y:
                scale.append("log y")
            scale_suffix = f" ({', '.join(scale)})" if scale else ""
            ax.set_title(f"{y_label} vs $\\kappa^\\star$, {mask}, {baseline_label}{scale_suffix}")

        ax.legend(title=r"$d$", title_fontsize=24, fontsize=22, frameon=True, ncol=2)

        save_fig(
            fig,
            output_dir / fname(
                f"loss_{filename_tag}_vs_kappa__by_d__{mask}__{baseline_prefix}.png",
                no_title,
            ),
        )


def plot_heatmap_over_d_kappa(
    rows: list[dict[str, Any]],
    output_dir: Path,
    baseline_prefix: str,
    baseline_label: str,
    y_metric: str,
    colorbar_label: str,
    filename_tag: str,
    no_title: bool,
) -> None:
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

        if not no_title:
            ax.set_title(f"{colorbar_label} over $(d, \\kappa^\\star)$, {mask}, {baseline_label}")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

        save_fig(
            fig,
            output_dir / fname(
                f"loss_heatmap_{filename_tag}_over_d_kappa__{mask}__{baseline_prefix}.png",
                no_title,
            ),
        )


def make_plots(aggregated_rows: list[dict[str, Any]], output_dir: Path, no_title: bool) -> None:
    plots_dir = output_dir / "plots_loss_terms"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_specs = [
        ("random_psd", "random PSD baseline"),
        ("clt", r"$1/\sqrt{d}$ baseline"),
    ]

    metric_specs = [
        (
            "loss_cross_ntrain_loss_terms",
            r"$n_{\mathrm{cross}}^{\mathrm{loss}}$",
            "loss_ncross",
        ),
        (
            "loss_cross_over_d",
            r"$n_{\mathrm{cross}}^{\mathrm{loss}}/d$",
            "loss_ncross_over_d",
        ),
        (
            "loss_cross_over_d2",
            r"$n_{\mathrm{cross}}^{\mathrm{loss}}/d^2$",
            "loss_ncross_over_d2",
        ),
        (
            "loss_cross_over_kappa_d2",
            r"$n_{\mathrm{cross}}^{\mathrm{loss}}/(\kappa^\star d^2)$",
            "loss_ncross_over_kappa_d2",
        ),
    ]

    for baseline_prefix, baseline_label in baseline_specs:
        baseline_dir = plots_dir / f"{baseline_prefix}_crossings"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        for y_metric, y_label, filename_tag in metric_specs:
            plot_all_masks_vs_d(
                rows=aggregated_rows,
                output_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                y_label=y_label,
                filename_tag=filename_tag,
                no_title=no_title,
                loglog=False,
            )

            plot_all_masks_vs_d(
                rows=aggregated_rows,
                output_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                y_label=y_label,
                filename_tag=filename_tag,
                no_title=no_title,
                loglog=True,
            )

            plot_over_kappa_by_d(
                rows=aggregated_rows,
                output_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                y_label=y_label,
                filename_tag=filename_tag,
                no_title=no_title,
                log_x=False,
                log_y=False,
            )

            plot_over_kappa_by_d(
                rows=aggregated_rows,
                output_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                y_label=y_label,
                filename_tag=f"loglog_{filename_tag}",
                no_title=no_title,
                log_x=True,
                log_y=True,
            )

            plot_heatmap_over_d_kappa(
                rows=aggregated_rows,
                output_dir=baseline_dir,
                baseline_prefix=baseline_prefix,
                baseline_label=baseline_label,
                y_metric=y_metric,
                colorbar_label=y_label,
                filename_tag=filename_tag,
                no_title=no_title,
            )


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
        help="Output directory. Defaults to <root>/loss_crossing_analysis.",
    )
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else root / "loss_crossing_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(
        p for p in root.rglob("summary.csv")
        if has_kappa_star_ancestor(p)
    )

    if not summary_paths:
        print(f"No summary.csv files found under {root}")
        return

    per_sweep_rows = []
    for summary_path in summary_paths:
        row = analyze_summary(summary_path)
        if row is not None:
            per_sweep_rows.append(row)

    if not per_sweep_rows:
        print("No valid loss-term crossing rows found.")
        return

    per_sweep_rows = sorted(
        per_sweep_rows,
        key=lambda r: (
            str(r["mask_label"]),
            int(r["d"]),
            float(r["kappa_star"]),
            str(r["summary_path"]),
        ),
    )

    per_sweep_path = output_dir / "loss_crossings_per_sweep.csv"
    write_csv(per_sweep_rows, per_sweep_path)

    aggregated_rows = aggregate_crossings(per_sweep_rows)
    aggregated_rows = sorted(
        aggregated_rows,
        key=lambda r: (
            str(r["mask_label"]),
            int(r["d"]),
            float(r["kappa_star"]),
        ),
    )

    aggregated_path = output_dir / "loss_crossings_aggregated.csv"
    write_csv(aggregated_rows, aggregated_path)

    base_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        base_sig = base_config_signature(str(row["config_signature"]))
        base_groups[base_sig].append(row)

    for base_sig, group_rows in sorted(base_groups.items()):
        group_dir = output_dir
        make_plots(group_rows, group_dir, args.no_title)

    print(f"[done] Wrote per-sweep rows to: {per_sweep_path}")
    print(f"[done] Wrote aggregated rows to: {aggregated_path}")
    print(f"[done] Wrote plots to: {output_dir}")


if __name__ == "__main__":
    main()