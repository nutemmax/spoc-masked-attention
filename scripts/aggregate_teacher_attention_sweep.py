from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plots import format_float_for_title


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
    "axes.grid": False,
    "mathtext.fontset": "cm",
    "savefig.bbox": "tight",
})


def extract_datetime_from_run_name(run_name: str) -> str | None:
    parts = run_name.split("_")
    if not parts:
        return None
    last = parts[-1]
    if len(last) == 15 and last[8] == "-":
        return last
    return None


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


def collect_runs(sweep_dir: Path) -> list[dict]:
    rows = []

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir() or subdir.name == "plots":
            continue

        metrics_files = find_metrics_files(subdir)
        if not metrics_files:
            continue

        if len(metrics_files) > 1:
            print(
                f"[warn] Multiple metrics files found in {subdir}. "
                f"Using the first one: {metrics_files[0].name}"
            )

        metrics_path = metrics_files[0]

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[skip] Failed to read {metrics_path}: {e}")
            continue

        row = {
            "run_name": subdir.name,
            "experiment_number": None,
            **metrics,
            "datetime": extract_datetime_from_run_name(subdir.name),
        }
        rows.append(row)

    return rows


def collect_run_configs(sweep_dir: Path) -> list[dict]:
    configs = []

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir() or subdir.name == "plots":
            continue

        config_files = find_config_files(subdir)
        if not config_files:
            continue

        if len(config_files) > 1:
            print(
                f"[warn] Multiple config files found in {subdir}. "
                f"Using the first one: {config_files[0].name}"
            )

        config_path = config_files[0]

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[skip] Failed to read {config_path}: {e}")
            continue

        configs.append(config)

    return configs


def detect_sweep_key(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("No rows found.")

    has_ntrain = any(row.get("n_train") is not None for row in rows)
    has_alpha = any(row.get("alpha") is not None for row in rows)

    if has_ntrain:
        return "n_train"
    if has_alpha:
        return "alpha"

    raise ValueError("Could not detect sweep variable (n_train or alpha).")


def build_sweep_metadata(run_configs: list[dict], sweep_key: str) -> dict | None:
    if not run_configs:
        return None

    base_config = copy.deepcopy(run_configs[0])

    if "experiment" in base_config:
        base_config["experiment"]["save_root"] = None
        base_config["experiment"]["run_name"] = None
        base_config["experiment"]["seed"] = None

    if "training" in base_config:
        if sweep_key == "alpha":
            base_config["training"]["alpha"] = None
        elif sweep_key == "n_train":
            base_config["training"]["n_train"] = None

    seeds = sorted({
        int(cfg["experiment"]["seed"])
        for cfg in run_configs
        if cfg.get("experiment", {}).get("seed") is not None
    })

    metadata = {
        "sweep_key": sweep_key,
        "seeds": seeds,
        "base_config": base_config,
    }

    if sweep_key == "alpha":
        metadata["alphas"] = sorted({
            float(cfg["training"]["alpha"])
            for cfg in run_configs
            if cfg.get("training", {}).get("alpha") is not None
        })

    elif sweep_key == "n_train":
        metadata["n_trains"] = sorted({
            int(cfg["training"]["n_train"])
            for cfg in run_configs
            if cfg.get("training", {}).get("n_train") is not None
        })

    return metadata


def save_sweep_metadata(sweep_dir: Path, metadata: dict) -> None:
    with open(sweep_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    preferred = [
        "alpha",
        "seed",
        "n_train",
        "n_population",
        "teacher_init",
        "r_star",
        "beta_star",
        "sigma_star",
        "train_loss",
        "population_risk",
        "generalization_gap",
        "ridge_train_loss",
        "ridge_population_risk",
        "ridge_generalization_gap",
        "attention_vs_ridge_gap",
        "attention_vs_ridge_relative_improvement",
        "pca_train_loss",
        "pca_population_risk",
        "pca_generalization_gap",
        "attention_vs_pca_gap",
        "attention_vs_pca_relative_improvement",
        "pca_n_components",
        "cosine_S_S_star",
        "relative_error_S_S_star",
        "final_attention_level_error",
        "runtime_seconds",
        "runtime_per_step_seconds",
        "initial_objective",
        "final_objective",
        "best_objective",
        "objective_reduction",
        "initial_train_loss_history",
        "final_train_loss_history",
        "best_train_loss_history",
        "train_loss_reduction",
        "weight_norm",
        "W_star_norm",
        "S_trace",
        "S_top_eigenvalue",
        "S_min_eigenvalue",
        "S_R1",
        "S_effective_rank",
        "S_frobenius_norm",
        "S_star_trace",
        "S_star_top_eigenvalue",
        "S_star_min_eigenvalue",
        "S_star_R1",
        "S_star_effective_rank",
        "S_star_frobenius_norm",
        "datetime",
        "run_name",
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


def has_run_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False

    for subdir in path.iterdir():
        if not subdir.is_dir():
            continue
        if find_metrics_files(subdir):
            return True

    return False


def is_aggregated(sweep_dir: Path) -> bool:
    summary_path = sweep_dir / "summary.csv"
    plots_dir = sweep_dir / "plots"

    if not summary_path.exists():
        return False
    if not plots_dir.exists():
        return False

    return any(plots_dir.glob("*.png"))


def get_numeric(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(out) or math.isinf(out):
        return None

    return out


def grouped_mean_std(
    rows: list[dict],
    sweep_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = {}

    for row in rows:
        x = get_numeric(row, sweep_key)
        y = get_numeric(row, metric_key)

        if x is None or y is None:
            continue

        grouped.setdefault(x, []).append(y)

    if not grouped:
        return np.array([]), np.array([]), np.array([])

    xs = np.array(sorted(grouped.keys()), dtype=float)
    means = np.array([np.mean(grouped[x]) for x in xs], dtype=float)
    stds = np.array([np.std(grouped[x]) for x in xs], dtype=float)

    return xs, means, stds


def set_sweep_xlabel(ax, sweep_key: str) -> None:
    if sweep_key == "n_train":
        ax.set_xlabel(r"$n_{\mathrm{train}}$")
    elif sweep_key == "alpha":
        ax.set_xlabel(r"$\alpha$")
    else:
        ax.set_xlabel(sweep_key)


def plot_metrics_vs_sweep(
    rows: list[dict],
    sweep_key: str,
    metrics: list[tuple[str, str]],
    ylabel: str,
    title: str,
    output_path: Path,
    zoom: tuple[float | None, float | None] | None = None,
) -> bool:
    fig, ax = plt.subplots(figsize=(11, 7.5))

    plotted = False

    for metric_key, label in metrics:
        xs, means, stds = grouped_mean_std(rows, sweep_key, metric_key)
        if xs.size == 0:
            continue

        ax.plot(
            xs,
            means,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=label,
        )

        if np.any(stds > 0):
            ax.fill_between(xs, means - stds, means + stds, alpha=0.2)

        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    set_sweep_xlabel(ax, sweep_key)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    ax.legend(frameon=True)

    if zoom is not None:
        xmin, xmax = zoom
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return True


def zoom_suffix(zoom: tuple[float | None, float | None] | None) -> str:
    if zoom is None:
        return "full"

    left, right = zoom
    if left is None and right is None:
        return "full"

    left_str = "min" if left is None else str(int(left))
    right_str = "max" if right is None else str(int(right))

    return f"zoom_{left_str}_{right_str}"


def format_r_star_label(r_star) -> str:
    if r_star is None:
        return "d"
    return str(r_star)


def build_teacher_attention_sweep_title(metric_title: str, base_config: dict | None) -> str:
    if base_config is None:
        return metric_title

    data_cfg = base_config.get("data", {})
    model_cfg = base_config.get("model", {})
    teacher_cfg = base_config.get("teacher", {})
    training_cfg = base_config.get("training", {})

    teacher_init = str(teacher_cfg.get("init", "NA")).replace("_", "-")
    r_star = format_r_star_label(teacher_cfg.get("r_star"))
    beta_star = teacher_cfg.get("beta_star", None)
    sigma_star = teacher_cfg.get("sigma_star", None)

    masking_strategy = data_cfg.get("masking_strategy", "NA")
    d = data_cfg.get("d", None)
    T = data_cfg.get("T", None)
    r = model_cfg.get("r", None)
    beta = model_cfg.get("beta", None)

    lambda_reg = training_cfg.get("lambda_reg", None)
    learning_rate = training_cfg.get("learning_rate", None)
    n_steps = training_cfg.get("n_steps", None)

    line1_parts = [
        rf"$W^\star$: {teacher_init}",
        rf"$r^\star = {r_star}$",
    ]

    if beta_star is not None:
        line1_parts.append(rf"$\beta^\star = {format_float_for_title(beta_star)}$")
    if sigma_star is not None:
        line1_parts.append(rf"$\sigma^\star = {format_float_for_title(sigma_star)}$")

    line1_parts.append(f"Mask={masking_strategy}")

    if lambda_reg is not None:
        line1_parts.append(rf"$\lambda = {format_float_for_title(lambda_reg)}$")
    if beta is not None:
        line1_parts.append(rf"$\beta = {format_float_for_title(beta)}$")

    line1 = ", ".join(line1_parts)

    line2_parts = []
    if d is not None:
        line2_parts.append(rf"$d = {d}$")
    if T is not None:
        line2_parts.append(rf"$T = {T}$")
    if r is not None:
        line2_parts.append(rf"$r = {r}$")
    if learning_rate is not None:
        line2_parts.append(rf"$\eta = {format_float_for_title(learning_rate)}$")
    if n_steps is not None:
        line2_parts.append(rf"$\mathrm{{iters}} = {n_steps}$")

    line2 = ", ".join(line2_parts)

    if line2:
        return f"{metric_title}\n{line1}\n{line2}"

    return f"{metric_title}\n{line1}"


def generate_teacher_attention_summary_plots(
    rows: list[dict],
    sweep_dir: Path,
    sweep_key: str,
    base_config: dict | None = None,
    zoom_ranges: list[tuple[float | None, float | None]] | None = None,
) -> None:
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if zoom_ranges is None:
        zoom_ranges = [(None, None)]

    loss_metrics = [
        ("train_loss", "Train loss"),
        ("population_risk", "Population risk"),
        ("ridge_population_risk", "Ridge"),
        ("pca_population_risk", "PCA"),
    ]

    for zoom in zoom_ranges:
        name = f"train_vs_population_vs_ridge_vs_pca_{sweep_key}_{zoom_suffix(zoom)}.png"
        plot_metrics_vs_sweep(
            rows=rows,
            sweep_key=sweep_key,
            metrics=loss_metrics,
            ylabel="Risk",
            title=build_teacher_attention_sweep_title(
                "Train loss vs population risk vs baselines",
                base_config,
            ),
            output_path=plots_dir / name,
            zoom=zoom,
        )

    loss_attention_only = [
        ("train_loss", "Train loss"),
        ("population_risk", "Population risk"),
    ]

    for zoom in zoom_ranges:
        name = f"train_vs_population_{sweep_key}_{zoom_suffix(zoom)}.png"
        plot_metrics_vs_sweep(
            rows=rows,
            sweep_key=sweep_key,
            metrics=loss_attention_only,
            ylabel="Risk",
            title=build_teacher_attention_sweep_title(
                "Train loss vs population risk",
                base_config,
            ),
            output_path=plots_dir / name,
            zoom=zoom,
        )

    loss_no_pca = [
        ("train_loss", "Train loss"),
        ("population_risk", "Population risk"),
        ("ridge_population_risk", "Ridge"),
    ]

    for zoom in zoom_ranges:
        name = f"train_vs_population_vs_ridge_{sweep_key}_{zoom_suffix(zoom)}.png"
        plot_metrics_vs_sweep(
            rows=rows,
            sweep_key=sweep_key,
            metrics=loss_no_pca,
            ylabel="Risk",
            title=build_teacher_attention_sweep_title(
                "Train loss vs population risk vs ridge",
                base_config,
            ),
            output_path=plots_dir / name,
            zoom=zoom,
        )

    recovery_metrics = [
        ("cosine_S_S_star", r"$\cos(S,S^\star)$"),
        ("relative_error_S_S_star", r"$\|S-S^\star\|_F/\|S^\star\|_F$"),
        ("final_attention_level_error", "Attention-level error"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=recovery_metrics,
        ylabel="Value",
        title=build_teacher_attention_sweep_title(
            "Teacher recovery metrics",
            base_config,
        ),
        output_path=plots_dir / f"teacher_recovery_metrics_{sweep_key}.png",
    )

    gap_metrics = [
        ("generalization_gap", "Generalization gap"),
        ("attention_vs_ridge_gap", "Attention - Ridge"),
        ("attention_vs_pca_gap", "Attention - PCA"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=gap_metrics,
        ylabel="Gap",
        title=build_teacher_attention_sweep_title(
            "Generalization and baseline gaps",
            base_config,
        ),
        output_path=plots_dir / f"gap_metrics_{sweep_key}.png",
    )

    relative_improvement_metrics = [
        ("attention_vs_ridge_relative_improvement", "vs Ridge"),
        ("attention_vs_pca_relative_improvement", "vs PCA"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=relative_improvement_metrics,
        ylabel="Relative improvement",
        title=build_teacher_attention_sweep_title(
            "Relative improvement of attention over baselines",
            base_config,
        ),
        output_path=plots_dir / f"relative_improvement_{sweep_key}.png",
    )

    trace_metrics = [
        ("S_trace", r"$\mathrm{Tr}(S)$"),
        ("S_star_trace", r"$\mathrm{Tr}(S^\star)$"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=trace_metrics,
        ylabel="Trace",
        title=build_teacher_attention_sweep_title(
            "Trace of learned and teacher matrices",
            base_config,
        ),
        output_path=plots_dir / f"trace_metrics_{sweep_key}.png",
    )

    top_eigen_metrics = [
        ("S_top_eigenvalue", r"$\lambda_1(S)$"),
        ("S_star_top_eigenvalue", r"$\lambda_1(S^\star)$"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=top_eigen_metrics,
        ylabel="Top eigenvalue",
        title=build_teacher_attention_sweep_title(
            "Top eigenvalue of learned and teacher matrices",
            base_config,
        ),
        output_path=plots_dir / f"top_eigenvalue_metrics_{sweep_key}.png",
    )

    r1_metrics = [
        ("S_R1", r"$R_1(S)$"),
        ("S_star_R1", r"$R_1(S^\star)$"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=r1_metrics,
        ylabel=r"$R_1$",
        title=build_teacher_attention_sweep_title(
            "Spectral concentration",
            base_config,
        ),
        output_path=plots_dir / f"spectral_concentration_metrics_{sweep_key}.png",
    )

    effective_rank_metrics = [
        ("S_effective_rank", r"$\mathrm{erank}(S)$"),
        ("S_star_effective_rank", r"$\mathrm{erank}(S^\star)$"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=effective_rank_metrics,
        ylabel="Effective rank",
        title=build_teacher_attention_sweep_title(
            "Effective rank of learned and teacher matrices",
            base_config,
        ),
        output_path=plots_dir / f"effective_rank_metrics_{sweep_key}.png",
    )

    runtime_metrics = [
        ("runtime_seconds", "Runtime"),
    ]

    plot_metrics_vs_sweep(
        rows=rows,
        sweep_key=sweep_key,
        metrics=runtime_metrics,
        ylabel="Seconds",
        title=build_teacher_attention_sweep_title(
            "Runtime",
            base_config,
        ),
        output_path=plots_dir / f"runtime_{sweep_key}.png",
    )


def aggregate_one_sweep(sweep_dir: Path, force: bool = False) -> None:
    rows = collect_runs(sweep_dir)
    if not rows:
        print(f"[skip] No runs found in {sweep_dir}")
        return

    sweep_key = detect_sweep_key(rows)

    if not force and is_aggregated(sweep_dir):
        print(f"[skip] Already aggregated: {sweep_dir}")
        return

    rows.sort(key=lambda x: x[sweep_key])

    write_summary_csv(rows, sweep_dir / "summary.csv")

    run_configs = collect_run_configs(sweep_dir)
    metadata = build_sweep_metadata(run_configs, sweep_key)

    if metadata is None:
        print(f"[skip] No run configs found in {sweep_dir}")
        return

    save_sweep_metadata(sweep_dir, metadata)

    generate_teacher_attention_summary_plots(
        rows=rows,
        sweep_dir=sweep_dir,
        sweep_key=sweep_key,
        base_config=metadata["base_config"],
        zoom_ranges=[(None, None), (0, 500), (0, 1000), (0, 2000)],
    )

    print(f"[done] Aggregated {sweep_dir} (sweep key: {sweep_key})")


def find_sweep_dirs(root: Path) -> list[Path]:
    sweep_dirs = []

    if has_run_subdirs(root):
        sweep_dirs.append(root)

    for path in root.rglob("*"):
        if has_run_subdirs(path):
            sweep_dirs.append(path)

    unique = []
    seen = set()

    for path in sweep_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.sweep_dir)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    sweep_dirs = find_sweep_dirs(root)

    if not sweep_dirs:
        print("No sweep directories found.")
        return

    for sweep_dir in sweep_dirs:
        aggregate_one_sweep(sweep_dir, force=args.force)


if __name__ == "__main__":
    main()