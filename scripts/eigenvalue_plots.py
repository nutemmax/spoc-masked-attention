from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ensure repository root is importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


# ----------------------------
# config formatting utilities
# ----------------------------


def load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Failed to read JSON file {path}: {e}")
        return None


def find_config_json(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("config*.json"))
    if not matches:
        return None
    return matches[0]


def format_float_for_title(x: float | None) -> str:
    if x is None:
        return "NA"

    x = float(x)

    if x == 0:
        return "0"

    if float(x).is_integer():
        return str(int(x))

    abs_x = abs(x)
    exp = round(math.log10(abs_x))

    if math.isclose(abs_x, 10 ** exp, rel_tol=1e-12, abs_tol=1e-15):
        if x < 0:
            return rf"-10^{{{exp}}}"
        return rf"10^{{{exp}}}"

    if abs_x < 1e-3 or abs_x >= 1e3:
        s = f"{x:.2e}"
        base, exponent = s.split("e")
        exponent = int(exponent)
        base = float(base)
        base_str = f"{base:.2f}".rstrip("0").rstrip(".")
        return rf"{base_str}\cdot 10^{{{exponent}}}"

    return f"{x:.4f}".rstrip("0").rstrip(".")


def format_covariance_label(covariance_type: str, rho: float | None) -> str:
    cov = str(covariance_type).lower()

    if cov == "identity":
        return r"$\Sigma = I$"
    if cov == "toeplitz":
        if rho is None:
            return r"$\Sigma = \mathrm{Toeplitz}$"
        return rf"$\Sigma = \mathrm{{Toeplitz}}(\rho={format_float_for_title(rho)})$"
    if cov == "tridiagonal":
        if rho is None:
            return r"$\Sigma = \mathrm{Tridiag}$"
        return rf"$\Sigma = \mathrm{{Tridiag}}(\rho={format_float_for_title(rho)})$"
    if rho is None:
        return rf"$\Sigma = \mathrm{{{covariance_type}}}$"
    return rf"$\Sigma = \mathrm{{{covariance_type}}}(\rho={format_float_for_title(rho)})$"


def format_masking_label(masking_strategy: str | None) -> str:
    if masking_strategy is None:
        return "Mask=NA"

    mask = str(masking_strategy).replace("_", "-")
    return f"Mask={mask}"


def format_config_label(config: dict | None) -> str:
    if config is None:
        return ""

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    covariance_type = data_cfg.get("covariance_type", "NA")
    masking_strategy = data_cfg.get("masking_strategy", "NA")
    rho = data_cfg.get("rho", None)
    lambda_reg = training_cfg.get("lambda_reg", None)
    beta = model_cfg.get("beta", None)
    d = data_cfg.get("d", None)
    T = data_cfg.get("T", None)
    lr = training_cfg.get("learning_rate", None)
    n_steps = training_cfg.get("n_steps", None)

    line1_parts = [
        format_covariance_label(covariance_type, rho),
        format_masking_label(masking_strategy),
        rf"$\lambda = {format_float_for_title(lambda_reg)}$" if lambda_reg is not None else None,
        rf"$\beta = {format_float_for_title(beta)}$" if beta is not None else None,
    ]
    line1 = ", ".join(part for part in line1_parts if part is not None)

    line2_parts = []
    if d is not None:
        line2_parts.append(rf"$d = {d}$")
    if T is not None:
        line2_parts.append(rf"$T = {T}$")
    if lr is not None:
        line2_parts.append(rf"$\eta = {format_float_for_title(lr)}$")
    if n_steps is not None:
        line2_parts.append(rf"$\mathrm{{iters}} = {n_steps}$")

    line2 = ", ".join(line2_parts)

    return f"{line1}\n{line2}" if line2 else line1


def parse_config_from_name(name: str) -> dict:
    """
    Fallback parser from folder names like:
    cov_toeplitz_maskrandom_rho0p9_lambda1e-05_beta1_d50_T5_lr0p001_iter10000
    """
    cfg = {
        "data": {},
        "model": {},
        "training": {},
    }

    lower_name = name.lower()

    if "toeplitz" in lower_name:
        cfg["data"]["covariance_type"] = "toeplitz"
    elif "tridiag" in lower_name or "tridiagonal" in lower_name:
        cfg["data"]["covariance_type"] = "tridiagonal"
    elif "identity" in lower_name:
        cfg["data"]["covariance_type"] = "identity"

    if "maskrandom" in lower_name:
        cfg["data"]["masking_strategy"] = "random"
    elif "masklast" in lower_name:
        cfg["data"]["masking_strategy"] = "last"
    elif "mask_first" in lower_name or "maskfirst" in lower_name:
        cfg["data"]["masking_strategy"] = "first"

    patterns = {
        ("data", "rho"): r"rho([0-9]+p[0-9]+|[0-9]+e[-+]?[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        ("training", "lambda_reg"): r"lambda([0-9]+p[0-9]+|[0-9]+e[-+]?[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        ("model", "beta"): r"beta([0-9]+p[0-9]+|[0-9]+e[-+]?[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        ("data", "d"): r"_d([0-9]+)",
        ("data", "T"): r"_t([0-9]+)",
        ("training", "learning_rate"): r"lr([0-9]+p[0-9]+|[0-9]+e[-+]?[0-9]+|[0-9]+(?:\.[0-9]+)?)",
        ("training", "n_steps"): r"iter([0-9]+)",
    }

    for (section, key), pattern in patterns.items():
        match = re.search(pattern, lower_name)
        if not match:
            continue

        value_str = match.group(1).replace("p", ".")
        if key in {"d", "T", "n_steps"}:
            cfg[section][key] = int(float(value_str))
        else:
            cfg[section][key] = float(value_str)

    return cfg


# ----------------------------
# discovery utilities
# ----------------------------

def extract_ntrain(path: Path) -> int | None:
    match = re.search(r"ntrain_(\d+)", path.name)
    return int(match.group(1)) if match else None


def extract_ntrain_or_raise(path: Path) -> int:
    ntrain = extract_ntrain(path)
    if ntrain is None:
        raise ValueError(f"Could not extract n_train from path: {path}")
    return ntrain


def find_run_dirs(job_dir: Path) -> list[Path]:
    run_dirs = []
    for path in job_dir.iterdir():
        if path.is_dir():
            ntrain = extract_ntrain(path)
            if ntrain is not None:
                run_dirs.append(path)
    return sorted(run_dirs, key=extract_ntrain_or_raise)


def find_eigen_file(run_dir: Path, pattern: str) -> Path | None:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None
    return matches[0]


def is_job_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    try:
        children = list(path.iterdir())
    except Exception:
        return False

    return any(child.is_dir() and extract_ntrain(child) is not None for child in children)


def find_job_dirs(root: Path) -> list[Path]:
    job_dirs = []

    if is_job_dir(root):
        job_dirs.append(root)

    for path in root.rglob("*"):
        if is_job_dir(path):
            job_dirs.append(path)

    unique = []
    seen = set()
    for path in job_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    unique.sort()
    return unique


# ----------------------------
# eigenvalue stats
# ----------------------------

def effective_rank(eigvals: np.ndarray, eps: float = 1e-12) -> float:
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = eigvals[eigvals > eps]

    if eigvals.size == 0:
        return 0.0

    p = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def compute_summary_stats(ntrain: int, eigvals: np.ndarray) -> dict:
    eigvals = np.asarray(eigvals, dtype=float).reshape(-1)
    eigvals_sorted = np.sort(eigvals)[::-1]

    trace = float(np.sum(eigvals))
    top = float(eigvals_sorted[0]) if eigvals_sorted.size > 0 else np.nan
    r1 = float(top / trace) if trace > 0 else np.nan

    return {
        "n_train": ntrain,
        "num_eigs": int(eigvals.size),
        "min_eig": float(np.min(eigvals)) if eigvals.size > 0 else np.nan,
        "max_eig": float(np.max(eigvals)) if eigvals.size > 0 else np.nan,
        "mean_eig": float(np.mean(eigvals)) if eigvals.size > 0 else np.nan,
        "median_eig": float(np.median(eigvals)) if eigvals.size > 0 else np.nan,
        "std_eig": float(np.std(eigvals)) if eigvals.size > 0 else np.nan,
        "trace": trace,
        "top_eigenvalue": top,
        "R1": r1,
        "effective_rank": effective_rank(eigvals),
    }


# ----------------------------
# plot title helpers
# ----------------------------

def make_plot_title(metric_title: str, config_label: str = "") -> str:
    if config_label:
        return f"{metric_title}\n{config_label}"
    return metric_title


# ----------------------------
# plotting helpers
# ----------------------------

def plot_sorted_eigenvalues(
    eigvals: np.ndarray,
    ntrain: int,
    save_path: Path,
    config_label: str = "",
    x_min: int = 1,
    x_max: int | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    use_log_y: bool = False,
    dpi: int = 300,
) -> None:
    eigvals_sorted = np.sort(np.asarray(eigvals).reshape(-1))[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(
        np.arange(1, len(eigvals_sorted) + 1),
        eigvals_sorted,
        marker="o",
        linewidth=2.2,
        markersize=7,
    )

    ax.set_xlabel(r"Rank $i$")
    ax.set_ylabel(r"$\lambda_i$")

    title = make_plot_title(
        metric_title=rf"Sorted eigenvalues of $S$ ($n_{{\mathrm{{train}}}} = {ntrain}$)",
        config_label=config_label,
    )
    ax.set_title(title, pad=16)

    xmax = len(eigvals_sorted) if x_max is None else x_max
    ax.set_xlim(x_min, xmax)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    if use_log_y:
        ax.set_yscale("log")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] Saved plot: {save_path}")


def plot_eigenvalue_histogram(
    eigvals: np.ndarray,
    ntrain: int,
    save_path: Path,
    config_label: str = "",
    bins: int = 30,
    hist_range: tuple[float, float] | None = None,
    density: bool = False,
    use_log_y: bool = False,
    dpi: int = 300,
) -> None:
    vals = np.asarray(eigvals).reshape(-1).copy()

    if hist_range is not None:
        vals = vals[(vals >= hist_range[0]) & (vals <= hist_range[1])]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.hist(
        vals,
        bins=bins,
        density=density,
        edgecolor="black",
    )

    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density" if density else "Count")

    title = make_plot_title(
        metric_title=rf"Eigenvalue histogram of $S$ ($n_{{\mathrm{{train}}}} = {ntrain}$)",
        config_label=config_label,
    )
    ax.set_title(title, pad=16)

    if hist_range is not None:
        ax.set_xlim(*hist_range)

    if use_log_y:
        ax.set_yscale("log")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] Saved plot: {save_path}")


def plot_summary_metric_vs_ntrain(
    df_stats: pd.DataFrame,
    y_col: str,
    ylabel: str,
    metric_title: str,
    save_path: Path,
    config_label: str = "",
    dpi: int = 300,
) -> None:
    data = df_stats.sort_values("n_train").copy()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(
        data["n_train"],
        data[y_col],
        marker="o",
        linewidth=2.2,
        markersize=8,
    )

    ax.set_xlabel(r"$n_{\mathrm{train}}$")
    ax.set_ylabel(ylabel)

    title = make_plot_title(metric_title=metric_title, config_label=config_label)
    ax.set_title(title, pad=16)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] Saved plot: {save_path}")


# ----------------------------
# main processing
# ----------------------------

def load_run_data(run_dir: Path, eigen_pattern: str) -> tuple[int, np.ndarray, dict | None] | None:
    ntrain = extract_ntrain(run_dir)
    if ntrain is None:
        return None

    eig_file = find_eigen_file(run_dir, eigen_pattern)
    if eig_file is None:
        print(f"[warn] No eigenvalue file matching '{eigen_pattern}' in {run_dir}")
        return None

    try:
        eigvals = np.load(eig_file)
        eigvals = np.asarray(eigvals).reshape(-1)
    except Exception as e:
        print(f"[warn] Failed to load eigenvalues from {eig_file}: {e}")
        return None

    config = None
    config_json = find_config_json(run_dir)
    if config_json is not None:
        config = load_json(config_json)

    return ntrain, eigvals, config


def infer_job_config(job_dir: Path, run_dirs: list[Path], eigen_pattern: str) -> dict | None:
    for run_dir in run_dirs:
        loaded = load_run_data(run_dir, eigen_pattern)
        if loaded is None:
            continue
        _, _, config = loaded
        if config is not None:
            return config

    return parse_config_from_name(job_dir.name)


def process_one_job_dir(
    job_dir: Path,
    eigen_pattern: str,
    bins: int,
    hist_density: bool,
    hist_log_y: bool,
    sorted_log_y: bool,
    dpi: int,
) -> None:
    run_dirs = find_run_dirs(job_dir)
    if not run_dirs:
        print(f"[skip] No ntrain_* folders found in {job_dir}")
        return

    output_dir = job_dir / "eigenvalue_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    job_config = infer_job_config(job_dir, run_dirs, eigen_pattern)
    config_label = format_config_label(job_config)

    rows = []
    eigvals_by_ntrain: dict[int, np.ndarray] = {}

    for run_dir in run_dirs:
        loaded = load_run_data(run_dir, eigen_pattern)
        if loaded is None:
            continue

        ntrain, eigvals, _ = loaded
        eigvals_by_ntrain[ntrain] = eigvals
        rows.append(compute_summary_stats(ntrain, eigvals))

    if not rows:
        print(f"[skip] No usable eigenvalue files found in {job_dir}")
        return

    df_stats = pd.DataFrame(rows).sort_values("n_train").reset_index(drop=True)

    summary_csv_path = output_dir / "eigenvalue_summary_stats.csv"
    df_stats.to_csv(summary_csv_path, index=False)
    print(f"[done] Saved summary CSV: {summary_csv_path}")

    for ntrain, eigvals in sorted(eigvals_by_ntrain.items()):
        sorted_path = output_dir / f"eigvals_sorted_ntrain_{ntrain}.png"
        hist_path = output_dir / f"eigvals_hist_ntrain_{ntrain}.png"

        plot_sorted_eigenvalues(
            eigvals=eigvals,
            ntrain=ntrain,
            save_path=sorted_path,
            config_label=config_label,
            use_log_y=sorted_log_y,
            dpi=dpi,
        )

        plot_eigenvalue_histogram(
            eigvals=eigvals,
            ntrain=ntrain,
            save_path=hist_path,
            config_label=config_label,
            bins=bins,
            density=hist_density,
            use_log_y=hist_log_y,
            dpi=dpi,
        )

    plot_summary_metric_vs_ntrain(
        df_stats=df_stats,
        y_col="top_eigenvalue",
        ylabel="Top eigenvalue",
        metric_title=r"Top eigenvalue vs $n_{\mathrm{train}}$",
        save_path=output_dir / "top_eigenvalue_vs_ntrain.png",
        config_label=config_label,
        dpi=dpi,
    )

    plot_summary_metric_vs_ntrain(
        df_stats=df_stats,
        y_col="trace",
        ylabel=r"$\mathrm{Tr}(S)$",
        metric_title=r"Trace of $S$ vs $n_{\mathrm{train}}$",
        save_path=output_dir / "trace_vs_ntrain.png",
        config_label=config_label,
        dpi=dpi,
    )

    plot_summary_metric_vs_ntrain(
        df_stats=df_stats,
        y_col="R1",
        ylabel=r"$R_1 = \lambda_1 / \mathrm{Tr}(S)$",
        metric_title=r"$R_1$ vs $n_{\mathrm{train}}$",
        save_path=output_dir / "R1_vs_ntrain.png",
        config_label=config_label,
        dpi=dpi,
    )

    plot_summary_metric_vs_ntrain(
        df_stats=df_stats,
        y_col="effective_rank",
        ylabel="Effective rank",
        metric_title=r"Effective rank vs $n_{\mathrm{train}}$",
        save_path=output_dir / "effective_rank_vs_ntrain.png",
        config_label=config_label,
        dpi=dpi,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Path to a root directory containing config folders and job folders.",
    )
    parser.add_argument(
        "--eigen-pattern",
        type=str,
        default="eigenvalues*.npy",
        help="Glob pattern for eigenvalue files inside each ntrain_* run folder.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for eigenvalue histograms.",
    )
    parser.add_argument(
        "--hist-density",
        action="store_true",
        help="Use density=True for histograms.",
    )
    parser.add_argument(
        "--hist-log-y",
        action="store_true",
        help="Use log scale on the y-axis for histograms.",
    )
    parser.add_argument(
        "--sorted-log-y",
        action="store_true",
        help="Use log scale on the y-axis for sorted eigenvalue plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved plots.",
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    job_dirs = find_job_dirs(root)
    if not job_dirs:
        print("No job directories containing ntrain_* subfolders were found.")
        return

    print(f"Found {len(job_dirs)} job directories.")

    for job_dir in job_dirs:
        print(f"\n[process] {job_dir}")
        process_one_job_dir(
            job_dir=job_dir,
            eigen_pattern=args.eigen_pattern,
            bins=args.bins,
            hist_density=args.hist_density,
            hist_log_y=args.hist_log_y,
            sorted_log_y=args.sorted_log_y,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()