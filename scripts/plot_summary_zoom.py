from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
})


def load_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)

    required_cols = {"n_train", "train_loss", "population_risk"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"summary.csv at {summary_path} is missing required columns: {sorted(missing)}"
        )

    return df


def load_sweep_config(sweep_dir: Path) -> dict | None:
    config_path = sweep_dir / "sweep_config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Failed to read {config_path}: {e}")
        return None


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


def format_config_label(base_config: dict | None) -> str:
    if base_config is None:
        return ""

    data_cfg = base_config.get("data", {})
    model_cfg = base_config.get("model", {})
    training_cfg = base_config.get("training", {})

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


def get_bayes_risk(df: pd.DataFrame) -> float | None:
    if "bayes_population_risk" not in df.columns:
        return None

    vals = df["bayes_population_risk"].dropna()
    if vals.empty:
        return None

    return float(vals.iloc[0])


def prepare_ntrain_data(
    df: pd.DataFrame,
    x_min: int | None = None,
    x_max: int | None = None,
) -> pd.DataFrame:
    data = df.copy()

    data = data.dropna(subset=["n_train", "train_loss", "population_risk"])
    data["n_train"] = pd.to_numeric(data["n_train"], errors="coerce")
    data["train_loss"] = pd.to_numeric(data["train_loss"], errors="coerce")
    data["population_risk"] = pd.to_numeric(data["population_risk"], errors="coerce")

    data = data.dropna(subset=["n_train", "train_loss", "population_risk"])
    data = data.sort_values("n_train")

    if x_min is not None:
        data = data[data["n_train"] >= x_min]
    if x_max is not None:
        data = data[data["n_train"] <= x_max]

    return data


def make_plot_title(
    config_label: str,
    include_metric_title: bool,
) -> str:
    title_parts = []

    if include_metric_title:
        title_parts.append(r"Train and population risk vs $n_{\mathrm{train}}$")

    if config_label:
        title_parts.append(config_label)

    return "\n".join(title_parts)


def plot_train_pop_vs_ntrain(
    df: pd.DataFrame,
    save_path: Path,
    config_label: str = "",
    x_min: int | None = None,
    x_max: int | None = None,
    include_metric_title: bool = True,
    dpi: int = 300,
) -> None:
    data = prepare_ntrain_data(df, x_min=x_min, x_max=x_max)

    if data.empty:
        print(f"[skip] No data left after filtering for plot: {save_path.name}")
        return

    bayes_risk = get_bayes_risk(df)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    ax.plot(
        data["n_train"],
        data["train_loss"],
        marker="o",
        linewidth=2.2,
        markersize=8,
        label="Train Loss",
    )

    ax.plot(
        data["n_train"],
        data["population_risk"],
        marker="s",
        linewidth=2.2,
        markersize=8,
        label="Population Risk",
    )

    if bayes_risk is not None:
        ax.axhline(
            bayes_risk,
            linestyle="--",
            linewidth=2.0,
            color="red",
            label="Bayes Optimal Risk",
        )

    ax.set_xlabel(r"$n_{\mathrm{train}}$")
    ax.set_ylabel("Risk")

    title = make_plot_title(
        config_label=config_label,
        include_metric_title=include_metric_title,
    )
    ax.set_title(title, pad=16)

    ax.grid(True, alpha=0.3)

    if x_min is not None or x_max is not None:
        ax.set_xlim(
            left=x_min if x_min is not None else None,
            right=x_max if x_max is not None else None,
        )

    ax.legend(loc="best")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] Saved plot: {save_path}")


def find_summary_dirs(root: Path) -> list[Path]:
    summary_dirs = []

    if (root / "summary.csv").exists():
        summary_dirs.append(root)

    for path in root.rglob("*"):
        if path.is_dir() and (path / "summary.csv").exists():
            summary_dirs.append(path)

    unique = []
    seen = set()
    for path in summary_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    return unique


def build_output_name(
    x_min: int | None,
    x_max: int | None,
    include_metric_title: bool,
    suffix: str = "",
) -> str:
    if x_min is None and x_max is None:
        name = "train_pop_vs_ntrain_full"
    else:
        left = x_min if x_min is not None else 0
        right = x_max if x_max is not None else "max"
        name = f"train_pop_vs_ntrain_zoom_{left}_{right}"

    title_tag = "with_metric_title" if include_metric_title else "config_only_title"
    name = f"{name}_{title_tag}"

    if suffix:
        name += f"_{suffix}"

    return name + ".png"


def process_one_sweep_dir(
    sweep_dir: Path,
    xmins: list[int | None],
    xmaxs: list[int | None],
    suffix: str = "",
) -> None:
    summary_path = sweep_dir / "summary.csv"
    df = load_summary(summary_path)

    if "n_train" not in df.columns:
        print(f"[skip] {sweep_dir} is not an n_train sweep.")
        return

    sweep_cfg = load_sweep_config(sweep_dir)
    base_config = sweep_cfg.get("base_config") if sweep_cfg is not None else None
    config_label = format_config_label(base_config)

    plots_dir = sweep_dir / "plots_zoom"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for x_min, x_max in zip(xmins, xmaxs):
        for include_metric_title in [True, False]:
            filename = build_output_name(
                x_min=x_min,
                x_max=x_max,
                include_metric_title=include_metric_title,
                suffix=suffix,
            )
            save_path = plots_dir / filename

            plot_train_pop_vs_ntrain(
                df=df,
                save_path=save_path,
                config_label=config_label,
                x_min=x_min,
                x_max=x_max,
                include_metric_title=include_metric_title,
            )


def parse_int_list(arg: str | None) -> list[int]:
    if arg is None or arg.strip() == "":
        return []

    out = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Path to one aggregated sweep directory or a root containing many aggregated sweep directories.",
    )
    parser.add_argument(
        "--xmaxs",
        type=str,
        default="1000,2000",
        help="Comma-separated list of x-axis upper bounds for zoomed plots. Example: 500,1000,2000",
    )
    parser.add_argument(
        "--xmins",
        type=str,
        default=None,
        help="Comma-separated list of x-axis lower bounds. If omitted, all are taken as 0.",
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also generate the full-range plot.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix appended to output filenames.",
    )
    args = parser.parse_args()

    root = Path(args.sweep_dir)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    xmaxs = parse_int_list(args.xmaxs)
    if not xmaxs and not args.include_full:
        raise ValueError("You must provide at least one xmax in --xmaxs, or use --include-full.")

    xmins = parse_int_list(args.xmins)

    if xmins:
        if len(xmins) != len(xmaxs):
            raise ValueError(
                f"--xmins and --xmaxs must have the same length. "
                f"Got {len(xmins)} xmins and {len(xmaxs)} xmaxs."
            )
    else:
        xmins = [0] * len(xmaxs)

    if args.include_full:
        xmins = [None] + xmins
        xmaxs = [None] + xmaxs

    sweep_dirs = find_summary_dirs(root)
    if not sweep_dirs:
        print("No aggregated sweep directories containing summary.csv were found.")
        return

    for sweep_dir in sweep_dirs:
        process_one_sweep_dir(
            sweep_dir=sweep_dir,
            xmins=xmins,
            xmaxs=xmaxs,
            suffix=args.suffix,
        )


if __name__ == "__main__":
    main()