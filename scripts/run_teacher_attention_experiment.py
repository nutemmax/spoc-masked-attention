from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.pca import evaluate_pca, fit_pca
from src.baselines.ridge import evaluate_ridge, fit_ridge_per_feature
from src.data.generator_teacher_attention import (
    generate_single_mask_teacher_attention_dataset_torch,
    generate_teacher_weights_torch,
)
from src.evaluation.metrics import (
    attention_level_error_torch,
    compute_effective_rank,
    compute_eigenvalues_symmetric,
    compute_spectral_concentration,
    compute_trace,
    compute_weights_norm,
    matrix_cosine_similarity_torch,
    relative_frobenius_error_torch,
    teacher_recovery_metrics_torch,
)
from src.models.attention import TiedSingleHeadAttention
from src.training.trainer import evaluate_reconstruction_loss, fit
from src.utils.config import apply_overrides, get_torch_dtype, load_config, validate_config
from src.utils.io import create_run_dir, format_float_for_name, save_json
from src.utils.wandb import (
    finish_wandb,
    init_wandb_if_enabled,
    log_final_metrics,
    log_training_history,
)
import src.utils.plots as plots


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_r_star(value, d: int) -> int:
    if value is None:
        return d
    if isinstance(value, str) and value.lower() == "d":
        return d

    r_star = int(value)
    if r_star <= 0:
        raise ValueError("r_star must be positive.")
    return r_star


def safe_relative_improvement(baseline_risk: float, model_risk: float) -> float:
    if np.isclose(baseline_risk, 0.0):
        return float("nan")
    return float((baseline_risk - model_risk) / baseline_risk)


def build_teacher_attention_suffix(config: dict, actual_n_train: int) -> str:
    data_cfg = config["data"]
    model_cfg = config["model"]
    teacher_cfg = config["teacher"]
    training_cfg = config["training"]
    r_star_label = teacher_cfg.get("r_star")
    if r_star_label is None:
        r_star_label = "d"

    parts = [
        "teacher_attention",
        f"init_{teacher_cfg['init']}",
        f"rstar_{r_star_label}",
        f"bstar_{format_float_for_name(float(teacher_cfg['beta_star']))}",
        f"sigstar_{format_float_for_name(float(teacher_cfg['sigma_star']))}",
        f"mask_{data_cfg['masking_strategy']}",
        f"d_{int(data_cfg['d'])}",
        f"T_{int(data_cfg['T'])}",
        f"r_{int(model_cfg['r'])}",
        f"beta_{format_float_for_name(float(model_cfg['beta']))}",
        f"lambda_{training_cfg['lambda_reg']}",
        f"lr_{training_cfg['learning_rate']}",
        f"iter_{int(training_cfg['n_steps'])}",
        f"ntrain_{int(actual_n_train)}",
        f"seed_{int(config['experiment']['seed'])}",
    ]

    return "__".join(parts).replace(".", "p")


def compute_spectral_metrics_from_matrix(matrix: np.ndarray, prefix: str) -> tuple[dict, np.ndarray]:
    eigenvalues = compute_eigenvalues_symmetric(matrix)

    trace = compute_trace(matrix)
    top_eigenvalue = float(eigenvalues[0])
    min_eigenvalue = float(eigenvalues[-1])
    R1 = compute_spectral_concentration(eigenvalues)
    effective_rank = compute_effective_rank(eigenvalues)
    frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))

    metrics = {
        f"{prefix}_trace": float(trace),
        f"{prefix}_top_eigenvalue": top_eigenvalue,
        f"{prefix}_min_eigenvalue": min_eigenvalue,
        f"{prefix}_R1": float(R1),
        f"{prefix}_effective_rank": float(effective_rank),
        f"{prefix}_frobenius_norm": frobenius_norm,
    }

    return metrics, eigenvalues


def compute_history_summaries(history: dict[str, list[float]]) -> dict[str, float]:
    objective_history = history.get("objective", [])
    train_loss_history = history.get("train_loss", [])

    initial_objective = float(objective_history[0]) if objective_history else float("nan")
    final_objective = float(objective_history[-1]) if objective_history else float("nan")
    best_objective = float(min(objective_history)) if objective_history else float("nan")
    objective_reduction = initial_objective - final_objective if objective_history else float("nan")

    initial_train_loss_history = float(train_loss_history[0]) if train_loss_history else float("nan")
    final_train_loss_history = float(train_loss_history[-1]) if train_loss_history else float("nan")
    best_train_loss_history = float(min(train_loss_history)) if train_loss_history else float("nan")
    train_loss_reduction = (
        initial_train_loss_history - final_train_loss_history
        if train_loss_history else float("nan")
    )

    return {
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "best_objective": best_objective,
        "objective_reduction": objective_reduction,
        "initial_train_loss_history": initial_train_loss_history,
        "final_train_loss_history": final_train_loss_history,
        "best_train_loss_history": best_train_loss_history,
        "train_loss_reduction": train_loss_reduction,
    }


def compute_run_metrics(
    model: TiedSingleHeadAttention,
    W_star_t: torch.Tensor,
    S_star_t: torch.Tensor,
    train_loss: float,
    population_risk: float,
    ridge_train_loss: float,
    ridge_population_risk: float,
    pca_train_loss: float,
    pca_population_risk: float,
    pca_n_components: int,
    history: dict[str, list[float]],
    runtime_seconds: float,
    n_steps: int,
    seed: int,
    n_train: int,
    n_population: int,
    alpha: float | None,
    config_suffix: str,
):
    W = model.W.detach().cpu().numpy()
    S = model.attention_matrix().detach().cpu().numpy()
    W_star = W_star_t.detach().cpu().numpy()
    S_star = S_star_t.detach().cpu().numpy()

    S_metrics, eigenvalues = compute_spectral_metrics_from_matrix(S, prefix="S")
    S_star_metrics, eigenvalues_star = compute_spectral_metrics_from_matrix(S_star, prefix="S_star")

    with torch.no_grad():
        S_t = model.attention_matrix()
        teacher_metrics = {
            "cosine_S_S_star": matrix_cosine_similarity_torch(S_t, S_star_t),
            "relative_error_S_S_star": relative_frobenius_error_torch(S_t, S_star_t),
        }

    metrics = {
        "alpha": float(alpha) if alpha is not None else None,
        "seed": int(seed),
        "n_train": int(n_train),
        "n_population": int(n_population),
        "train_loss": float(train_loss),
        "population_risk": float(population_risk),
        "generalization_gap": float(population_risk - train_loss),
        "runtime_seconds": float(runtime_seconds),
        "runtime_per_step_seconds": float(runtime_seconds / n_steps) if n_steps > 0 else float("nan"),
        "weight_norm": float(compute_weights_norm(W)),
        "W_star_norm": float(compute_weights_norm(W_star)),
        "ridge_lambda": 1e-2,
        "ridge_train_loss": float(ridge_train_loss),
        "ridge_population_risk": float(ridge_population_risk),
        "ridge_generalization_gap": float(ridge_population_risk - ridge_train_loss),
        "attention_vs_ridge_gap": float(population_risk - ridge_population_risk),
        "attention_vs_ridge_relative_improvement": safe_relative_improvement(
            ridge_population_risk,
            population_risk,
        ),
        "pca_n_components": int(pca_n_components),
        "pca_train_loss": float(pca_train_loss),
        "pca_population_risk": float(pca_population_risk),
        "pca_generalization_gap": float(pca_population_risk - pca_train_loss),
        "attention_vs_pca_gap": float(population_risk - pca_population_risk),
        "attention_vs_pca_relative_improvement": safe_relative_improvement(
            pca_population_risk,
            population_risk,
        ),
        "config_suffix": config_suffix,
    }

    metrics.update(compute_history_summaries(history))
    metrics.update(S_metrics)
    metrics.update(S_star_metrics)
    metrics.update(teacher_metrics)

    return metrics, W, S, W_star, S_star, eigenvalues, eigenvalues_star


def run_experiment(config: dict) -> dict:
    seed = int(config["experiment"]["seed"])
    set_seeds(seed)

    T = int(config["data"]["T"])
    d = int(config["data"]["d"])
    r = int(config["model"]["r"])
    alpha = config["training"].get("alpha")
    n_train_override = config["training"].get("n_train")
    mask_value = float(config["data"]["mask_value"])
    masking_strategy = str(config["data"]["masking_strategy"])

    if masking_strategy != "random":
        raise ValueError("The teacher-attention setting currently uses random masking only.")

    if r != d:
        raise ValueError("Current teacher-attention student setup expects r = d.")

    if n_train_override is not None:
        n_train_override = int(n_train_override)
        if n_train_override <= 0:
            raise ValueError("n_train must be positive.")
    else:
        if alpha is None:
            raise ValueError("Either training.n_train or training.alpha must be provided.")
        alpha = float(alpha)

    if n_train_override is not None:
        actual_n_train = n_train_override
    else:
        actual_n_train = int(round(float(alpha) * (d ** 2)))

    dtype = get_torch_dtype(config["model"]["dtype"])
    device = config["model"]["device"]
    normalize_sqrt_d = bool(config["model"]["normalize_sqrt_d"])

    teacher_cfg = config["teacher"]
    r_star = resolve_r_star(teacher_cfg.get("r_star"), d)
    beta_star = float(teacher_cfg["beta_star"])
    teacher_init = str(teacher_cfg["init"])
    sigma_star = float(teacher_cfg["sigma_star"])

    torch.manual_seed(seed)
    W_star_t, S_star_t = generate_teacher_weights_torch(
        d=d,
        r_star=r_star,
        teacher_init=teacher_init,
        sigma_star=sigma_star,
        dtype=dtype,
        device=device,
    )

    torch.manual_seed(seed + 1)
    train_data = generate_single_mask_teacher_attention_dataset_torch(
        n_samples=actual_n_train,
        T=T,
        d=d,
        S_star=S_star_t,
        beta_star=beta_star,
        mask_value=mask_value,
        dtype=dtype,
        device=device,
        masking_strategy=masking_strategy,
        normalize_sqrt_d=normalize_sqrt_d,
    )

    n_population = int(config["evaluation"]["n_population"])

    torch.manual_seed(seed + 2)
    pop_data = generate_single_mask_teacher_attention_dataset_torch(
        n_samples=n_population,
        T=T,
        d=d,
        S_star=S_star_t,
        beta_star=beta_star,
        mask_value=mask_value,
        dtype=dtype,
        device=device,
        masking_strategy=masking_strategy,
        normalize_sqrt_d=normalize_sqrt_d,
    )

    X_train_t = train_data["X"]
    X_tilde_train_t = train_data["X_tilde"]
    mask_train_t = train_data["mask_indices"]

    X_pop_t = pop_data["X"]
    X_tilde_pop_t = pop_data["X_tilde"]
    mask_pop_t = pop_data["mask_indices"]

    torch.manual_seed(seed + 3)
    model = TiedSingleHeadAttention(
        d=d,
        r=r,
        beta=float(config["model"]["beta"]),
        normalize_sqrt_d=normalize_sqrt_d,
        dtype=dtype,
        device=device,
    )

    wandb_module = init_wandb_if_enabled(
        config=config,
        actual_n_train=actual_n_train,
        alpha=float(alpha) if n_train_override is None else None,
        seed=seed,
    )

    eval_every = int(config["evaluation"].get("eval_every", 25))
    track_attention_error = bool(
        config["evaluation"].get("track_attention_error_during_training", True)
    )
    subset_size = int(config["evaluation"].get("attention_metric_subset_size", 512))
    subset_size = min(subset_size, n_population)

    X_tilde_metric_t = X_tilde_pop_t[:subset_size]
    A_star_metric_t = pop_data["A_star"][:subset_size]

    def extra_eval_fn(current_model) -> dict[str, float]:
        if track_attention_error:
            return teacher_recovery_metrics_torch(
                model=current_model,
                S_star=S_star_t,
                X_tilde=X_tilde_metric_t,
                A_star=A_star_metric_t,
            )

        return teacher_recovery_metrics_torch(
            model=current_model,
            S_star=S_star_t,
        )

    n_steps = int(config["training"]["n_steps"])
    start_time = time.perf_counter()

    try:
        history = fit(
            model=model,
            X_tilde=X_tilde_train_t,
            X=X_train_t,
            mask_indices=mask_train_t,
            n_steps=n_steps,
            learning_rate=float(config["training"]["learning_rate"]),
            lambda_reg=float(config["training"]["lambda_reg"]),
            X_tilde_eval=X_tilde_pop_t,
            X_eval=X_pop_t,
            mask_eval=mask_pop_t,
            eval_every=eval_every,
            extra_eval_fn=extra_eval_fn,
        )
        runtime_seconds = time.perf_counter() - start_time

        train_loss = evaluate_reconstruction_loss(
            model,
            X_tilde_train_t,
            X_train_t,
            mask_train_t,
        )
        population_risk = evaluate_reconstruction_loss(
            model,
            X_tilde_pop_t,
            X_pop_t,
            mask_pop_t,
        )

        X_train_np = X_train_t.detach().cpu().numpy()
        mask_train_np = mask_train_t.detach().cpu().numpy()
        X_pop_np = X_pop_t.detach().cpu().numpy()
        mask_pop_np = mask_pop_t.detach().cpu().numpy()

        ridge_lambda = 1e-2
        ridge_weights = fit_ridge_per_feature(X_train_np, lambda_reg=ridge_lambda)
        ridge_train_loss = evaluate_ridge(
            X=X_train_np,
            mask_indices=mask_train_np,
            weights_by_token=ridge_weights,
        )
        ridge_population_risk = evaluate_ridge(
            X=X_pop_np,
            mask_indices=mask_pop_np,
            weights_by_token=ridge_weights,
        )

        Td = T * d
        pca_n_components_cfg = config["evaluation"].get("pca_n_components")
        if pca_n_components_cfg is None:
            pca_n_components = min(75, Td // 2)
        else:
            pca_n_components = int(pca_n_components_cfg)
        pca_model = fit_pca(X_train_np, n_components=pca_n_components)
        pca_train_loss = evaluate_pca(
            X=X_train_np,
            mask_indices=mask_train_np,
            pca_model=pca_model,
        )
        pca_population_risk = evaluate_pca(
            X=X_pop_np,
            mask_indices=mask_pop_np,
            pca_model=pca_model,
        )

        config_suffix = build_teacher_attention_suffix(config, actual_n_train=actual_n_train)

        metrics, W, S, W_star, S_star, eigenvalues, eigenvalues_star = compute_run_metrics(
            model=model,
            W_star_t=W_star_t,
            S_star_t=S_star_t,
            train_loss=train_loss,
            population_risk=population_risk,
            ridge_train_loss=ridge_train_loss,
            ridge_population_risk=ridge_population_risk,
            pca_train_loss=pca_train_loss,
            pca_population_risk=pca_population_risk,
            pca_n_components=pca_n_components,
            history=history,
            runtime_seconds=runtime_seconds,
            n_steps=n_steps,
            seed=seed,
            n_train=actual_n_train,
            n_population=n_population,
            alpha=float(alpha) if n_train_override is None else None,
            config_suffix=config_suffix,
        )

        metrics["teacher_init"] = teacher_init
        metrics["r_star"] = int(r_star)
        metrics["beta_star"] = float(beta_star)
        metrics["sigma_star"] = float(sigma_star)

        if track_attention_error:
            metrics["final_attention_level_error"] = attention_level_error_torch(
                model=model,
                X_tilde=X_tilde_metric_t,
                A_star=A_star_metric_t,
                normalize_by_T2=True,
            )

        log_training_history(
            wandb_module=wandb_module,
            history=history,
            eval_every=eval_every,
        )

        log_final_metrics(
            wandb_module=wandb_module,
            metrics=metrics,
            step=n_steps,
        )

        return {
            "config": config,
            "metrics": metrics,
            "history": history,
            "W": W,
            "S": S,
            "W_star": W_star,
            "S_star": S_star,
            "eigenvalues": eigenvalues,
            "eigenvalues_star": eigenvalues_star,
            "model_state_dict": model.state_dict(),
            "actual_n_train": actual_n_train,
            "config_suffix": config_suffix,
        }

    finally:
        finish_wandb(wandb_module)


def save_run_plots(results: dict, run_dir: Path) -> None:
    suffix = results["config_suffix"]
    config = results["config"]
    actual_n_train = results["actual_n_train"]
    metrics = results["metrics"]

    S = results["S"]
    S_star = results["S_star"]
    eigenvalues = results["eigenvalues"]
    eigenvalues_star = results["eigenvalues_star"]
    history = results["history"]

    fig, _ = plots.plot_eigenvalues(
        eigenvalues,
        title=plots.build_teacher_attention_plot_title(
            metric_title=r"Sorted eigenvalues of learned matrix $S$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"eigenvalues__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_eigenvalues(
        eigenvalues_star,
        title=plots.build_teacher_attention_plot_title(
            metric_title=r"Sorted eigenvalues of teacher matrix $S^\star$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"eigenvalues_star__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_eigenvalues_comparison(
        eigenvalues,
        eigenvalues_star,
        title=plots.build_teacher_attention_plot_title(
            metric_title=r"Sorted eigenvalues of $S$ and $S^\star$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"eigenvalues_comparison__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_eigenvalue_histogram(
        eigenvalues,
        title=plots.build_teacher_attention_plot_title(
            metric_title=r"Eigenvalue histogram of learned matrix $S$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"eigenvalue_histogram__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_eigenvalue_histogram(
        eigenvalues_star,
        title=plots.build_teacher_attention_plot_title(
            metric_title=r"Eigenvalue histogram of teacher matrix $S^\star$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"eigenvalue_histogram_star__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_matrix_heatmap(
        S,
        title=plots.build_teacher_attention_plot_title(
            metric_title=r"Heatmap of learned matrix $S$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"S_heatmap__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_matrix_heatmap(
        S_star,
        title=plots.build_teacher_attention_plot_title(
            metric_title=r"Heatmap of teacher matrix $S^\star$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"S_star_heatmap__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_training_history(
        history,
        title=plots.build_teacher_attention_plot_title(
            metric_title="Training convergence without baselines",
            config=config,
            actual_n_train=actual_n_train,
        ),
        bayes_population_risk=None,
        ridge_population_risk=None,
        pca_population_risk=None,
        show_objective=True,
    )
    fig.savefig(run_dir / f"training_convergence_no_baselines__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_training_history(
        history,
        title=plots.build_teacher_attention_plot_title(
            metric_title="Training convergence",
            config=config,
            actual_n_train=actual_n_train,
        ),
        bayes_population_risk=None,
        ridge_population_risk=metrics.get("ridge_population_risk"),
        pca_population_risk=metrics.get("pca_population_risk"),
        show_objective=True,
    )
    fig.savefig(run_dir / f"training_convergence_RR_PCA__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    recovery_title_prefix = plots.build_teacher_attention_plot_title(
        metric_title="Teacher recovery",
        config=config,
        actual_n_train=actual_n_train,
    )

    for name, fig in plots.plot_teacher_recovery_history(
        history,
        title_prefix=recovery_title_prefix,
    ):
        fig.savefig(run_dir / f"{name}__{suffix}.png", bbox_inches="tight")
        plt.close(fig)


def save_experiment_outputs(results: dict, run_dir: Path) -> None:
    suffix = results["config_suffix"]

    save_json(results["config"], run_dir / f"config__{suffix}.json")
    save_json(results["metrics"], run_dir / f"metrics__{suffix}.json")

    np.save(run_dir / f"W__{suffix}.npy", results["W"])
    np.save(run_dir / f"S__{suffix}.npy", results["S"])
    np.save(run_dir / f"W_star__{suffix}.npy", results["W_star"])
    np.save(run_dir / f"S_star__{suffix}.npy", results["S_star"])
    np.save(run_dir / f"eigenvalues__{suffix}.npy", results["eigenvalues"])
    np.save(run_dir / f"eigenvalues_star__{suffix}.npy", results["eigenvalues_star"])

    torch.save(results["model_state_dict"], run_dir / f"model_state__{suffix}.pt")
    save_run_plots(results, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SPOC teacher-attention experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file.")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha.")
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Override training set size. If provided, overrides alpha.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--save-root", type=str, default=None, help="Override save root directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config = load_config(args.config)

        if args.save_root is not None:
            config.setdefault("experiment", {})
            config["experiment"]["save_root"] = args.save_root
        else:
            if "save_root" not in config.get("experiment", {}):
                config_name = Path(args.config).stem
                config.setdefault("experiment", {})
                config["experiment"]["save_root"] = f"results/individual/{config_name}"

        
        config = apply_overrides(
            config=config,
            alpha=args.alpha,
            n_train=args.n_train,
            seed=args.seed,
            save_root=None,
            run_name=args.run_name,
        )
        validate_config(config)

        run_dir = create_run_dir(PROJECT_ROOT, config)
        results = run_experiment(config)
        save_experiment_outputs(results, run_dir)

        print(f"Saved run to: {run_dir}")
        print(json.dumps(results["metrics"], indent=2))

    except Exception as e:
        print(f"[ERROR] Run failed for config: {args.config}")
        print(str(e))
        raise


if __name__ == "__main__":
    main()