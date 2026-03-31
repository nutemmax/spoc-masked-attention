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

# ensure repository root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.bayes import (
    bayes_population_risk_empirical,
    bayes_population_risk_last_mask,
    bayes_population_risk_uniform_mask,
)
from src.data.covariance import build_covariance, is_positive_definite
from src.data.generator import (
    generate_single_mask_dataset_from_alpha_torch,
    generate_single_mask_dataset_torch,
)
from src.evaluation.metrics import (
    attention_matrix,
    compute_eigenvalues_symmetric,
    compute_spectral_concentration,
    compute_trace,
    compute_weights_norm,
    compute_effective_rank
)
from src.models.attention import TiedSingleHeadAttention
from src.training.trainer import evaluate_reconstruction_loss, fit
from src.utils.config import apply_overrides, get_torch_dtype, load_config, validate_config
from src.utils.io import create_run_dir, save_json, save_run_arrays
from src.utils.wandb import (
    finish_wandb,
    init_wandb_if_enabled,
    log_final_metrics,
    log_training_history,
)
import src.utils.plots as plots


def set_seeds(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def compute_run_metrics(
    W: np.ndarray,
    sigma: np.ndarray,
    train_loss: float,
    population_risk: float,
    bayes_population_risk: float,
    empirical_bayes_risk: float,
    alpha: float | None,
    seed: int,
    n_train: int,
    n_population: int,
    history: dict[str, list[float]],
    runtime_seconds: float,
    n_steps: int,
    config_suffix: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    S = attention_matrix(W)
    eigenvalues = compute_eigenvalues_symmetric(S)

    trace_s = compute_trace(S)
    top_eigenvalue = float(eigenvalues[0])
    min_eigenvalue = float(eigenvalues[-1])
    R1 = compute_spectral_concentration(eigenvalues)
    effective_rank = compute_effective_rank(eigenvalues)
    weight_norm = compute_weights_norm(W)

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

    generalization_gap = float(population_risk - train_loss)
    excess_population_risk = float(population_risk - bayes_population_risk)

    metrics = {
        "alpha": float(alpha) if alpha is not None else None,
        "seed": int(seed),
        "n_train": int(n_train),
        "n_population": int(n_population),
        "train_loss": float(train_loss),
        "population_risk": float(population_risk),
        "bayes_population_risk": float(bayes_population_risk),
        "empirical_bayes_risk": float(empirical_bayes_risk),
        "generalization_gap": generalization_gap,
        "excess_population_risk": excess_population_risk,
        "runtime_seconds": float(runtime_seconds),
        "runtime_per_step_seconds": float(runtime_seconds / n_steps) if n_steps > 0 else float("nan"),
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "best_objective": best_objective,
        "objective_reduction": objective_reduction,
        "initial_train_loss_history": initial_train_loss_history,
        "final_train_loss_history": final_train_loss_history,
        "best_train_loss_history": best_train_loss_history,
        "train_loss_reduction": train_loss_reduction,
        "weight_norm": float(weight_norm),
        "trace_s": float(trace_s),
        "top_eigenvalue": float(top_eigenvalue),
        "min_eigenvalue": float(min_eigenvalue),
        "R1": float(R1),
        "effective_rank": float(effective_rank),
        "config_suffix": config_suffix,
    }

    return metrics, S, eigenvalues


def save_run_plots(
    run_dir: Path,
    S: np.ndarray,
    sigma: np.ndarray,
    eigenvalues: np.ndarray,
    history: dict[str, list[float]],
    config: dict,
    actual_n_train: int,
    bayes_population_risk: float | None = None,
) -> None:
    suffix = plots.build_config_suffix(config, actual_n_train=actual_n_train)

    fig, _ = plots.plot_eigenvalues(
        eigenvalues,
        title=plots.build_plot_title(
            metric_title=r"Sorted eigenvalues of learned matrix $S$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"eigenvalues__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_eigenvalue_histogram(
        eigenvalues,
        title=plots.build_plot_title(
            metric_title=r"Eigenvalue histogram of learned matrix $S$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"eigenvalue_histogram__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_matrix_heatmap(
        S,
        title=plots.build_plot_title(
            metric_title=r"Heatmap of learned matrix $S$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"S_heatmap__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_matrix_heatmap(
        sigma,
        title=plots.build_plot_title(
            metric_title=r"Heatmap of teacher covariance $\Sigma$",
            config=config,
            actual_n_train=actual_n_train,
        ),
    )
    fig.savefig(run_dir / f"sigma_heatmap__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_training_history(
        history,
        title=plots.build_plot_title(
            metric_title="Training convergence",
            config=config,
            actual_n_train=actual_n_train,
        ),
        bayes_population_risk=bayes_population_risk,
    )
    fig.savefig(run_dir / f"training_convergence_withBO__{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plots.plot_training_history(
        history,
        title=plots.build_plot_title(
            metric_title="Training convergence",
            config=config,
            actual_n_train=actual_n_train,
        ),
        bayes_population_risk=None,
    )
    fig.savefig(run_dir / f"training_convergence__{suffix}.png", bbox_inches="tight")
    plt.close(fig)


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
    eval_every = 25

    if n_train_override is not None:
        n_train_override = int(n_train_override)
        if n_train_override <= 0:
            raise ValueError("n_train must be positive.")
    else:
        if alpha is None:
            raise ValueError("Either training.n_train or training.alpha must be provided.")
        alpha = float(alpha)

    if masking_strategy not in {"random", "last"}:
        raise ValueError(
            f"Unsupported masking_strategy='{masking_strategy}'. Expected 'random' or 'last'."
        )

    if r != d:
        raise ValueError("Current setup expects r = d.")

    dtype = get_torch_dtype(config["model"]["dtype"])
    device = config["model"]["device"]

    sigma_np = build_covariance(
        covariance_type=config["data"]["covariance_type"],
        T=T,
        rho=config["data"]["rho"],
        length_scale=config["data"]["length_scale"],
        eta=config["data"]["eta"],
    )

    if not is_positive_definite(sigma_np):
        raise ValueError("Chosen covariance matrix is not positive definite.")

    sigma_t = torch.from_numpy(sigma_np).to(dtype=dtype, device=device)

    if n_train_override is not None:
        X_train_t, X_tilde_train_t, mask_train_t = generate_single_mask_dataset_torch(
            n_samples=n_train_override,
            sigma=sigma_t,
            d=d,
            mask_value=mask_value,
            dtype=dtype,
            device=device,
            masking_strategy=masking_strategy,
        )
    else:
        X_train_t, X_tilde_train_t, mask_train_t = generate_single_mask_dataset_from_alpha_torch(
            alpha=alpha,
            sigma=sigma_t,
            d=d,
            mask_value=mask_value,
            dtype=dtype,
            device=device,
            masking_strategy=masking_strategy,
        )

    actual_n_train = int(X_train_t.shape[0])

    n_population = int(config["evaluation"]["n_population"])
    X_pop_t, X_tilde_pop_t, mask_pop_t = generate_single_mask_dataset_torch(
        n_samples=n_population,
        sigma=sigma_t,
        d=d,
        mask_value=mask_value,
        dtype=dtype,
        device=device,
        masking_strategy=masking_strategy,
    )

    wandb_module = init_wandb_if_enabled(
        config=config,
        actual_n_train=actual_n_train,
        alpha=float(alpha) if n_train_override is None else None,
        seed=seed,
    )

    model = TiedSingleHeadAttention(
        d=d,
        r=r,
        beta=float(config["model"]["beta"]),
        normalize_sqrt_d=bool(config["model"]["normalize_sqrt_d"]),
        dtype=dtype,
        device=device,
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
        )
        runtime_seconds = time.perf_counter() - start_time

        train_loss = evaluate_reconstruction_loss(
            model, X_tilde_train_t, X_train_t, mask_train_t
        )
        population_risk = evaluate_reconstruction_loss(
            model, X_tilde_pop_t, X_pop_t, mask_pop_t
        )

        if masking_strategy == "random":
            bayes_population_risk = bayes_population_risk_uniform_mask(sigma_np)
        else:
            bayes_population_risk = bayes_population_risk_last_mask(sigma_np)

        X_pop_np = X_pop_t.detach().cpu().numpy()
        mask_pop_np = mask_pop_t.detach().cpu().numpy()
        empirical_bayes_risk = bayes_population_risk_empirical(X_pop_np, sigma_np, mask_pop_np)
        del X_pop_np, mask_pop_np

        W = model.W.detach().cpu().numpy()
        config_suffix = plots.build_config_suffix(config, actual_n_train=actual_n_train)

        metrics, S, eigenvalues = compute_run_metrics(
            W=W,
            sigma=sigma_np,
            train_loss=train_loss,
            population_risk=population_risk,
            bayes_population_risk=bayes_population_risk,
            empirical_bayes_risk=empirical_bayes_risk,
            alpha=float(alpha) if n_train_override is None else None,
            seed=seed,
            n_train=actual_n_train,
            n_population=n_population,
            history=history,
            runtime_seconds=runtime_seconds,
            n_steps=n_steps,
            config_suffix=config_suffix,
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
            "sigma": sigma_np,
            "eigenvalues": eigenvalues,
            "model_state_dict": model.state_dict(),
            "actual_n_train": actual_n_train,
            "config_suffix": config_suffix,
        }

    finally:
        finish_wandb(wandb_module)


def save_experiment_outputs(results: dict, run_dir: Path) -> None:
    suffix = results["config_suffix"]
    actual_n_train = results["actual_n_train"]

    save_json(results["config"], run_dir / f"config__{suffix}.json")
    save_json(results["metrics"], run_dir / f"metrics__{suffix}.json")

    save_run_arrays(
        run_dir=run_dir,
        W=results["W"],
        S=results["S"],
        sigma=results["sigma"],
        eigenvalues=results["eigenvalues"],
        config_suffix=suffix,
    )

    torch.save(results["model_state_dict"], run_dir / f"model_state__{suffix}.pt")

    save_run_plots(
        run_dir=run_dir,
        S=results["S"],
        sigma=results["sigma"],
        eigenvalues=results["eigenvalues"],
        history=results["history"],
        config=results["config"],
        actual_n_train=actual_n_train,
        bayes_population_risk=results["metrics"]["bayes_population_risk"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SPOC masked-attention experiment.")
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

        validate_config(config)
        config = apply_overrides(
            config=config,
            alpha=args.alpha,
            n_train=args.n_train,
            seed=args.seed,
            save_root=None,
            run_name=args.run_name,
        )

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