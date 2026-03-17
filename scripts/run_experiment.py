from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import time

# ensure repository root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.bayes import bayes_population_risk_empirical, bayes_population_risk_uniform_mask
from src.data.covariance import build_covariance, is_positive_definite
from src.data.generator import generate_single_mask_dataset, generate_single_mask_dataset_from_alpha
from src.evaluation.metrics import attention_matrix, compute_eigenvalues_symmetric, compute_spectral_concentration, compute_trace, compute_weights_norm
from src.models.attention import TiedSingleHeadAttention
from src.training.trainer import evaluate_reconstruction_loss, fit
from src.utils.plots import plot_eigenvalue_histogram, plot_eigenvalues, plot_matrix_heatmap, plot_training_history


def get_default_config() -> dict:
    """Return default experiment config."""
    return {
        "experiment": {
            "save_root": "results/individual",
            "run_name": None,
            "seed": 0,
        },
        "data": {
            "T": 4,
            "d": 64,
            "covariance_type": "tridiagonal",
            "rho": 0.5,
            "length_scale": None,
            "eta": None,
            "mask_value": 1.0,
        },
        "model": {
            "r": 64,
            "beta": 10.0,
            "normalize_sqrt_d": False,
            "dtype": "float64",
            "device": "cpu",
        },
        "training": {
            "alpha": 5.0,
            "n_train": None,
            "n_steps": 600,
            "learning_rate": 1e-3,
            "lambda_reg": 0.0,
        },
        "evaluation": {
            "n_population": 5000,
        },
    }


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively update a nested dictionary."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | None) -> dict:
    """Load config from yaml or use defaults."""
    config = get_default_config()
    if config_path is None:
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        return config
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a dictionary at top level.")

    return deep_update(config, loaded)


def apply_overrides(
    config: dict,
    alpha: float | None,
    n_train: int | None,
    seed: int | None,
    save_root: str | None,
    run_name: str | None,
) -> dict:
    """Apply command-line overrides to config."""
    updated = copy.deepcopy(config)

    if alpha is not None:
        updated["training"]["alpha"] = float(alpha)
    if n_train is not None:
        updated["training"]["n_train"] = int(n_train)
    if seed is not None:
        updated["experiment"]["seed"] = int(seed)
    if save_root is not None:
        updated["experiment"]["save_root"] = save_root
    if run_name is not None:
        updated["experiment"]["run_name"] = run_name
    return updated

def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map dtype name to torch dtype."""
    name = dtype_name.lower()
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be 'float64' or 'float32'.")


def set_seeds(seed: int) -> np.random.Generator:
    """Set numpy and torch seeds and return a numpy generator."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def numpy_to_torch(X: np.ndarray, dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.tensor(X, dtype=dtype, device=device)


def format_float_for_name(x: float) -> str:
    """Format float compactly for folder names."""
    if float(x).is_integer():
        return f"{x:.1f}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def build_run_name(config: dict) -> str:
    """Build a unique and informative run name."""
    custom_name = config["experiment"]["run_name"]
    if custom_name is not None:
        return str(custom_name)

    seed = int(config["experiment"]["seed"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    n_train = config["training"].get("n_train")
    if n_train is not None:
        return f"ntrain_{int(n_train)}_seed_{seed}_{timestamp}"

    alpha = format_float_for_name(float(config["training"]["alpha"]))
    return f"alpha_{alpha}_seed_{seed}_{timestamp}"


def create_run_dir(config: dict) -> Path:
    """Create and return the run directory."""
    save_root = PROJECT_ROOT / config["experiment"]["save_root"]
    save_root.mkdir(parents=True, exist_ok=True)

    run_name = build_run_name(config)
    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

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
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute scalar, convergence, runtime, and spectral metrics for one run."""
    S = attention_matrix(W)
    eigenvalues = compute_eigenvalues_symmetric(S)

    trace_s = compute_trace(S)
    top_eigenvalue = float(eigenvalues[0])
    min_eigenvalue = float(eigenvalues[-1])
    R1 = compute_spectral_concentration(eigenvalues)
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

        # main risk metrics
        "train_loss": float(train_loss),
        "population_risk": float(population_risk),
        "bayes_population_risk": float(bayes_population_risk),
        "empirical_bayes_risk": float(empirical_bayes_risk),
        "generalization_gap": generalization_gap,
        "excess_population_risk": excess_population_risk,

        # runtime
        "runtime_seconds": float(runtime_seconds),
        "runtime_per_step_seconds": float(runtime_seconds / n_steps) if n_steps > 0 else float("nan"),

        # convergence
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "best_objective": best_objective,
        "objective_reduction": objective_reduction,
        "initial_train_loss_history": initial_train_loss_history,
        "final_train_loss_history": final_train_loss_history,
        "best_train_loss_history": best_train_loss_history,
        "train_loss_reduction": train_loss_reduction,

        # spectral
        "weight_norm": float(weight_norm),
        "trace_s": float(trace_s),
        "top_eigenvalue": float(top_eigenvalue),
        "min_eigenvalue": float(min_eigenvalue),
        "R1": float(R1),
    }

    return metrics, S, eigenvalues

def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as json."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_run_arrays(run_dir: Path, W: np.ndarray, S: np.ndarray, sigma: np.ndarray, eigenvalues: np.ndarray) -> None:
    """Save numpy arrays for one run."""
    np.save(run_dir / "W.npy", W)
    np.save(run_dir / "S.npy", S)
    np.save(run_dir / "sigma.npy", sigma)
    np.save(run_dir / "eigenvalues.npy", eigenvalues)


def save_run_plots(run_dir: Path, S: np.ndarray, sigma: np.ndarray, eigenvalues: np.ndarray, history: dict[str, list[float]]) -> None:
    """Save plots for one run."""
    fig, _ = plot_eigenvalues(eigenvalues, title="Eigenvalues of S")
    fig.savefig(run_dir / "eigenvalues.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_eigenvalue_histogram(eigenvalues, title="Histogram of eigenvalues of S")
    fig.savefig(run_dir / "eigenvalue_histogram.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_matrix_heatmap(S, title="Learned attention matrix S")
    fig.savefig(run_dir / "S_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_matrix_heatmap(sigma, title="Teacher covariance Sigma")
    fig.savefig(run_dir / "sigma_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_training_history(history, title="Training convergence")
    fig.savefig(run_dir / "training_convergence.png", bbox_inches="tight")
    plt.close(fig)


def run_experiment(config: dict) -> dict:
    """Run one experiment and return all outputs."""
    seed = int(config["experiment"]["seed"])
    rng = set_seeds(seed)

    T = int(config["data"]["T"])
    d = int(config["data"]["d"])
    r = int(config["model"]["r"])
    alpha = config["training"].get("alpha")
    n_train_override = config["training"].get("n_train")
    mask_value = float(config["data"]["mask_value"])
    if n_train_override is not None:
        n_train_override = int(n_train_override)
        if n_train_override <= 0:
            raise ValueError("n_train must be positive.")
    else:
        alpha = float(alpha)

    if r != d:
        raise ValueError("Current setup expects r = d.")

    sigma = build_covariance(
        covariance_type=config["data"]["covariance_type"],
        T=T,
        rho=config["data"]["rho"],
        length_scale=config["data"]["length_scale"],
        eta=config["data"]["eta"],
    )

    if not is_positive_definite(sigma):
        raise ValueError("Chosen covariance matrix is not positive definite.")

    if n_train_override is not None:
        X_train, X_tilde_train, _, mask_train = generate_single_mask_dataset(
            n_samples=n_train_override,
            sigma=sigma,
            d=d,
            mask_value=mask_value,
            rng=rng,
        )
    else:
        X_train, X_tilde_train, _, mask_train = generate_single_mask_dataset_from_alpha(
            alpha=alpha,
            sigma=sigma,
            d=d,
            mask_value=mask_value,
            rng=rng,
        )

    n_population = int(config["evaluation"]["n_population"])
    X_pop, X_tilde_pop, _, mask_pop = generate_single_mask_dataset(
        n_samples=n_population,
        sigma=sigma,
        d=d,
        mask_value=mask_value,
        rng=rng,
    )

    dtype = get_torch_dtype(config["model"]["dtype"])
    device = config["model"]["device"]

    X_train_t = numpy_to_torch(X_train, dtype=dtype, device=device)
    X_tilde_train_t = numpy_to_torch(X_tilde_train, dtype=dtype, device=device)
    mask_train_t = torch.tensor(mask_train, dtype=torch.long, device=device)

    X_pop_t = numpy_to_torch(X_pop, dtype=dtype, device=device)
    X_tilde_pop_t = numpy_to_torch(X_tilde_pop, dtype=dtype, device=device)
    mask_pop_t = torch.tensor(mask_pop, dtype=torch.long, device=device)

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
    history = fit(
        model=model,
        X_tilde=X_tilde_train_t,
        X=X_train_t,
        mask_indices=mask_train_t,
        n_steps=n_steps,
        learning_rate=float(config["training"]["learning_rate"]),
        lambda_reg=float(config["training"]["lambda_reg"]),
    )
    runtime_seconds = time.perf_counter() - start_time

    train_loss = evaluate_reconstruction_loss(model, X_tilde_train_t, X_train_t, mask_train_t)
    population_risk = evaluate_reconstruction_loss(model, X_tilde_pop_t, X_pop_t, mask_pop_t)

    bayes_population_risk = bayes_population_risk_uniform_mask(sigma)
    empirical_bayes_risk = bayes_population_risk_empirical(X_pop, sigma, mask_pop)

    W = model.W.detach().cpu().numpy()
    metrics, S, eigenvalues = compute_run_metrics(
        W=W,
        sigma=sigma,
        train_loss=train_loss,
        population_risk=population_risk,
        bayes_population_risk=bayes_population_risk,
        empirical_bayes_risk=empirical_bayes_risk,
        alpha=float(alpha) if n_train_override is None else None,
        seed=seed,
        n_train=X_train.shape[0],
        n_population=n_population,
        history=history,
        runtime_seconds=runtime_seconds,
        n_steps=n_steps,
    )

    return {
        "config": config,
        "metrics": metrics,
        "history": history,
        "W": W,
        "S": S,
        "sigma": sigma,
        "eigenvalues": eigenvalues,
        "model_state_dict": model.state_dict(),
    }


def save_experiment_outputs(results: dict, run_dir: Path) -> None:
    """Save all outputs of one run."""
    save_json(results["config"], run_dir / "config.json")
    save_json(results["metrics"], run_dir / "metrics.json")
    # save_json(results["history"], run_dir / "history.json")

    # disable for now to save disk space!!! temporary!!
    save_run_arrays(
        run_dir=run_dir,
        W=results["W"],
        S=results["S"],
        sigma=results["sigma"],
        eigenvalues=results["eigenvalues"],
    )
    
    torch.save(results["model_state_dict"], run_dir / "model_state.pt")

    save_run_plots(
        run_dir=run_dir,
        S=results["S"],
        sigma=results["sigma"],
        eigenvalues=results["eigenvalues"],
        history = results["history"]
    )

    
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one SPOC masked-attention experiment.")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config file.")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha.")
    parser.add_argument("--n-train", type=int, default=None, help="Override training set size. If provided, overrides alpha.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--save-root", type=str, default=None, help="Override save root directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(
        config=config,
        alpha=args.alpha,
        n_train=args.n_train,
        seed=args.seed,
        save_root=args.save_root,
        run_name=args.run_name,
    )

    if args.save_root is None:
        if args.config is None:
            config_name = "default"
        else:
            config_name = Path(args.config).stem
        config["experiment"]["save_root"] = f"results/individual/{config_name}"

    run_dir = create_run_dir(config)
    results = run_experiment(config)
    save_experiment_outputs(results, run_dir)

    print(f"Saved run to: {run_dir}")
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()