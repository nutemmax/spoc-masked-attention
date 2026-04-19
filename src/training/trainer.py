from __future__ import annotations
import torch
from torch import Tensor
from tqdm.auto import tqdm
from collections.abc import Callable
from src.training.losses import reconstruction_loss, regularized_training_objective


def train_step(
    model,
    optimizer: torch.optim.Optimizer,
    X_tilde: Tensor,
    X: Tensor,
    mask_indices: Tensor,
    lambda_reg: float,
) -> tuple[float, float]:
    """Run one gradient step."""
    model.train()
    optimizer.zero_grad()

    objective = regularized_training_objective(model, X_tilde, X, mask_indices, lambda_reg)
    objective.backward()
    optimizer.step()

    with torch.no_grad():
        recon = reconstruction_loss(model, X_tilde, X, mask_indices)

    return float(objective.item()), float(recon.item())


@torch.no_grad()
def evaluate_reconstruction_loss(model, X_tilde: Tensor, X: Tensor, mask_indices: Tensor) -> float:
    """Evaluate reconstruction loss."""
    model.eval()
    loss = reconstruction_loss(model, X_tilde, X, mask_indices)
    return float(loss.item())


def fit(
    model,
    X_tilde: Tensor,
    X: Tensor,
    mask_indices: Tensor,
    n_steps: int,
    learning_rate: float,
    lambda_reg: float,
    X_tilde_eval: Tensor | None = None,
    X_eval: Tensor | None = None,
    mask_eval: Tensor | None = None,
    eval_every: int = 25,
    show_progress: bool = True,
    print_every: int | None = None,
    extra_eval_fn: Callable[[object], dict[str, float]] | None = None,
) -> dict[str, list[float]]:
    """Train the model for a fixed number of steps."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative.")
    if eval_every <= 0:
        raise ValueError("eval_every must be positive.")

    do_eval = X_tilde_eval is not None and X_eval is not None and mask_eval is not None
    do_extra_eval = extra_eval_fn is not None

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "objective": [],
        "train_loss": [],
        "steps": [],
        "test_loss": [],
        "generalization_gap": [],
    }

    if show_progress:
        progress_bar = tqdm(range(n_steps), desc="training", leave=True)
        iterator = progress_bar
    else:
        progress_bar = None
        iterator = range(n_steps)

    for step in iterator:
        objective_value, recon_value = train_step(
            model, optimizer, X_tilde, X, mask_indices, lambda_reg
        )

        history["objective"].append(objective_value)
        history["train_loss"].append(recon_value)

        should_eval = ((step + 1) % eval_every == 0) or (step == 0) or (step == n_steps - 1)
        if should_eval and (do_eval or do_extra_eval):
            history["steps"].append(step + 1)

            if do_eval:
                test_loss_value = evaluate_reconstruction_loss(model, X_tilde_eval, X_eval, mask_eval)
                history["test_loss"].append(test_loss_value)
                history["generalization_gap"].append(test_loss_value - recon_value)

            if do_extra_eval:
                assert extra_eval_fn is not None
                extra_metrics = extra_eval_fn(model)

                for key, value in extra_metrics.items():
                    history.setdefault(key, []).append(float(value))

        if progress_bar is not None:
            postfix = {
                "objective": f"{objective_value:.4e}",
                "train_loss": f"{recon_value:.4e}",
            }
            if do_eval and should_eval:
                postfix["test_loss"] = f"{test_loss_value:.4e}"
            progress_bar.set_postfix(**postfix)

        if print_every is not None and (step + 1) % print_every == 0:
            msg = f"[step {step + 1}/{n_steps}] objective={objective_value:.6e} train_loss={recon_value:.6e}"
            if do_eval and should_eval:
                msg += f" test_loss={test_loss_value:.6e}"
            print(msg)

    return history