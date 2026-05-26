import os
import time

import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim

import config
from loss import reconstruction_loss, total_loss


def make_inputs_from_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one batch into model inputs and supervised targets."""
    if config.USE_FIRST_STEP_INPUT:
        targets = batch.to(device, non_blocking=True)
        inputs = targets[:, 0, :].unsqueeze(2)
    else:
        params, targets = batch
        inputs = params.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    return inputs, targets


def train_one_epoch(model, train_loader, optimizer, device: torch.device) -> tuple[float, float]:
    """Train for one epoch and return average reconstruction loss and auxiliary loss."""
    model.train()
    total_mse = 0.0
    total_aux = 0.0

    for batch in train_loader:
        inputs, targets = make_inputs_from_batch(batch, device)

        optimizer.zero_grad()
        predictions, gate_weights_list = model(inputs)
        loss, mse_value, aux_value = total_loss(predictions, targets, gate_weights_list)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_mse += mse_value.item()
        total_aux += aux_value.item()

    batch_count = max(len(train_loader), 1)
    return total_mse / batch_count, total_aux / batch_count


def evaluate_loss(model, val_loader, device: torch.device) -> float:
    """Compute average reconstruction loss on the validation set."""
    model.eval()
    total_mse = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = make_inputs_from_batch(batch, device)
            predictions, _ = model(inputs)
            total_mse += reconstruction_loss(predictions, targets).item()

    return total_mse / max(len(val_loader), 1)


def build_optimizer(model) -> optim.Optimizer:
    """Create the training optimizer."""
    print(f"[Optimizer] Adam, learning_rate={config.LR}")
    return optim.Adam(model.parameters(), lr=config.LR)


def run_training(model, train_loader, val_loader, device: torch.device) -> tuple[float, int]:
    """Full training loop: validate after each epoch and save the checkpoint with the lowest validation loss."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    optimizer = build_optimizer(model)
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_mse, train_aux = train_one_epoch(model, train_loader, optimizer, device)
        val_mse = evaluate_loss(model, val_loader, device)

        save_message = ""
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            save_message = f"  >>> Best checkpoint saved: {config.MODEL_SAVE_PATH}"

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1:5d}/{config.EPOCHS}  "
            f"Train MSE: {train_mse:.8f}  "
            f"Aux Loss: {train_aux:.6f}  "
            f"Val MSE: {val_mse:.8f}  "
            f"Best Val: {best_val_loss:.8f}  "
            f"Time: {elapsed:.2f}s"
            f"{save_message}"
        )

    print("\n[Training] Training completed")
    if best_epoch != -1:
        print(f"[Training] Best model obtained at epoch {best_epoch}, best validation MSE={best_val_loss:.8f}")
    else:
        print("[Training] No checkpoint was saved. Please verify that the data is non-empty and the training is functioning correctly.")

    return best_val_loss, best_epoch


def calculate_prediction_metrics(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> dict:
    """Compute overall error, final-step error, spectral angle, and centroid error for spectral evolution maps."""
    diff = predictions - targets
    overall_mse = float(np.mean(diff**2))
    final_step_mse = float(np.mean(diff[:, -1, :] ** 2))

    dot = np.sum(predictions * targets, axis=-1)
    pred_norm = np.linalg.norm(predictions, axis=-1)
    target_norm = np.linalg.norm(targets, axis=-1)
    cosine = dot / (pred_norm * target_norm + eps)
    mean_sam = float(np.mean(np.arccos(np.clip(cosine, -1.0, 1.0))))

    wavelength_axis = np.arange(predictions.shape[-1], dtype=np.float64)
    pred_sum = np.sum(predictions, axis=-1) + eps
    target_sum = np.sum(targets, axis=-1) + eps
    pred_centroid = np.sum(predictions * wavelength_axis, axis=-1) / pred_sum
    target_centroid = np.sum(targets * wavelength_axis, axis=-1) / target_sum
    centroid_mae = float(np.mean(np.abs(pred_centroid - target_centroid)))

    return {
        "overall_mse": overall_mse,
        "final_step_mse": final_step_mse,
        "mean_sam": mean_sam,
        "centroid_mae": centroid_mae,
    }


def print_metrics(name: str, metrics: dict) -> None:
    """Print a set of prediction metrics."""
    print(
        f"[{name}] "
        f"Overall MSE={metrics['overall_mse']:.10f}  "
        f"Final-step MSE={metrics['final_step_mse']:.10f}  "
        f"Mean SAM={metrics['mean_sam']:.10f}  "
        f"Centroid MAE={metrics['centroid_mae']:.10f}"
    )


def collect_predictions(model, data_loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions and targets for one dataset."""
    model.eval()
    predictions = []
    targets_all = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = make_inputs_from_batch(batch, device)
            preds, _ = model(inputs)

            # The first-step spectrum is known input, so saved predictions keep it equal to the ground truth.
            preds[:, 0:1, :] = targets[:, 0, :].unsqueeze(1)

            predictions.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    return np.concatenate(predictions, axis=0), np.concatenate(targets_all, axis=0)


def run_final_evaluation(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    """Load the best checkpoint, predict on train and validation sets, and save .mat results."""
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print(f"[Evaluation] Best checkpoint loaded: {config.MODEL_SAVE_PATH}")
    except Exception as exc:
        print(f"[Evaluation] Failed to load best checkpoint; using current model state. Reason: {exc}")

    start_time = time.time()
    train_predictions, train_targets = collect_predictions(model, train_loader, device)
    val_predictions, val_targets = collect_predictions(model, val_loader, device)
    prediction_time = time.time() - start_time

    train_metrics = calculate_prediction_metrics(train_predictions, train_targets)
    val_metrics = calculate_prediction_metrics(val_predictions, val_targets)
    print_metrics("Train", train_metrics)
    print_metrics("Val", val_metrics)
    print(f"[Evaluation] Prediction time: {prediction_time:.4f}s")

    result_payload = {
        "train_predictions": train_predictions,
        "train_targets": train_targets,
        "val_predictions": val_predictions,
        "val_targets": val_targets,
        "train_overall_mse": train_metrics["overall_mse"],
        "train_final_step_mse": train_metrics["final_step_mse"],
        "train_mean_sam": train_metrics["mean_sam"],
        "train_centroid_mae": train_metrics["centroid_mae"],
        "val_overall_mse": val_metrics["overall_mse"],
        "val_final_step_mse": val_metrics["final_step_mse"],
        "val_mean_sam": val_metrics["mean_sam"],
        "val_centroid_mae": val_metrics["centroid_mae"],
        "prediction_time_seconds": prediction_time,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }

    os.makedirs(os.path.dirname(config.RESULT_SAVE_PATH), exist_ok=True)
    sio.savemat(config.RESULT_SAVE_PATH, result_payload)
    print(f"[Evaluation] Prediction results saved to: {config.RESULT_SAVE_PATH}")
