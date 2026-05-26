import os

import config

# CPU thread controls must be set before importing torch
os.environ.setdefault("OMP_NUM_THREADS", config.OMP_NUM_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", config.MKL_NUM_THREADS)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="mamba_ssm")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch

from load import build_loaders, set_seed
from net import build_model
from train import run_final_evaluation, run_training


def main() -> None:
    """Training entry point: load data, build the model, train, and save final train/validation results."""
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    print("\n[Data] Building training and validation datasets...")
    train_loader, val_loader = build_loaders()

    print("\n[Model] Initializing model...")
    model = build_model(device)

    if config.LOAD_TRAINED_WEIGHTS:
        print(f"\n[Checkpoint] Attempting to load existing checkpoint: {config.MODEL_SAVE_PATH}")
        try:
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
            print("[Checkpoint] Loaded successfully; resuming training from the loaded checkpoint")
        except FileNotFoundError:
            print("[Checkpoint] No existing checkpoint found; starting training from random initialization")
        except Exception as exc:
            print(f"[Checkpoint] Failed to load checkpoint; starting from random initialization. Reason: {exc}")

    print("\n[Training] Starting training...")
    best_val_loss, best_epoch = run_training(model, train_loader, val_loader, device)

    print("\n[Evaluation] Loading best checkpoint and generating predictions for training and validation sets...")
    set_seed(config.SEED)
    run_final_evaluation(model, train_loader, val_loader, device, best_val_loss, best_epoch)

    print("\n[Done] Training and evaluation pipeline finished")


if __name__ == "__main__":
    main()
