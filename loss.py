"""Loss functions.

Includes:
  - reconstruction_loss: MSE reconstruction loss.
  - load_balance_loss: MoE load-balancing auxiliary loss.
  - total_loss: unified weighted loss interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import config


# ============================================================
# Reconstruction loss
# ============================================================
_mse = nn.MSELoss()


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise MSE reconstruction loss.
    pred and target both have shape [B, 200, 251].
    """
    return _mse(pred, target)


# ============================================================
# MoE load-balancing auxiliary loss
# ============================================================
def load_balance_loss(gate_weights_list: List[torch.Tensor]) -> torch.Tensor:
    """Load-balancing auxiliary loss to prevent expert collapse.
    
    The loss combines an importance term and a load term for each MoE stage, encouraging expert usage to stay close to uniform.
    """
    total = torch.tensor(0.0, device=gate_weights_list[0].device)

    for weights in gate_weights_list:
        # weights: [B, num_experts]
        num_experts = weights.shape[1]
        uniform = 1.0 / num_experts

        # --- Importance Loss ---
        # Sum expert weights within the batch and normalize them into probabilities
        importance = weights.sum(dim=0)                  # [num_experts]
        importance = importance / (importance.sum() + 1e-8)
        importance_loss = ((importance - uniform) ** 2).mean()

        # --- Load Loss ---
        # Average gating activation probability for each expert within the batch
        load      = weights.mean(dim=0)                  # [num_experts]
        load_loss = ((load - uniform) ** 2).mean()

        total = total + importance_loss + load_loss

    return total


# ============================================================
# Total loss used by the training loop
# ============================================================
def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gate_weights_list: List[torch.Tensor],
    aux_weight: float | None = None,
) -> tuple:
    """Compute total loss = MSE + aux_weight * load-balancing loss.
    
    Returns total loss, reconstruction loss, and auxiliary loss as scalar tensors.
    """
    if aux_weight is None:
        aux_weight = config.MOE_AUX_WEIGHT

    mse_val = reconstruction_loss(pred, target)
    aux_val = load_balance_loss(gate_weights_list)
    return mse_val + aux_weight * aux_val, mse_val, aux_val
