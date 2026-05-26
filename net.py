"""Model definitions.

The model combines a 1D convolutional encoder, Mamba sequence blocks, and physics-aware MoE decoder stages for supercontinuum evolution prediction.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba
import config


# ============================================================
# Basic components
# ============================================================

class ConvBlock1d(nn.Module):
    """Standard Conv1d -> ReLU -> BatchNorm encoder block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.relu(self.conv(x)))


class MambaWrapper(nn.Module):
    """Wrap Mamba SSM with a Conv1d-compatible [B, D, L] interface.
    Input is transposed to [B, L, D], processed by Mamba and LayerNorm, then transposed back.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, L]  →  [B, L, D]  →  Mamba  →  [B, D, L]
        x = x.transpose(1, 2)
        x = self.norm(self.mamba(x))
        return x.transpose(1, 2)


# ============================================================
# Stage-1 physical experts: propagation step 1 to 5
# ============================================================

class E1A_WeakNonlinear(nn.Module):
    """Stage-1 expert for weak nonlinear SPM with smooth symmetric broadening."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.smooth = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        self.out = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.smooth(x) + x)


class E1B_StrongNonlinear(nn.Module):
    """Stage-1 expert for strong nonlinear SPM and asymmetric higher-order dispersion features."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.branch_small = nn.Conv1d(d_model, d_model // 2,
                                      kernel_size=3, padding=1)
        self.branch_large = nn.Conv1d(d_model, d_model // 2,
                                      kernel_size=9, padding=4)
        self.fuse = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.branch_small(x)
        l = self.branch_large(x)
        return self.fuse(torch.cat([s, l], dim=1))


class E1C_SolitonFormation(nn.Module):
    """Stage-1 expert for soliton formation and long-range compression dynamics."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.mamba = MambaWrapper(d_model)
        self.proj  = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.mamba(x))


class E1D_ContinuumBackground(nn.Module):
    """Stage-1 expert for broadband continuum background and low-power baseline structure."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.global_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.out = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x.mean(dim=-1)                    # [B, D]
        g = self.global_fc(g)                 # [B, D]
        g = g.unsqueeze(-1).expand_as(x)      # [B, D, L]
        return self.out(x + g)


# ============================================================
# Stage-2 physical experts: propagation step 5 to 25
# ============================================================

class E2A_SingleFission(nn.Module):
    """Stage-2 expert for single soliton fission with medium-scale local features."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.conv(x)
        scale = self.se(feat).unsqueeze(-1)   # Channel attention
        return feat * scale + x               # SE residual connection


class E2B_MultiFission(nn.Module):
    """Stage-2 expert for dense multi-soliton fission using multi-dilation convolutions."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        c1 = d_model // 4           # 16
        c2 = d_model // 4           # 16
        c3 = d_model - c1 - c2     # 32, ensuring c1 + c2 + c3 == d_model
        self.d1 = nn.Conv1d(d_model, c1, kernel_size=3, padding=1, dilation=1)
        self.d2 = nn.Conv1d(d_model, c2, kernel_size=3, padding=2, dilation=2)
        self.d3 = nn.Conv1d(d_model, c3, kernel_size=3, padding=4, dilation=4)
        self.fuse = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.d1(x), self.d2(x), self.d3(x)], dim=1)
        return self.fuse(out) + x


class E2C_DispersiveWave(nn.Module):
    """Stage-2 expert for dispersive-wave radiation and narrowband high-frequency peaks."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        # High-frequency-sensitive path
        self.hf_path = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        # Difference convolution: per-channel first-order difference to highlight narrowband DW transitions
        self.diff_conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                                   padding=1, groups=d_model, bias=False)
        # Initialize as a first-order difference kernel [-0.5, 0, 0.5]
        nn.init.constant_(self.diff_conv.weight, 0.0)
        with torch.no_grad():
            for i in range(d_model):
                self.diff_conv.weight[i, 0, 0] = -0.5
                self.diff_conv.weight[i, 0, 2] =  0.5

        self.out = nn.Conv1d(d_model * 2, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf   = self.hf_path(x)
        diff = self.diff_conv(x)
        return self.out(torch.cat([hf, diff], dim=1))


class E2D_PeregrinePeak(nn.Module):
    """Stage-2 expert for Peregrine-like maximum compression peaks."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.mamba = MambaWrapper(d_model)
        self.peak_attn = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m     = self.mamba(x)
        scale = self.peak_attn(m).unsqueeze(-1)
        return self.out(m * scale)


# ============================================================
# Stage-3 physical experts: propagation step 25 to 200
# ============================================================

class E3A_SlowRaman(nn.Module):
    """Stage-3 expert for slow Raman frequency shift with smooth long-range trends."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.trend = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=11,
                      padding=5, groups=d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        self.out = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.trend(x) + x)


class E3B_FastRaman(nn.Module):
    """Stage-3 expert for fast Raman shift and multi-soliton drift dynamics."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.mamba = MambaWrapper(d_model)
        self.ms = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
        )
        self.fuse = nn.Conv1d(d_model * 2, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m  = self.mamba(x)
        ms = self.ms(x)
        return self.fuse(torch.cat([m, ms], dim=1))


class E3C_InterferenceFringe(nn.Module):
    """Stage-3 expert for interference fringes between dispersive waves and Raman-shifted solitons."""

    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        # Kernel sizes 3, 5, 7, and 9 correspond to different fringe spacings
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, head_dim,
                          kernel_size=2 * i + 1, padding=i),
                nn.ReLU(),
            )
            for i in range(1, num_heads + 1)
        ])
        self.out = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [h(x) for h in self.heads]
        return self.out(torch.cat(parts, dim=1))


class E3D_OctaveBoundary(nn.Module):
    """Stage-3 expert for octave-spanning spectrum boundaries and global envelope modulation."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.out   = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx   = self.global_ctx(x).unsqueeze(-1)   # [B, D, 1] broadcast modulation coefficients
        local = self.local(x)
        return self.out(local * ctx + x)


# ============================================================
# Input-aware sparse gating
# ============================================================

class PhysicsAwareGating(nn.Module):
    """Input-aware sparse gating network.
    
    The forward pass uses Top-K sparse weights, while straight-through estimation keeps gradients flowing through the full softmax distribution.
    """

    def __init__(self, feat_dim: int, global_dim: int,
                 num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.noise_std   = 0.1        # Training noise standard deviation

        self.local_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )

    def forward(self, local_feat: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        l      = self.local_proj(local_feat)                # [B, 32]
        g      = self.global_proj(global_cond)              # [B, 32]
        logits = self.gate_fc(torch.cat([l, g], dim=-1))    # [B, num_experts]

        # Inject noise during training to prevent gating collapse
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Top-K sparsification
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1)            # [B, top_k]

        # Build the sparse weight matrix
        sparse = torch.zeros_like(logits)
        sparse.scatter_(1, topk_idx, topk_w)                 # [B, num_experts]

        # Straight-through: sparse forward pass with full softmax gradients in backward pass
        full = torch.softmax(logits, dim=-1)
        sparse_st = sparse + (full - full.detach())

        return sparse_st   # [B, num_experts]


# ============================================================
# Physics-aware MoE layer
# ============================================================

class PhysicsMoELayer(nn.Module):
    """Single physics-aware MoE layer.
    
    All experts are evaluated, then their outputs are combined with sparse gating weights.
    """

    def __init__(self, experts: nn.ModuleList,
                 feat_dim: int, global_dim: int,
                 num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.experts = experts
        self.gating  = PhysicsAwareGating(
            feat_dim, global_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor,
                global_cond: torch.Tensor) -> tuple:
        """Run the model forward pass.
        
        When USE_FIRST_STEP_INPUT=True, x is [B, spectrum_length, 1].
        When USE_FIRST_STEP_INPUT=False, x is [B, 3].
        Returns the predicted evolution map and the MoE gating weights.
        """
        weights = self.gating(x, global_cond)          # [B, num_experts]

        # Run all experts to keep gradient paths available
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )                                               # [B, num_experts, D, L]

        # Sparse weighted sum
        w   = weights.unsqueeze(-1).unsqueeze(-1)       # [B, num_experts, 1, 1]
        out = (expert_outs * w).sum(dim=1)              # [B, D, L]

        return out, weights


# ============================================================
# Full model
# ============================================================

class CNNModelWithPhysicsMoE(nn.Module):
    """CNN-Mamba model with physics-aware sparse MoE decoder stages.
    
    The encoder compresses the input spectrum or physical-parameter embedding into a bottleneck feature. The decoder expands propagation steps through three stages and inserts an MoE layer at each stage.
    """

    def __init__(
        self,
        input_size:      int   = None,
        output_size:     int   = None,
        channel_list:    list  = None,
        kernel_params:   list  = None,
        expert_dim:      int   = config.EXPERT_DIM,
        global_cond_dim: int   = config.GLOBAL_COND_DIM,
        num_experts:     int   = config.NUM_EXPERTS,
        top_k:           int   = config.TOP_K,
    ):
        super().__init__()

        input_size = config.INPUT_SIZE if input_size is None else input_size
        output_size = config.OUTPUT_SIZE if output_size is None else output_size
        channel_list  = channel_list  or config.CHANNEL_LIST
        kernel_params = kernel_params or config.KERNEL_PARAMS
        assert len(kernel_params) == len(channel_list)
        last_enc_ch = channel_list[-1]

        # -------- Physical-parameter embedding layer, used only in parameter-input mode --------
        # This layer is inactive when USE_FIRST_STEP_INPUT=True, saving computation
        enc_in_size = output_size  # Align with the spectrum length of the current dataset
        if not config.USE_FIRST_STEP_INPUT:
            self.param_embed = nn.Sequential(
                nn.Linear(input_size, 128),   # 3 → 128
                nn.ReLU(),
                nn.Linear(128, enc_in_size),  # 128 → 251
                nn.ReLU(),
            )
            # Apply BN to the three input parameters to reduce scale differences, e.g. pump_power 3200 vs gamma 1.0
            self.param_bn = nn.BatchNorm1d(input_size)
        else:
            self.param_embed = None
            self.param_bn    = None

        # -------- Encoder with input channels aligned to the spectrum length --------
        enc_layers, in_ch = [], enc_in_size
        for out_ch, (ks, pad) in zip(channel_list, kernel_params):
            enc_layers.append(ConvBlock1d(in_ch, out_ch, ks, pad))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # Global physical condition vector pooled from the encoder bottleneck
        self.global_cond_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(last_enc_ch, global_cond_dim),
            nn.ReLU(),
        )

        # -------- Decoder stage 1: propagation step 1 to 5 --------
        dec_ch1 = 256
        self.deconv1   = nn.Sequential(
            nn.ConvTranspose1d(last_enc_ch, dec_ch1,
                               kernel_size=5, stride=5, padding=0),
            nn.ReLU(),
        )
        self.proj_in1  = nn.Conv1d(dec_ch1, expert_dim, kernel_size=1)
        self.moe1      = PhysicsMoELayer(
            experts     = nn.ModuleList([
                E1A_WeakNonlinear(expert_dim),
                E1B_StrongNonlinear(expert_dim),
                E1C_SolitonFormation(expert_dim),
                E1D_ContinuumBackground(expert_dim),
            ]),
            feat_dim    = expert_dim,
            global_dim  = global_cond_dim,
            num_experts = num_experts,
            top_k       = top_k,
        )
        self.proj_out1 = nn.Conv1d(expert_dim, dec_ch1, kernel_size=1)

        # -------- Decoder stage 2: propagation step 5 to 25 --------
        dec_ch2 = 256
        self.deconv2   = nn.Sequential(
            nn.ConvTranspose1d(dec_ch1, dec_ch2,
                               kernel_size=5, stride=5, padding=0),
            nn.ReLU(),
        )
        self.proj_in2  = nn.Conv1d(dec_ch2, expert_dim, kernel_size=1)
        self.moe2      = PhysicsMoELayer(
            experts     = nn.ModuleList([
                E2A_SingleFission(expert_dim),
                E2B_MultiFission(expert_dim),
                E2C_DispersiveWave(expert_dim),
                E2D_PeregrinePeak(expert_dim),
            ]),
            feat_dim    = expert_dim,
            global_dim  = global_cond_dim,
            num_experts = num_experts,
            top_k       = top_k,
        )
        self.proj_out2 = nn.Conv1d(expert_dim, dec_ch2, kernel_size=1)

        # -------- Decoder stage 3: propagation step 25 to 200 --------
        self.deconv3   = nn.Sequential(
            nn.ConvTranspose1d(dec_ch2, expert_dim,
                               kernel_size=8, stride=8, padding=0),
            nn.ReLU(),
        )
        self.moe3      = PhysicsMoELayer(
            experts     = nn.ModuleList([
                E3A_SlowRaman(expert_dim),
                E3B_FastRaman(expert_dim),
                E3C_InterferenceFringe(expert_dim),
                E3D_OctaveBoundary(expert_dim),
            ]),
            feat_dim    = expert_dim,
            global_dim  = global_cond_dim,
            num_experts = num_experts,
            top_k       = top_k,
        )
        self.final_proj = nn.ConvTranspose1d(
            expert_dim, output_size, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Run the model forward pass.
        
        When USE_FIRST_STEP_INPUT=True, x is [B, spectrum_length, 1].
        When USE_FIRST_STEP_INPUT=False, x is [B, 3].
        Returns the predicted evolution map and the MoE gating weights.
        """
        if config.USE_FIRST_STEP_INPUT:
            # First-step spectrum mode: x [B, 251, 1] is fed directly into the encoder
            feat = self.encoder(x)                     # [B, last_enc_ch, 1]
        else:
            # Physical-parameter mode: [B, 3] -> BN -> fully connected embedding -> [B, 251, 1]
            x    = self.param_bn(x)                    # Reduce scale differences
            x    = self.param_embed(x)                 # [B, 251]
            x    = x.unsqueeze(2)                      # [B, 251, 1]
            feat = self.encoder(x)                     # [B, last_enc_ch, 1]

        global_cond = self.global_cond_proj(feat)      # [B, global_cond_dim]

        gate_weights_list = []

        # Stage 1
        d1      = self.deconv1(feat)                   # [B, 256, 5]
        d1_e    = self.proj_in1(d1)                    # [B, expert_dim, 5]
        d1_moe, w1 = self.moe1(d1_e, global_cond)
        d1_out  = self.proj_out1(d1_moe) + d1          # Residual connection
        gate_weights_list.append(w1)

        # Stage 2
        d2      = self.deconv2(d1_out)                 # [B, 256, 25]
        d2_e    = self.proj_in2(d2)                    # [B, expert_dim, 25]
        d2_moe, w2 = self.moe2(d2_e, global_cond)
        d2_out  = self.proj_out2(d2_moe) + d2          # Residual connection
        gate_weights_list.append(w2)

        # Stage 3
        d3      = self.deconv3(d2_out)                 # [B, expert_dim, 200]
        d3_moe, w3 = self.moe3(d3, global_cond)
        gate_weights_list.append(w3)

        # Output
        out = self.final_proj(d3_moe)                  # [B, output_size, 200]
        out = out.transpose(1, 2)                      # [B, 200, output_size]

        # Auxiliary loss is computed in loss.py; only the weight list is returned here
        return out, gate_weights_list


# ============================================================
# Utility functions
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(device: torch.device) -> CNNModelWithPhysicsMoE:
    """Instantiate the model, move it to the target device, and print the parameter count."""
    model = CNNModelWithPhysicsMoE(
        input_size=config.INPUT_SIZE,
        output_size=config.OUTPUT_SIZE,
    ).to(device)
    print(f"[Model] Total trainable parameters: {count_parameters(model):,}")
    return model
