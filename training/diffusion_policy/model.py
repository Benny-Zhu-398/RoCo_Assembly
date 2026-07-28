"""State-only Diffusion Policy network: StateEncoder + TaskEncoder (FiLM
task conditioning) + ConditionalUnet1D denoiser.

Architecture follows Chi et al., "Diffusion Policy" (2023): a 1D temporal
U-Net over the action-chunk axis, with every residual block FiLM-modulated
by a global conditioning vector (diffusion-timestep embedding concatenated
with the observation/task conditioning). The down/up-sampling here is kept
fully symmetric (every down stage halves the temporal length, every up
stage doubles it back) rather than the paper's asymmetric last-stage
variant, specifically so it stays correct for arbitrary `horizon` values
(including non-powers-of-2) instead of only the horizon the paper tuned
for -- skip connections are length-matched defensively for the same
reason. This is a functionally equivalent FiLM-conditioned temporal U-Net,
not a byte-for-byte port.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import ModelConfig  # noqa: E402


class StateEncoder(nn.Module):
    """MLP over the 22-D left-arm state. No vision encoder (state-only)."""

    def __init__(self, state_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TaskEncoder(nn.Module):
    """part label -> task embedding c. One row per part in PART_TO_IDX order.

    Even in stage-1 single-part training this path is fully wired (just
    fed a constant index every batch) so stage-2/3 curriculum only has to
    change which indices show up, never the model.
    """

    def __init__(self, num_parts: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_parts, emb_dim)

    def forward(self, task_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_idx)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _num_groups(n_groups: int, channels: int) -> int:
    """Largest divisor of `channels` that is <= n_groups (GroupNorm requires
    channels % num_groups == 0; small channel counts like the 7/10-D action
    space don't divide evenly by the default 8)."""
    g = min(n_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int) -> None:
        super().__init__()
        n_groups = _num_groups(n_groups, out_channels)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with a FiLM modulation (scale, bias) from `cond`
    applied between them, plus a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int, n_groups: int) -> None:
        super().__init__()
        self.block1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.block2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        scale, bias = self.cond_encoder(cond).chunk(2, dim=-1)   # each (B, out_channels)
        out = out * scale.unsqueeze(-1) + bias.unsqueeze(-1)     # FiLM
        out = self.block2(out)
        return out + self.residual_conv(x)


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or crop the last (time) dim of x to target_len. Handles the
    off-by-one lengths that show up for non-power-of-2 horizons."""
    cur_len = x.shape[-1]
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[..., :target_len]
    return F.pad(x, (0, target_len - cur_len))


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.down_modules = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=2, padding=1),
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out):
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_in, cond_dim, kernel_size, n_groups),
                nn.ConvTranspose1d(dim_in, dim_in, kernel_size=4, stride=2, padding=1),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(input_dim, input_dim, kernel_size, n_groups=n_groups),
            nn.Conv1d(input_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """sample: (B, T, input_dim). Returns predicted noise, (B, T, input_dim)."""
        x = sample.transpose(1, 2)  # (B, input_dim, T)
        orig_len = x.shape[-1]

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(sample.shape[0]).to(sample.device)
        step_feat = self.diffusion_step_encoder(timestep)          # (B, diffusion_step_embed_dim)
        cond = torch.cat([step_feat, global_cond], dim=-1)         # (B, cond_dim)

        skips = []
        for resblock1, resblock2, downsample in self.down_modules:
            x = resblock1(x, cond)
            x = resblock2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for resblock1, resblock2, upsample in self.up_modules:
            skip = skips.pop()
            x = _match_length(x, skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = resblock1(x, cond)
            x = resblock2(x, cond)
            x = upsample(x)

        x = _match_length(x, orig_len)
        x = self.final_conv(x)
        return x.transpose(1, 2)  # (B, T, input_dim)


class DiffusionPolicyNet(nn.Module):
    """StateEncoder + TaskEncoder -> global_cond -> ConditionalUnet1D."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_encoder = StateEncoder(cfg.state_dim, cfg.state_hidden_dim, cfg.state_feature_dim)
        self.task_encoder = TaskEncoder(cfg.num_parts, cfg.task_emb_dim)
        global_cond_dim = cfg.state_feature_dim + cfg.task_emb_dim
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.down_dims,
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
        )

    def global_cond(self, state: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        state_feat = self.state_encoder(state)
        task_emb = self.task_encoder(task_idx)
        return torch.cat([state_feat, task_emb], dim=-1)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> torch.Tensor:
        """noisy_action: (B, horizon, action_dim). Returns predicted noise, same shape."""
        cond = self.global_cond(state, task_idx)
        return self.unet(noisy_action, timestep, cond)
