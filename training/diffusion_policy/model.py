"""State-only Diffusion Policy network: StateEncoder + TaskEncoder (FiLM
task conditioning) + ConditionalUnet1D denoiser.

ConditionalResidualBlock1D / ConditionalUnet1D below are a faithful port of
real-stanford/diffusion_policy's
diffusion_policy/model/diffusion/conditional_unet1d.py (+ conv1d_components.py
+ positional_embedding.py), fetched 2026-07-29. Ported (not hand-derived)
after a months-long debugging effort on a hand-written version of this
U-Net turned up a real shape bug in the up-path (see git history) plus a
persistent, never-fully-explained ceiling on how much the model used state
conditioning, with every hypothesis that had a clear mechanism (FiLM
ordering, magnitude imbalance vs the diffusion-step embedding, sampling
config) ruled out by direct experiment. Swapping in the reference
implementation is the last diagnostic that can distinguish "our
hand-written denoiser has a bug we haven't found" from "the problem is
elsewhere" -- see training/diffusion_policy debugging history.

Deliberately kept different from the reference:
  - local_cond / impainting-style conditioning: dropped. We only ever use
    global_cond (state+task FiLM conditioning), never local_cond or
    masked-impainting trajectories, so that machinery is dead weight here.
  - cond_predict_scale=True is our default (the reference class defaults
    to False, i.e. bias-only FiLM: out = out + embed). We use full
    scale+bias FiLM (out = out*scale + embed), matching what this repo's
    StateEncoder/TaskEncoder sizing was already designed around.
  - No einops dependency (not in requirements.txt) -- `rearrange`/
    `Rearrange` calls are replaced with equivalent `transpose`/`reshape`.
  - GroupNorm group count is not defensively clamped to a divisor of the
    channel count (the reference doesn't either): down_dims=(128,256,512)
    all divide evenly by n_groups=8, so this only matters if down_dims
    changes to something that doesn't -- in which case nn.GroupNorm raises
    a clear construction-time error rather than silently doing something
    else.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import ModelConfig  # noqa: E402
from vision import MultiCameraEncoder  # noqa: E402


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


# --- ported from diffusion_policy/model/diffusion/positional_embedding.py ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# --- ported from diffusion_policy/model/diffusion/conv1d_components.py ---
class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# --- ported from diffusion_policy/model/diffusion/conditional_unet1d.py ---
class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871 -- predicts
        # per-channel scale and bias (cond_predict_scale=True) or bias only.
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, horizon). cond: (B, cond_dim). Returns (B, out_channels, horizon)."""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed.unsqueeze(-1)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # NOTE (matches the reference exactly): this iterates reversed(in_out[1:]),
        # i.e. it DROPS the shallowest down-stage's skip connection -- the up
        # path never consumes it, reconstructing the final resolution purely
        # from its own upsampling. len(up_modules) == len(down_modules) - 1.
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """sample: (B, horizon, input_dim). Returns predicted noise, (B, horizon, input_dim)."""
        x = sample.transpose(1, 2)  # (B, input_dim, horizon)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        skips = []
        for resblock1, resblock2, downsample in self.down_modules:
            x = resblock1(x, global_feature)
            x = resblock2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for resblock1, resblock2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resblock1(x, global_feature)
            x = resblock2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)  # (B, horizon, input_dim)


class DiffusionPolicyNet(nn.Module):
    """StateEncoder + TaskEncoder (+ optional MultiCameraEncoder) -> global_cond -> ConditionalUnet1D."""

    def __init__(self, cfg: ModelConfig, vision_image_hw: Tuple[int, int] = (240, 320)) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_encoder = StateEncoder(cfg.state_dim, cfg.state_hidden_dim, cfg.state_feature_dim)
        self.task_encoder = TaskEncoder(cfg.num_parts, cfg.task_emb_dim)
        global_cond_dim = cfg.state_feature_dim + cfg.task_emb_dim

        self.vision_encoder: Optional[MultiCameraEncoder] = None
        if cfg.use_vision:
            # vision_image_hw must match the actual (possibly decode-time
            # resized, see config.py::DataConfig.image_resize_hw) pixel
            # shape images arrive at -- only matters when vision_crop_hw is
            # None (see vision.py::RgbEncoder), but always passed through
            # for the backbone's shape-probe dummy forward pass.
            self.vision_encoder = MultiCameraEncoder(
                camera_keys=list(cfg.camera_keys),
                image_hw=vision_image_hw,
                backbone_name=cfg.vision_backbone,
                pretrained=cfg.vision_pretrained,
                use_group_norm=cfg.vision_use_group_norm,
                num_keypoints=cfg.vision_num_keypoints,
                crop_hw=cfg.vision_crop_hw,
                crop_is_random=cfg.vision_crop_is_random,
            )
            global_cond_dim += self.vision_encoder.feature_dim

        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.down_dims,
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
        )

    def global_cond(
        self,
        state: torch.Tensor,
        task_idx: torch.Tensor,
        images: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        state_feat = self.state_encoder(state)
        task_emb = self.task_encoder(task_idx)
        feats = [state_feat, task_emb]
        if self.vision_encoder is not None:
            if images is None:
                raise ValueError("cfg.use_vision=True but global_cond() got images=None")
            feats.append(self.vision_encoder(images))
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
        task_idx: torch.Tensor,
        images: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """noisy_action: (B, horizon, action_dim). Returns predicted noise, same shape."""
        cond = self.global_cond(state, task_idx, images)
        return self.unet(noisy_action, timestep, global_cond=cond)
