"""Vision encoder: turns a camera image into a fixed-length feature vector
that gets concatenated into `global_cond` alongside the state/task features
(see model.py::DiffusionPolicyNet.global_cond), the same slot lerobot's own
DiffusionPolicy plugs its image features into
(policies/diffusion/modeling_diffusion.py::DiffusionModel._prepare_global_conditioning).
Nothing else about the state/action pipeline (normalization, rotation
convention, gripper units) changes -- this module only adds a new input.

`RgbEncoder`/`SpatialSoftmax` below are a from-scratch reimplementation of
lerobot's `DiffusionRgbEncoder`/`SpatialSoftmax`
(lerobot/policies/diffusion/modeling_diffusion.py, fetched 2026-08-13), not
an import -- this package intentionally has no lerobot runtime dependency
(see requirements.txt / train.py's module docstring), same reason
model.py's ConditionalUnet1D is a ported reimplementation rather than an
import of the upstream reference. Defaults (resnet18, ImageNet-pretrained,
use_group_norm=False, spatial_softmax_num_keypoints=32) match lerobot's
DiffusionConfig defaults as of the fetch date.

Two cameras from tools/roco2026_by_part are used: `head` (global scene
view) and `left_hand` (left-wrist close-up). `right_hand` is deliberately
excluded -- the right arm never moves in this dataset (see
constants.py::LEFT_STATE_IDX's comment and right_arm_constants.json), so a
camera rigidly mounted to it is effectively a second static, low-information
viewpoint, not worth doubling encoder compute for.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import CAMERA_KEYS  # noqa: E402  (re-exported for convenience)


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax (Finn et al., https://huggingface.co/papers/1509.06113).

    Turns a (B, C, H, W) conv feature map into (B, num_kp, 2) image-space
    keypoint coordinates: per (possibly 1x1-conv-projected) channel, a
    channel-wise softmax over spatial positions gives an attention map, and
    the expected (x, y) under that attention is the keypoint. This is a much
    lower-dimensional, more stable summary of "where in the image the
    interesting stuff is" than a global average/max pool.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_kp: Optional[int] = None) -> None:
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))  # (H*W, 2)

    @property
    def out_dim(self) -> int:
        return self._out_c * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, H, W) -> (B, num_kp * 2) flattened (x, y) keypoints."""
        if self.nets is not None:
            features = self.nets(features)
        b = features.shape[0]
        features = features.reshape(-1, self._in_h * self._in_w)  # (B*out_c, H*W)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid  # (B*out_c, 2)
        return expected_xy.view(b, self._out_c * 2)


def _replace_bn_with_gn(module: nn.Module, features_per_group: int = 16) -> None:
    """In-place: swap every nn.BatchNorm2d in `module` for an nn.GroupNorm
    with the same channel count. Only meaningful when NOT loading pretrained
    weights (see RgbEncoder.__init__'s guard) -- BatchNorm running stats
    from a pretrained checkpoint have no GroupNorm equivalent to inherit."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(max(1, child.num_features // features_per_group), child.num_features))
        else:
            _replace_bn_with_gn(child, features_per_group)


class RgbEncoder(nn.Module):
    """One camera stream -> a fixed-length feature vector.

    torchvision ResNet backbone (classifier head stripped, conv trunk only)
    -> SpatialSoftmax keypoint pooling -> Linear+ReLU projection.
    """

    def __init__(
        self,
        image_hw: Tuple[int, int] = (240, 320),
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        use_group_norm: bool = False,
        num_keypoints: int = 32,
        crop_hw: Optional[Tuple[int, int]] = None,
        crop_is_random: bool = True,
    ) -> None:
        super().__init__()
        if use_group_norm and pretrained:
            # Same guard lerobot's DiffusionRgbEncoder has: swapping BatchNorm
            # for freshly-initialized GroupNorm in a pretrained backbone
            # discards the pretrained running statistics -- not a supported
            # combination, not a silent downgrade.
            raise ValueError("use_group_norm=True is incompatible with pretrained=True")

        self.crop_hw = crop_hw
        self.crop_is_random = crop_is_random
        if crop_hw is not None:
            self.center_crop = torchvision.transforms.CenterCrop(crop_hw)
            self.random_crop = (
                torchvision.transforms.RandomCrop(crop_hw) if crop_is_random else self.center_crop
            )
            eff_hw = crop_hw
        else:
            self.center_crop = None
            self.random_crop = None
            eff_hw = image_hw

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(torchvision.models, backbone_name)(weights=weights)
        # Drop avgpool + fc (children()[-2:]), keep the conv trunk.
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
        if use_group_norm:
            _replace_bn_with_gn(self.backbone)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, *eff_hw)
            feat_map_shape = self.backbone(dummy).shape[1:]  # (C, H, W)

        self.pool = SpatialSoftmax(tuple(feat_map_shape), num_kp=num_keypoints)
        self.feature_dim = self.pool.out_dim
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) float in [0, 1]. Returns (B, feature_dim)."""
        if self.crop_hw is not None:
            img = self.random_crop(img) if (self.training and self.crop_is_random) else self.center_crop(img)
        feat_map = self.backbone(img)
        kp = self.pool(feat_map)
        return self.relu(self.out(kp))


class MultiCameraEncoder(nn.Module):
    """One independent RgbEncoder per camera key -- weights are NOT shared
    across cameras, since `head` (global scene view) and `left_hand`
    (wrist close-up) see structurally different content. Output is the
    per-camera features concatenated in `camera_keys` order."""

    def __init__(self, camera_keys: Optional[List[str]] = None, **encoder_kwargs) -> None:
        super().__init__()
        self.camera_keys = list(camera_keys) if camera_keys is not None else list(CAMERA_KEYS)
        self.encoders = nn.ModuleDict({k: RgbEncoder(**encoder_kwargs) for k in self.camera_keys})
        self.feature_dim = sum(self.encoders[k].feature_dim for k in self.camera_keys)

    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """images: {camera_key: (B, 3, H, W)} -> (B, feature_dim)."""
        feats = [self.encoders[k](images[k]) for k in self.camera_keys]
        return torch.cat(feats, dim=-1)
