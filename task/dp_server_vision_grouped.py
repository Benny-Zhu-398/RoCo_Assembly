"""Inference server for this repo's own GROUPED VISION-conditioned Diffusion
Policy (training/diffusion_policy/{grouped_model.py,train.py}'s --group
path), NOT the single-part vision checkpoint (see dp_server_vision.py for
that one). Same process-isolation pattern, same length-prefixed
pickle-over-stdio protocol as dp_server_vision.py, but the message shape
differs in one way: a grouped checkpoint answers for every part in
ckpt["parts"] (e.g. connectors = {usb_a, hdmi}), not a single part fixed at
process startup, so the "reset" message must say which part the upcoming
episode is for -- see grouped_model.py's module docstring, "SCOPE OF THIS
PASS": this was the deferred protocol change that note called out
("dp_server_stateonly.py's protocol in particular would need each query to
carry which part it's for").

Message in : {"cmd": "reset", "part": <str, one of ckpt["parts"]>} OR
              {"state": (22,) f32,          -- ALREADY left-arm-sliced, RAW
                                                units (not normalized)
               "images": {"head": (H,W,3) uint8, "left_hand": (H,W,3) uint8}}
                                             -- RAW camera resolution is fine;
                                                this server resizes to match
                                                the checkpoint's own training-
                                                time resolution
                                                (cfg.data.image_resize_hw or
                                                native 240x320) before the
                                                model ever sees them. Requires
                                                a prior "reset" in this
                                                process's lifetime (there is
                                                no default part).
Message out: {"ok": True} OR
              {"action_horizon": (horizon, 7) f32 list-of-lists}  -- RAW
              (unnormalized) units: xyz(3) + euler-xyz(3) + gripper(1) --
              see task/policies/diffusion_vision.py's ACTION ROTATION
              CONVENTION note for why this is Euler XYZ, not rotvec, despite
              the column names. Unnormalized using the CURRENT part's own
              gripper range (set by the most recent "reset"), same per-part
              norm-stat handling as the single-part server.

Usage:
    python dp_server_vision_grouped.py <ckpt_path> [--num-inference-steps N] [--no-ema] [--seed S]

Refuses to load a checkpoint with ckpt["group"] is None -- that's a
single-part checkpoint, use dp_server_vision.py instead.
"""
from __future__ import annotations

import argparse
import os
import pickle
import struct
import sys
import warnings

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DP_DIR = os.path.join(os.path.dirname(_THIS_DIR), "training", "diffusion_policy")
if _DP_DIR not in sys.path:
    sys.path.insert(0, _DP_DIR)

import numpy as np
import torch

from inference_utils import (  # noqa: E402
    build_grouped_model_from_checkpoint,
    ddim_sample_grouped,
    load_checkpoint,
    load_norm_stats_from_checkpoint,
    part_to_idx_from_checkpoint,
)
from normalization import normalize_state, unnormalize_action  # noqa: E402


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--num-inference-steps", type=int, default=None,
                     help="override the checkpoint's config value (sweep target)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _read(inp):
    h = inp.read(4)
    if len(h) < 4:
        return None
    n = struct.unpack(">I", h)[0]
    buf = b""
    while len(buf) < n:
        chunk = inp.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write(out, obj):
    b = pickle.dumps(obj)
    out.write(struct.pack(">I", len(b)) + b)
    out.flush()


def _prep_image(img_hwc_uint8: np.ndarray, target_hw, device) -> torch.Tensor:
    """(H,W,3) uint8, any resolution -> (1,3,target_h,target_w) float32 in
    [0,1] on device. Resize path matches dataset.py's _resize_rgb (torchvision,
    antialiased) so train/deploy preprocessing agree."""
    t = torch.from_numpy(np.ascontiguousarray(img_hwc_uint8)).permute(2, 0, 1).float() / 255.0
    if (t.shape[1], t.shape[2]) != tuple(target_hw):
        import torchvision.transforms.functional as TF
        t = TF.resize(t, list(target_hw), antialias=True)
    return t.unsqueeze(0).to(device)


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt)
    group = ckpt.get("group")
    if group is None:
        raise ValueError(
            f"{args.ckpt} has no group (ckpt['group'] is None) -- this is a single-part "
            "checkpoint, use dp_server_vision.py instead."
        )
    parts = list(ckpt["parts"])
    model, cfg = build_grouped_model_from_checkpoint(ckpt, device, use_ema=not args.no_ema)
    if cfg.model.rotation_repr != "rotvec":
        raise NotImplementedError(
            f"rotation_repr={cfg.model.rotation_repr!r} not supported here (see evaluate_grouped.py)"
        )
    norm_stats = load_norm_stats_from_checkpoint(ckpt)
    part_to_idx = part_to_idx_from_checkpoint(ckpt)
    camera_keys = list(cfg.model.camera_keys)
    image_hw = cfg.data.image_resize_hw or (240, 320)

    horizon = cfg.data.horizon
    action_dim = cfg.model.action_dim
    num_inference_steps = args.num_inference_steps or cfg.diffusion.num_inference_steps

    gen = torch.Generator(device=device).manual_seed(args.seed)
    current_part = None  # set by the first "reset" message

    sys.stderr.write(
        f"[dp_server_vision_grouped] loaded {args.ckpt} group={group} parts={parts} "
        f"weights={'ema' if not args.no_ema else 'raw'} horizon={horizon} "
        f"cameras={camera_keys} image_hw={image_hw} "
        f"num_inference_steps={num_inference_steps} device={device}\n"
    )
    sys.stderr.flush()

    _in = sys.stdin.buffer
    _out = sys.stdout.buffer

    while True:
        msg = _read(_in)
        if msg is None:
            break
        if msg.get("cmd") == "reset":
            part = msg.get("part")
            if part not in parts:
                raise ValueError(f"reset requested part={part!r}, but this checkpoint only covers {parts}")
            current_part = part
            # Stateless per-query model (no RNN/EMA-in-the-loop state to
            # clear) -- reseed so each episode's sampling noise is
            # reproducible relative to episode start, not global call count.
            gen.manual_seed(args.seed)
            _write(_out, {"ok": True})
            continue

        if current_part is None:
            raise RuntimeError("received a query before any 'reset' -- server does not know which part to serve")

        state_raw = np.asarray(msg["state"], dtype=np.float32).reshape(1, -1)
        if state_raw.shape[1] != len(norm_stats.state_mean):
            raise ValueError(
                f"state dim mismatch: got {state_raw.shape[1]}, "
                f"checkpoint norm_stats expects {len(norm_stats.state_mean)}"
            )
        state_n = normalize_state(state_raw, norm_stats).astype(np.float32)
        state_t = torch.from_numpy(state_n).to(device)
        task_idx = torch.full((1,), part_to_idx[current_part], dtype=torch.long, device=device)

        raw_images = msg["images"]
        images = {
            cam: _prep_image(np.asarray(raw_images[cam], dtype=np.uint8), image_hw, device)
            for cam in camera_keys
        }

        action_n = ddim_sample_grouped(
            model, state_t, task_idx, horizon, action_dim,
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=cfg.diffusion.beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=cfg.diffusion.clip_sample,
            num_inference_steps=num_inference_steps,
            device=device,
            generator=gen,
            images=images,
        )
        action = unnormalize_action(action_n.cpu().numpy(), norm_stats, current_part)[0]  # (horizon, 7)
        _write(_out, {"action_horizon": action.astype(np.float32).tolist()})


if __name__ == "__main__":
    main()
