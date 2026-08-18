"""Inference server for this repo's own state-only Diffusion Policy
(training/diffusion_policy/{model.py,train.py}), NOT lerobot's DiffusionPolicy
(see dp_server.py for that one).

Runs in the training venv (torch + diffusers, no Isaac/omni, no lerobot --
see training/diffusion_policy/requirements.txt) so the Isaac-side harness
process never has to import torch. Same length-prefixed pickle-over-stdio
protocol as dp_server.py, but the message shapes are different:

Message in : {"cmd": "reset"} OR
              {"state": (22,) f32}   -- ALREADY left-arm-sliced, RAW units
                                        (not normalized -- this server
                                        normalizes with the checkpoint's own
                                        norm_stats, never recomputed)
Message out: {"ok": True} OR
              {"action_horizon": (horizon, 7) f32 list-of-lists}  -- RAW
              (unnormalized) units: xyz(3) + euler-xyz(3) + gripper(1) --
              see task/policies/diffusion_stateonly.py's ACTION ROTATION
              CONVENTION note for why this is Euler XYZ, not rotvec, despite
              the column names.
              Caller (diffusion_stateonly.py) picks how many of the horizon
              steps to actually execute (n_action_steps) before re-querying;
              this server always returns the *full* predicted horizon so
              that knob is free to sweep on the Isaac side with no server
              restart.

Usage:
    python dp_server_stateonly.py <ckpt_path> [--num-inference-steps N] [--no-ema] [--seed S]

`--num-inference-steps` overrides the checkpoint's own diffusion config
(mirrors evaluate.py's --num-inference-steps) so the num_inference_steps
sweep (16 -> 50 -> 100) needs no code change, just a different server launch
arg from diffusion_stateonly.py's DP_NUM_INFERENCE_STEPS env var.
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
    build_model_from_checkpoint,
    ddim_sample,
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


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt)
    part = ckpt["part"]
    model, cfg = build_model_from_checkpoint(ckpt, device, use_ema=not args.no_ema)
    if cfg.model.rotation_repr != "rotvec":
        raise NotImplementedError(
            f"rotation_repr={cfg.model.rotation_repr!r} not supported here (see evaluate.py)"
        )
    norm_stats = load_norm_stats_from_checkpoint(ckpt)
    part_to_idx = part_to_idx_from_checkpoint(ckpt)
    task_idx_value = part_to_idx[part]

    horizon = cfg.data.horizon
    action_dim = cfg.model.action_dim
    num_inference_steps = args.num_inference_steps or cfg.diffusion.num_inference_steps

    gen = torch.Generator(device=device).manual_seed(args.seed)

    sys.stderr.write(
        f"[dp_server_stateonly] loaded {args.ckpt} part={part} "
        f"weights={'ema' if not args.no_ema else 'raw'} horizon={horizon} "
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
            # Stateless per-query model (no RNN/EMA-in-the-loop state to
            # clear) -- reseed so each episode's sampling noise is
            # reproducible relative to episode start, not global call count.
            gen.manual_seed(args.seed)
            _write(_out, {"ok": True})
            continue

        state_raw = np.asarray(msg["state"], dtype=np.float32).reshape(1, -1)
        if state_raw.shape[1] != len(norm_stats.state_mean):
            raise ValueError(
                f"state dim mismatch: got {state_raw.shape[1]}, "
                f"checkpoint norm_stats expects {len(norm_stats.state_mean)}"
            )
        state_n = normalize_state(state_raw, norm_stats).astype(np.float32)
        state_t = torch.from_numpy(state_n).to(device)
        task_idx = torch.full((1,), task_idx_value, dtype=torch.long, device=device)

        action_n = ddim_sample(
            model, state_t, task_idx, horizon, action_dim,
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=cfg.diffusion.beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=cfg.diffusion.clip_sample,
            num_inference_steps=num_inference_steps,
            device=device,
            generator=gen,
        )
        action = unnormalize_action(action_n.cpu().numpy(), norm_stats, part)[0]  # (horizon, 7)
        _write(_out, {"action_horizon": action.astype(np.float32).tolist()})


if __name__ == "__main__":
    main()
