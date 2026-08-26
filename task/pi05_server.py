"""Pi0.5 inference server for the LeRobot environment.

Runs outside Isaac Sim's Python environment. The Isaac-side policy talks to
this process through a length-prefixed pickle protocol.
"""
# ruff: noqa: E402
from __future__ import annotations

import os
import pickle
import struct
import sys
import time
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION

# Keep the state slicing rule shared with the left-only dataset builder.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "training",
        "diffusion_policy",
    ),
)
from constants import LEFT_STATE_IDX, STATE_DIM_FULL  # noqa: E402


if len(sys.argv) != 2:
    raise SystemExit("usage: python pi05_server.py /path/to/checkpoint/pretrained_model")

CKPT = sys.argv[1]
DEV = os.environ.get("PI05_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TASK = os.environ.get("PI05_TASK", "assemble parts onto the task board")

policy = PI05Policy.from_pretrained(CKPT)
policy.eval().to(DEV)
ACTION_DIM = int(policy.config.output_features[ACTION].shape[0])

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CKPT,
    pretrained_revision=getattr(policy.config, "pretrained_revision", None),
    preprocessor_overrides={"device_processor": {"device": DEV}},
    postprocessor_overrides={"device_processor": {"device": "cpu"}},
)

def _expected_state_dim():
    """Read the true training state width from the normalizer statistics."""
    for step in preprocessor.steps:
        stats = getattr(step, "stats", None)
        if stats and "observation.state" in stats:
            return int(stats["observation.state"]["mean"].shape[0])
    raise SystemExit("no observation.state normalization stats found in preprocessor")


STATE_DIM = _expected_state_dim()
if STATE_DIM == len(LEFT_STATE_IDX):
    STATE_SLICE = list(LEFT_STATE_IDX)
elif STATE_DIM == STATE_DIM_FULL:
    STATE_SLICE = None
else:
    raise SystemExit(
        f"checkpoint expects unsupported {STATE_DIM}-D state; expected "
        f"{len(LEFT_STATE_IDX)} or {STATE_DIM_FULL}"
    )


def _adapt_state(raw):
    """Accept the client's full state and slice it for left-only checkpoints."""
    state = np.asarray(raw, dtype=np.float32).reshape(-1)
    if state.shape[0] == STATE_DIM:
        return state
    if STATE_SLICE is not None and state.shape[0] == STATE_DIM_FULL:
        return state[STATE_SLICE]
    raise RuntimeError(
        f"got {state.shape[0]}-D state, expected {STATE_DIM}-D"
        + (f" or full {STATE_DIM_FULL}-D" if STATE_SLICE is not None else "")
    )


sys.stderr.write(
    f"[pi05_server] loaded {CKPT} on {DEV} "
    f"action_dim={ACTION_DIM} state_dim={STATE_DIM} "
    f"right_arm={'excluded' if STATE_SLICE is not None else 'included'}\n"
)
sys.stderr.flush()

_in = sys.stdin.buffer
_out = sys.stdout.buffer
request_index = 0
queue_capacity = int(policy.config.n_action_steps)
queued_actions_remaining = 0


def _queue_state():
    return queued_actions_remaining, queue_capacity


def _action_to_numpy(action):
    if isinstance(action, dict):
        action = action[ACTION]
    action_np = action.squeeze(0).float().cpu().numpy().reshape(-1)
    if action_np.shape != (ACTION_DIM,):
        raise RuntimeError(
            f"expected {ACTION_DIM}-D pi0.5 action, got shape {action_np.shape}"
        )
    if not np.isfinite(action_np).all():
        raise RuntimeError("pi0.5 action contains non-finite values")
    return action_np


def _read():
    header = _in.read(4)
    if len(header) < 4:
        return None
    size = struct.unpack(">I", header)[0]
    buf = b""
    while len(buf) < size:
        chunk = _in.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write(obj):
    payload = pickle.dumps(obj)
    _out.write(struct.pack(">I", len(payload)) + payload)
    _out.flush()


def _img(arr):
    # HxWx3 uint8/float -> 3xHxW float32 in [0, 1].
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype == np.uint8:
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return t.float().div(255.0)
    t = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1)
    return t.float().clamp(0.0, 1.0)


while True:
    msg = _read()
    if msg is None:
        break
    command = msg.get("cmd")
    if command == "reset":
        queue_before, queue_maxlen = _queue_state()
        policy.reset()
        queued_actions_remaining = 0
        queue_after, _ = _queue_state()
        sys.stderr.write(
            f"[pi05_server] reset queue_before={queue_before} "
            f"queue_after={queue_after} queue_maxlen={queue_maxlen}\n"
        )
        sys.stderr.flush()
        _write({"ok": True})
        continue

    if command == "next_action":
        request_index += 1
        queue_before, queue_maxlen = _queue_state()
        if queue_before is None or queue_before <= 0:
            error = "action queue is empty; send a full observation request first"
            sys.stderr.write(
                f"[pi05_server] request={request_index} cmd=next_action "
                f"queue_before={queue_before} queue_maxlen={queue_maxlen} "
                f"error={error}\n"
            )
            sys.stderr.flush()
            _write({"ok": False, "error": error})
            continue

        with torch.inference_mode():
            started = time.perf_counter()
            action = policy.select_action({})
            select_action_ms = (time.perf_counter() - started) * 1000.0
            queued_actions_remaining -= 1

            started = time.perf_counter()
            action = postprocessor(action)
            postprocessor_ms = (time.perf_counter() - started) * 1000.0
        action_np = _action_to_numpy(action)
        queue_after, _ = _queue_state()
        sys.stderr.write(
            f"[pi05_server] request={request_index} cmd=next_action "
            f"queue_before={queue_before} queue_after={queue_after} "
            f"queue_maxlen={queue_maxlen} select_action_ms={select_action_ms:.3f} "
            f"postprocessor_ms={postprocessor_ms:.3f}\n"
        )
        sys.stderr.flush()
        _write({"ok": True, "action": action_np.tolist()})
        continue

    if command not in (None, "observation", "replan"):
        _write({"ok": False, "error": f"unknown command: {command!r}"})
        continue

    try:
        exec_horizon = int(msg.get("exec_horizon", 1))
    except (TypeError, ValueError):
        _write({"ok": False, "error": "exec_horizon must be an integer"})
        continue
    max_horizon = int(policy.config.n_action_steps)
    if not 1 <= exec_horizon <= max_horizon:
        _write(
            {
                "ok": False,
                "error": f"exec_horizon must be between 1 and {max_horizon}",
            }
        )
        continue

    # A full observation is always a closed-loop replan. Only next_action is
    # allowed to consume an existing queue without images/state.
    queue_before_replan, _ = _queue_state()
    discarded_actions = queue_before_replan or 0
    policy.reset()
    queued_actions_remaining = 0

    obs = {
        "observation.state": torch.as_tensor(
            _adapt_state(msg["state"]), dtype=torch.float32
        ),
        "observation.images.head": _img(msg["head"]),
        "observation.images.left_hand": _img(msg["left"]),
        "observation.images.right_hand": _img(msg["right"]),
        "task": msg.get("task", TASK),
    }

    request_index += 1
    queue_before, queue_maxlen = _queue_state()
    with torch.inference_mode():
        started = time.perf_counter()
        batch = preprocessor(obs)
        preprocessor_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        action = policy.select_action(batch)
        select_action_ms = (time.perf_counter() - started) * 1000.0
        queued_actions_remaining = (
            queue_capacity - 1
            if queue_before == 0
            else max(queue_before - 1, 0)
        )
        queue_after, _ = _queue_state()

        started = time.perf_counter()
        action = postprocessor(action)
        postprocessor_ms = (time.perf_counter() - started) * 1000.0
    action_np = _action_to_numpy(action)
    action_chunk = [action_np]
    queued_select_action_ms = 0.0
    queued_postprocessor_ms = 0.0
    for _ in range(1, exec_horizon):
        queue_remaining, _ = _queue_state()
        if not queue_remaining:
            break
        with torch.inference_mode():
            started = time.perf_counter()
            queued_action = policy.select_action({})
            queued_select_action_ms += (time.perf_counter() - started) * 1000.0
            queued_actions_remaining -= 1

            started = time.perf_counter()
            queued_action = postprocessor(queued_action)
            queued_postprocessor_ms += (time.perf_counter() - started) * 1000.0
        action_chunk.append(_action_to_numpy(queued_action))
    queue_after_chunk, _ = _queue_state()
    sys.stderr.write(
        f"[pi05_server] request={request_index} cmd=replan "
        f"queue_before_replan={queue_before_replan} "
        f"queue_before={queue_before} queue_after={queue_after} "
        f"queue_maxlen={queue_maxlen} preprocessor_ms={preprocessor_ms:.3f} "
        f"select_action_ms={select_action_ms:.3f} "
        f"postprocessor_ms={postprocessor_ms:.3f} exec_horizon={exec_horizon} "
        f"chunk_length={len(action_chunk)} discarded_actions={discarded_actions} "
        f"queue_after_chunk={queue_after_chunk} "
        f"queued_select_action_ms={queued_select_action_ms:.3f} "
        f"queued_postprocessor_ms={queued_postprocessor_ms:.3f}\n"
    )
    sys.stderr.flush()
    reply = {"ok": True, "action": action_np.tolist()}
    if exec_horizon > 1:
        reply["actions"] = [item.tolist() for item in action_chunk]
    _write(reply)
