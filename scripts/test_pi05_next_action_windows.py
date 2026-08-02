"""Measure Pi0.5 observation and next_action requests over Windows SSH."""
from __future__ import annotations

import argparse
import os
import pickle
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_DIR = _REPO_ROOT / "task"
sys.path.insert(0, str(_TASK_DIR))

from policies.pi05_lerobot import (  # noqa: E402
    _display_command,
    build_pi05_sidecar_launch,
)


_DIAGNOSTIC_SERVER = (
    "/media/iam-lab/strange_external/yudongluo/pi05/scripts/"
    "pi05_server_next_action_diagnostic.py"
)


def _recv(proc):
    header = proc.stdout.read(4)
    if len(header) != 4:
        raise RuntimeError("sidecar closed before sending a response")
    size = struct.unpack(">I", header)[0]
    chunks = []
    remaining = size
    while remaining:
        chunk = proc.stdout.read(remaining)
        if not chunk:
            raise RuntimeError("sidecar closed while sending a response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return pickle.loads(b"".join(chunks))


def _recv_with_timeout(proc, timeout):
    result = queue.Queue(maxsize=1)

    def read_response():
        try:
            result.put((True, _recv(proc)))
        except BaseException as exc:
            result.put((False, exc))

    threading.Thread(target=read_response, daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError(f"no sidecar response within {timeout:.0f} seconds") from exc
    if not ok:
        raise value
    return value


def _request(proc, message, timeout):
    serialize_started = time.perf_counter()
    payload = pickle.dumps(message)
    serialization_ms = (time.perf_counter() - serialize_started) * 1000.0
    round_trip_started = time.perf_counter()
    proc.stdin.write(struct.pack(">I", len(payload)) + payload)
    proc.stdin.flush()
    reply = _recv_with_timeout(proc, timeout)
    round_trip_ms = (time.perf_counter() - round_trip_started) * 1000.0
    return reply, len(payload), serialization_ms, round_trip_ms


def _close_process(proc):
    if proc is None:
        return
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()


def _load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape != (240, 320, 3):
        raise RuntimeError(f"expected image shape (240, 320, 3), got {image.shape}: {path}")
    return image


def _load_state(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"state_44d:\s*\[(.*?)\]\s*prediction_14d:", text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"state_44d block not found: {path}")
    state = np.fromstring(match.group(1).replace(",", " "), sep=" ", dtype=np.float32)
    if state.shape != (44,):
        raise RuntimeError(f"expected 44-D state, got {state.shape}: {path}")
    if not np.isfinite(state).all():
        raise RuntimeError(f"state contains non-finite values: {path}")
    return state


def _validate_action(reply, request_index):
    if not reply.get("ok", True):
        raise RuntimeError(f"request {request_index} failed: {reply.get('error', reply)!r}")
    action = np.asarray(reply["action"], dtype=np.float64).reshape(-1)
    if action.shape != (14,):
        raise RuntimeError(f"request {request_index}: expected (14,), got {action.shape}")
    if not np.isfinite(action).all():
        raise RuntimeError(f"request {request_index}: action contains non-finite values")
    return action


def _array_text(values):
    return np.array2string(
        np.asarray(values),
        precision=9,
        separator=", ",
        suppress_small=False,
        threshold=1000,
        max_line_width=240,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--observation-log",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "pi05_one_step_dry_run.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "pi05_next_action_timing.log",
    )
    args = parser.parse_args()

    images = {
        "head": _load_rgb(_REPO_ROOT / "artifacts" / "pi05_dry_run_head.png"),
        "left": _load_rgb(_REPO_ROOT / "artifacts" / "pi05_dry_run_left.png"),
        "right": _load_rgb(_REPO_ROOT / "artifacts" / "pi05_dry_run_right.png"),
    }
    state = _load_state(args.observation_log.resolve())

    os.environ["PI05_REMOTE"] = "1"
    os.environ.setdefault("PI05_REMOTE_SERVER", _DIAGNOSTIC_SERVER)
    command, env, checkpoint, mode = build_pi05_sidecar_launch()
    if mode != "remote":
        raise RuntimeError("next_action test unexpectedly selected local mode")

    server_log = Path(
        os.environ.get(
            "PI05_SERVER_LOG",
            str(_REPO_ROOT / "artifacts" / "pi05_next_action_server.log"),
        )
    ).resolve()
    server_log.parent.mkdir(parents=True, exist_ok=True)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    proc = None
    command_display = _display_command(command)
    try:
        with server_log.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=env,
                cwd=_TASK_DIR,
            )
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("failed to open sidecar pipes")

            reset_reply, _, _, reset_ms = _request(
                proc, {"cmd": "reset", "task": None}, args.timeout
            )
            if reset_reply != {"ok": True}:
                raise RuntimeError(f"unexpected reset reply: {reset_reply!r}")

            messages = [
                {
                    "state": state,
                    **images,
                    "task": "assemble parts onto the task board",
                },
                *[{"cmd": "next_action"} for _ in range(4)],
            ]
            previous_action = None
            for request_index, message in enumerate(messages, start=1):
                reply, request_bytes, serialization_ms, round_trip_ms = _request(
                    proc, message, args.timeout
                )
                action = _validate_action(reply, request_index)
                difference = None if previous_action is None else action - previous_action
                records.append(
                    {
                        "index": request_index,
                        "kind": "full_observation" if request_index == 1 else "next_action",
                        "request_bytes": request_bytes,
                        "serialization_ms": serialization_ms,
                        "round_trip_ms": round_trip_ms,
                        "action": action,
                        "difference": difference,
                    }
                )
                previous_action = action
    except BaseException as exc:
        returncode = proc.poll() if proc is not None else "not started"
        raise RuntimeError(
            f"next_action remote diagnostic failed: {exc}\n"
            f"ssh command: {command_display}\n"
            f"return code: {returncode}\n"
            f"server log: {server_log}"
        ) from exc
    finally:
        _close_process(proc)

    lines = [
        "pi0.5 next_action Windows SSH diagnostic",
        "safety: saved real observation only; no Isaac, IK, controller, or robot action",
        f"checkpoint: {checkpoint}",
        f"reset_round_trip_ms: {reset_ms:.3f}",
        f"state_44d: {_array_text(state)}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"request_index: {record['index']}",
                f"request_kind: {record['kind']}",
                f"request_bytes: {record['request_bytes']}",
                f"pickle_serialization_ms: {record['serialization_ms']:.3f}",
                f"ssh_server_round_trip_ms: {record['round_trip_ms']:.3f}",
                f"action_14d: {_array_text(record['action'])}",
            ]
        )
        difference = record["difference"]
        if difference is None:
            lines.append("difference_from_previous: N/A")
        else:
            lines.extend(
                [
                    f"difference_from_previous: {_array_text(difference)}",
                    f"difference_l2_norm: {np.linalg.norm(difference):.9f}",
                    f"difference_max_abs: {np.max(np.abs(difference)):.9f}",
                ]
            )
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"next_action diagnostic passed: {len(records)} actions, all finite")
    print(f"timing log: {args.output}")
    print(f"server log: {server_log}")


if __name__ == "__main__":
    main()
