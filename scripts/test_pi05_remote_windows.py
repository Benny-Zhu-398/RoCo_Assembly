"""Smoke-test the Pi0.5 length-prefixed pickle protocol over Windows SSH."""
from __future__ import annotations

import argparse
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_DIR = _REPO_ROOT / "task"
sys.path.insert(0, str(_TASK_DIR))

from policies.pi05_lerobot import (  # noqa: E402
    _display_command,
    build_pi05_sidecar_launch,
)


def _send(proc, obj):
    payload = pickle.dumps(obj)
    proc.stdin.write(struct.pack(">I", len(payload)) + payload)
    proc.stdin.flush()


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


def _close_process(proc):
    if proc is None:
        return
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    os.environ["PI05_REMOTE"] = "1"
    command, env, checkpoint, mode = build_pi05_sidecar_launch()
    if mode != "remote":
        raise RuntimeError("remote smoke test unexpectedly selected local mode")

    log_path = Path(
        os.environ.get(
            "PI05_SERVER_LOG",
            str(_REPO_ROOT / "artifacts" / "pi05_remote_windows.log"),
        )
    ).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_display = _display_command(command)
    proc = None
    try:
        with log_path.open("w") as log_file:
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

            _send(proc, {"cmd": "reset", "task": None})
            reset_reply = _recv_with_timeout(proc, args.timeout)
            if reset_reply != {"ok": True}:
                raise RuntimeError(f"unexpected reset reply: {reset_reply!r}")

            head = np.zeros((240, 320, 3), dtype=np.uint8)
            left = np.full((240, 320, 3), 64, dtype=np.uint8)
            right = np.full((240, 320, 3), 192, dtype=np.uint8)
            state = np.zeros(44, dtype=np.float32)
            _send(
                proc,
                {
                    "state": state,
                    "head": head,
                    "left": left,
                    "right": right,
                    "task": "assemble parts onto the task board",
                },
            )
            reply = _recv_with_timeout(proc, args.timeout)
            action = np.asarray(reply["action"], dtype=np.float32)
            if action.shape != (14,):
                raise RuntimeError(f"expected action shape (14,), got {action.shape}")
            if not np.isfinite(action).all():
                raise RuntimeError(f"action contains non-finite values: {action}")
    except BaseException as exc:
        returncode = proc.poll() if proc is not None else "not started"
        raise RuntimeError(
            f"remote pi0.5 smoke test failed: {exc}\n"
            f"ssh command: {command_display}\n"
            f"return code: {returncode}\n"
            f"log file: {log_path}"
        ) from exc
    finally:
        _close_process(proc)

    print(f"remote pi0.5 smoke test passed (checkpoint={checkpoint})")
    print("action shape: (14,), all finite: true")
    print(f"log file: {log_path}")


if __name__ == "__main__":
    main()
