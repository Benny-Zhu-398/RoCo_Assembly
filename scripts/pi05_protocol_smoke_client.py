#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import struct
import subprocess
import sys

import numpy as np


def send(proc: subprocess.Popen[bytes], message: object) -> None:
    payload = pickle.dumps(message)
    assert proc.stdin is not None
    proc.stdin.write(struct.pack(">I", len(payload)) + payload)
    proc.stdin.flush()


def receive(proc: subprocess.Popen[bytes]) -> object:
    assert proc.stdout is not None
    header = proc.stdout.read(4)
    if len(header) != 4:
        raise RuntimeError("server closed before returning a protocol response")
    size = struct.unpack(">I", header)[0]
    payload = proc.stdout.read(size)
    if len(payload) != size:
        raise RuntimeError("server returned a truncated protocol response")
    return pickle.loads(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--server", required=True)
    parser.add_argument("--server-log", required=True)
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PI05_DEVICE"] = "cuda"
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    state = np.zeros(44, dtype=np.float32)

    with open(args.server_log, "wb") as server_log:
        proc = subprocess.Popen(
            [sys.executable, args.server, args.checkpoint],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=server_log,
            env=env,
        )
        try:
            send(proc, {"cmd": "reset"})
            reset = receive(proc)
            if reset != {"ok": True}:
                raise RuntimeError(f"reset failed: {reset!r}")

            send(
                proc,
                {
                    "state": state,
                    "head": image,
                    "left": image,
                    "right": image,
                    "task": "assemble parts onto the task board",
                },
            )
            reply = receive(proc)
            if not isinstance(reply, dict) or reply.get("ok") is not True:
                raise RuntimeError(f"inference failed: {reply!r}")
            action = np.asarray(reply.get("action"), dtype=np.float32)
            if action.shape != (14,):
                raise RuntimeError(f"expected action shape (14,), got {action.shape}")
            if not np.isfinite(action).all():
                raise RuntimeError("action contains non-finite values")
            print(f"reset: {reset}")
            print(f"action shape: {action.shape}")
            print(f"action finite: {bool(np.isfinite(action).all())}")
            print(f"action: {action.tolist()}")
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
            try:
                return_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                return_code = proc.wait(timeout=30)
            if return_code != 0:
                raise RuntimeError(f"server exited with status {return_code}")


if __name__ == "__main__":
    main()
