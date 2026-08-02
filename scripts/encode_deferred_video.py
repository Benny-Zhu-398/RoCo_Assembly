"""Encode a completed deferred rollout frame directory into MP4."""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "task"))

from deferred_video import encode_frame_directory  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("frame_dir")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    count, backend = encode_frame_directory(args.output, args.frame_dir, args.fps)
    print(
        f"encoded {count} frames with {backend} -> {os.path.abspath(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
