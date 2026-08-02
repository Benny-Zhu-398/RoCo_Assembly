"""Frame-spooled video recording for stable Isaac Sim shutdown on Windows."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np


def _rgb8(frame):
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"video frame must have shape (H, W, >=3), got {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size and float(np.max(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def encode_frame_directory(path, frame_dir, fps):
    """Encode frame_XXXXXX.png files from a completed Isaac process."""
    import cv2

    path = os.path.abspath(path)
    frame_dir = os.path.abspath(frame_dir)
    frame_names = sorted(
        name
        for name in os.listdir(frame_dir)
        if name.startswith("frame_") and name.endswith(".png")
    )
    if not frame_names:
        raise RuntimeError(f"no deferred frames found in {frame_dir}")
    first = cv2.imread(os.path.join(frame_dir, frame_names[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"could not read first deferred frame in {frame_dir}")
    height, width = first.shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        pattern = os.path.join(frame_dir, "frame_%06d.png")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            path,
        ]
        subprocess.run(command, check=True)
        backend = "ffmpeg/libx264"
    else:
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open MP4 writer: {path}")
        try:
            for frame_name in frame_names:
                frame_path = os.path.join(frame_dir, frame_name)
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(
                        f"could not read deferred frame: {frame_path}"
                    )
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"deferred frame size changed at {frame_path}: "
                        f"{frame.shape[:2]} != {(height, width)}"
                    )
                writer.write(frame)
        finally:
            writer.release()
        backend = "opencv/mp4v"
    return len(frame_names), backend


class DeferredFrameVideoRecorder:
    """Write PNG frames during simulation and encode only after Isaac exits."""

    deferred = True

    def __init__(self, path, fps=30, camera="head", frame_dir=None):
        self.path = os.path.abspath(path) if path else None
        self.fps = int(fps)
        self.camera = camera
        self.frame_dir = (
            os.path.abspath(frame_dir)
            if frame_dir
            else (self.path + ".frames" if self.path else None)
        )
        self.frames = 0
        self._shape = None

    @property
    def enabled(self):
        return bool(self.path)

    def write(self, frame):
        if not self.enabled or frame is None:
            return
        import cv2

        arr = _rgb8(frame)
        shape = tuple(arr.shape[:2])
        if self._shape is None:
            self._shape = shape
            os.makedirs(self.frame_dir, exist_ok=True)
        elif self._shape != shape:
            raise ValueError(
                f"video frame size changed from {self._shape} to {shape}"
            )
        frame_path = os.path.join(self.frame_dir, f"frame_{self.frames:06d}.png")
        if not cv2.imwrite(frame_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write video frame: {frame_path}")
        self.frames += 1

    def close(self):
        if not self.enabled:
            return
        if self.frames <= 0:
            raise RuntimeError(f"no frames captured for deferred video: {self.path}")
        frame_count, backend = encode_frame_directory(
            self.path, self.frame_dir, self.fps
        )
        print(
            f"[video] encoded {frame_count} deferred frames with {backend} "
            f"-> {self.path}",
            flush=True,
        )
