"""Unit tests for deferred Windows rollout video encoding."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "task"))

from deferred_video import DeferredFrameVideoRecorder, _rgb8  # noqa: E402


class DeferredVideoTest(unittest.TestCase):
    def test_rgb8_converts_float_rgba(self):
        frame = np.zeros((8, 10, 4), dtype=np.float32)
        frame[..., 0] = 1.0
        converted = _rgb8(frame)
        self.assertEqual(converted.shape, (8, 10, 3))
        self.assertEqual(converted.dtype, np.uint8)
        self.assertTrue(np.all(converted[..., 0] == 255))

    def test_spools_and_encodes_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.mp4")
            recorder = DeferredFrameVideoRecorder(path, fps=10)
            recorder.write(np.zeros((16, 20, 3), dtype=np.uint8))
            recorder.write(np.full((16, 20, 3), 255, dtype=np.uint8))
            recorder.close()
            self.assertEqual(recorder.frames, 2)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
