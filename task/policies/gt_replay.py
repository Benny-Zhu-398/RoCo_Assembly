"""Closed-loop GT-action-replay "policy": ignores any trained model entirely
and replays a recorded val-split episode's actual left-arm actions through
the SAME run_pick_place.py harness (same IK controller, same physics, same
grading) that a learned policy would run through.

This is debugging step 1 from the battery_size1 rollout-failure triage: if
replaying ground-truth actions through this harness ALSO fails, the failure
is in the control stack (IK/PD tracking, timing, gripper-release logic), not
in the trained policy -- and iterating on the model is a waste of time until
this passes. If GT replay succeeds reliably, the failure is policy-side
(covariate shift, bad state construction, chunking/inference-step choices --
see diffusion_stateonly.py), not control-side.

Needs a .npz produced by training/diffusion_policy/export_val_episodes.py
(see that script's docstring for why the export step exists rather than
reading parquet directly here: keeps pandas/pyarrow out of Isaac's env).

Only ONE part is replayed per run; every other pc.part_order entry is
skipped immediately (mirrors diffusion_stateonly.py's skip logic) so a run
reaches the target part in a handful of sim-seconds. Point --max-parts at
the target part's 1-based index in pc.part_order to also stop the harness
right after grading it (battery_size1 = 8th entry -> --max-parts 8).

To get an actual success-RATE (not one data point), launch run_pick_place.py
once per trial with a different GT_REPLAY_EPISODE_IDX (0..n_val_episodes-1,
wrap if you want more trials than val episodes have) -- each launch is a
fresh Isaac Sim process, there's no in-process multi-episode loop here (the
harness only spawns each part's scene state once per process; see
run_pick_place.py's module docstring).

CAVEAT: the sim always spawns battery_size1 at the SAME fixed pose from
part_init_poses.json -- there is no per-episode spawn-pose randomization in
this harness. If the recorded episodes' initial arm/part configuration
varied at collection time, replaying a different episode's absolute-frame
actions against this one fixed spawn pose can be systematically off in a way
that has nothing to do with control lag. Cross-check against
training/diffusion_policy/speed_error_analysis.py's control-lag numbers
(||action_xyz[af]-state_xyz[af+1]||) computed straight from the recorded
data (no sim involved) if replay failures don't look like a pose mismatch.

Env vars:
  GT_REPLAY_NPZ           path to the exported .npz (required)
  GT_REPLAY_EPISODE_IDX   index into the npz's sorted episode_ids array
                           (default 0) -- vary this per trial to sample
                           different val episodes across a multi-trial run
  GT_REPLAY_TARGET_PARTS  comma-separated part names to replay; default is
                           whatever part the npz was exported for
  GT_REPLAY_HOLD_STEPS    physics steps to hold the last recorded action
                           after the episode's last frame, before declaring
                           is_done() (default 100 = 0.5s at 200Hz -- lets
                           gravity-settle / snap detection register instead
                           of cutting off exactly on the last recorded frame)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402

DATASET_FPS = 10.0  # training/diffusion_policy/constants.py::DATASET_FPS


def _rotvec_to_quat_wxyz(rx, ry, rz):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_rotvec([rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class GTReplayPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        self._physics_dt = env_info.physics_dt

        npz_path = os.environ.get("GT_REPLAY_NPZ")
        if not npz_path:
            raise ValueError("set GT_REPLAY_NPZ to a .npz from export_val_episodes.py")
        data = np.load(npz_path)
        self._part = str(data["part"])
        self._episode_ids = sorted(int(i) for i in data["episode_ids"])
        ep_idx_pos = int(os.environ.get("GT_REPLAY_EPISODE_IDX", "0")) % len(self._episode_ids)
        chosen_ep = self._episode_ids[ep_idx_pos]
        self._actions = np.asarray(data[f"ep_{chosen_ep}"], dtype=np.float64)  # (L, 7) raw units
        self._chosen_ep = chosen_ep

        target_parts_env = os.environ.get("GT_REPLAY_TARGET_PARTS")
        self._target_parts = (
            {p.strip() for p in target_parts_env.split(",") if p.strip()}
            if target_parts_env else {self._part}
        )
        self._hold_steps = int(os.environ.get("GT_REPLAY_HOLD_STEPS", "100"))

        steps_per_frame_f = (1.0 / DATASET_FPS) / self._physics_dt
        self._steps_per_frame = max(1, round(steps_per_frame_f))
        print(f"[gt-replay] part={self._part} episode={chosen_ep} "
              f"({ep_idx_pos + 1}/{len(self._episode_ids)} val episodes) "
              f"n_frames={len(self._actions)} steps_per_frame={self._steps_per_frame} "
              f"targets={sorted(self._target_parts)}", flush=True)

        self._skip = True
        self._step_count = 0

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._skip = target.name not in self._target_parts
        self._step_count = 0

    def _frame_for_step(self, step: int) -> int:
        return min(step // self._steps_per_frame, len(self._actions) - 1)

    def act(self, obs: Observation):
        if self._skip:
            return None
        frame = self._frame_for_step(self._step_count)
        a = self._actions[frame]
        self._step_count += 1
        pos = a[:3]
        quat = _rotvec_to_quat_wxyz(a[3], a[4], a[5])
        grip = float(a[6])  # raw joint radians, recorded verbatim -- see diffusion_stateonly.py's GRIPPER UNITS note
        return self.L.forward(pos, quat, grip)

    def is_done(self, obs: Observation) -> bool:
        if self._skip:
            return True
        total_frame_steps = len(self._actions) * self._steps_per_frame
        return self._step_count >= total_frame_steps + self._hold_steps

    def __del__(self):
        pass
