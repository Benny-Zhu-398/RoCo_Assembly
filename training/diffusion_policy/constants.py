"""Single source of truth for part ordering and state/action layout.

Every other module in this package imports these instead of redefining
them. Values are pinned by the July-2026 data exploration of
tools/roco2026_by_part (see conversation history / PARTS.md) — do not
change without re-verifying against tools/roco2026_by_part/meta/info.json.
"""
from __future__ import annotations

# Canonical part order. Matches task/param_config.py::part_order and
# tools/segment_by_part.py::PART_ORDER, and tools/roco2026_by_part/meta/tasks.parquet
# task_index assignment (gear_20teeth=0 ... battery_size5=8).
PART_ORDER = (
    "gear_20teeth",
    "gear_60teeth",
    "rod_16mm",
    "bolt_8mm",
    "usb_a",
    "hdmi",
    "pin",
    "battery_size1",
    "battery_size5",
)

# Fixed part -> task-embedding row index. TaskEncoder is an
# nn.Embedding(NUM_PARTS, emb_dim); this mapping is what turns a part name
# into the row to look up. Keep this order stable across every checkpoint
# ever trained — reordering silently invalidates old checkpoints.
PART_TO_IDX = {name: i for i, name in enumerate(PART_ORDER)}
IDX_TO_PART = {i: name for name, i in PART_TO_IDX.items()}
NUM_PARTS = len(PART_ORDER)

RELEASE_MODE = {
    "gear_20teeth": "open",
    "gear_60teeth": "open",
    "rod_16mm": "snap",
    "bolt_8mm": "snap",
    "usb_a": "snap",
    "hdmi": "snap",
    "pin": "snap",
    "battery_size1": "open",
    "battery_size5": "open",
}

# --- observation.state, 44-D (tools/roco2026_by_part/meta/info.json) ---
STATE_DIM_FULL = 44
STATE_NAMES_FULL = (
    "left_ee_x", "left_ee_y", "left_ee_z",
    "left_ee_qw", "left_ee_qx", "left_ee_qy", "left_ee_qz",
    "right_ee_x", "right_ee_y", "right_ee_z",
    "right_ee_qw", "right_ee_qx", "right_ee_qy", "right_ee_qz",
    *(f"left_jpos_{i}" for i in range(7)),
    *(f"right_jpos_{i}" for i in range(7)),
    *(f"left_jvel_{i}" for i in range(7)),
    *(f"right_jvel_{i}" for i in range(7)),
    "left_gripper", "right_gripper",
)
assert len(STATE_NAMES_FULL) == STATE_DIM_FULL

# Left-arm-only slice of the 44-D state (confirmed static-right-arm data
# exploration): ee pose (0-6) + jpos (14-20) + jvel (28-34) + gripper (42).
LEFT_STATE_IDX = list(range(0, 7)) + list(range(14, 21)) + list(range(28, 35)) + [42]
STATE_DIM = len(LEFT_STATE_IDX)  # 22
assert STATE_DIM == 22

STATE_NAMES = tuple(STATE_NAMES_FULL[i] for i in LEFT_STATE_IDX)

# Slices *within* the 22-D left state (i.e. indices into STATE_NAMES /
# a state vector already sliced by LEFT_STATE_IDX).
STATE_XYZ_SLICE = slice(0, 3)
STATE_QUAT_SLICE = slice(3, 7)      # wxyz, unit norm
STATE_JPOS_SLICE = slice(7, 14)
STATE_JVEL_SLICE = slice(14, 21)  # NOT a reliable ee-linear-speed proxy: on the
# battery_size1/gear_60teeth val sets, ||jvel|| vs. the action-diff ee linear
# speed (||action_xyz[t+1]-action_xyz[t]||*FPS) has spearman rho=-0.157 and
# -0.162 respectively (see speed_error_analysis.py) -- joint-space speed does
# not track end-effector speed here (Jacobian/redundancy effects), so any
# future velocity-based modeling (distance metrics, contact-force warm-start)
# should use the action-diff ee linear speed, not this slice's norm.
STATE_GRIPPER_IDX = 21

# --- action, 14-D (tools/roco2026_by_part/meta/info.json) ---
ACTION_DIM_FULL = 14
ACTION_NAMES_FULL = (
    "left_ee_x", "left_ee_y", "left_ee_z",
    "left_ee_rx", "left_ee_ry", "left_ee_rz", "left_gripper",
    "right_ee_x", "right_ee_y", "right_ee_z",
    "right_ee_rx", "right_ee_ry", "right_ee_rz", "right_gripper",
)
assert len(ACTION_NAMES_FULL) == ACTION_DIM_FULL

LEFT_ACTION_IDX = list(range(0, 7))
RIGHT_ACTION_IDX = list(range(7, 14))
ACTION_DIM = len(LEFT_ACTION_IDX)  # 7
assert ACTION_DIM == 7

ACTION_NAMES = tuple(ACTION_NAMES_FULL[i] for i in LEFT_ACTION_IDX)
RIGHT_ACTION_NAMES = tuple(ACTION_NAMES_FULL[i] for i in RIGHT_ACTION_IDX)

# Slices *within* the 7-D left action.
ACTION_XYZ_SLICE = slice(0, 3)
ACTION_ROT_SLICE = slice(3, 6)      # rotvec, axis-angle, NOT canonical-range
ACTION_GRIPPER_IDX = 6

DATASET_FPS = 10.0
DATASET_DEFAULT_ROOT = "tools/roco2026_by_part"
