# State-only Diffusion Policy (stage 1: single part)

Diffusion Policy (Chi et al.) with FiLM task conditioning, trained on
`tools/roco2026_by_part`. State-only (no vision encoder). Stage-1
curriculum = one independent model per part; `TaskEncoder` is fully wired
to all 9 parts so stage-2/3 ("one group" / "three groups") only need to
change which part indices a training run sees.

## Environment

This is **not** part of the repo's Isaac Sim `uv` project (root
`pyproject.toml` is pinned to `isaacsim` + `numpy<2` on Python 3.11 only,
and has no torch/diffusers). Set up a separate venv:

```bash
python -m venv .venv-dp && source .venv-dp/bin/activate   # or your usual tooling
pip install -r training/diffusion_policy/requirements.txt
```

## Rotation conventions: two datasets, two conventions (easy to re-break -- read this)

This repo has **two** sources of teleop/policy data with **different**
action-rotation encodings. Mixing them up is exactly the bug that shipped
once already (see below), so it's called out here instead of only in code
comments.

| dataset | action rotation dims (3 numbers) | how we know | consumers |
|---|---|---|---|
| `tools/roco2026_by_part` (sliced from the public HF dataset `rocochallenge2025/rocochallenge2026_Industrial_Assembly` by `tools/segment_by_part.py`) | **Euler XYZ, extrinsic** (`R = Rz(rz) @ Ry(ry) @ Rx(rx)`, scipy's lowercase `'xyz'`) | Empirically confirmed in `rotation_convention_audit.py`: for every frame of every part, decode the action's 3 rotation dims both as rotvec and as Euler XYZ, convert to a rotation matrix, and compare (geodesic degrees) against the *same frame's* state quaternion. Pooled across all 9 parts: rotvec decode gives median/mean/p90 = **1.0 / 16.1 / 60.9 deg**; Euler-XYZ-extrinsic decode gives **0.6 / 4.0 / 3.4 deg**. Small per-step rotations look similar under either convention (hence rotvec's deceptively small *median*); larger reorientations diverge sharply, which is what the *tail* (p90) exposes. | Everything in `training/diffusion_policy/` (this whole package trains only on this dataset) and `task/policies/diffusion_stateonly.py` + `task/policies/gt_replay.py` on the deployment side. |
| self-collected `task/collect_lerobot_v3.py` / `collect_lerobot_v4.py` output | **rotvec (axis-angle)** | Stated directly in the dataset's own metadata: `"absolute_cartesian_target_xyz_rotvec_gripper"`. | `task/policies/act_eval_usb.py`, `act_eval_gear.py`, `act_eval.py` (trained on `training/build_gear_act_dataset.py`-merged versions of this data) -- their rotvec decode is correct, do not change it. |

**The bug this caused:** `task/policies/diffusion_stateonly.py` (the
state-only DP deployment adapter, trained on `roco2026_by_part`) decoded
the action's rotation dims as rotvec via `Rotation.from_rotvec(...)`. Per
the table above, that's wrong for this dataset. Most frames have small
per-step rotation deltas where the two conventions happen to agree closely
(hence the model still mostly "worked"), but on larger reorientations the
decoded orientation was off by tens of degrees -- enough that Lula IK
rejected the target pose, the controller held the last good command, and
the part timed out with the arm frozen. `task/policies/gt_replay.py` (built
to replay ground-truth actions and isolate control-stack bugs from
policy-side bugs) had the *identical* decode bug, which would have made it
look like a control-stack failure even on a perfect policy. Both are now
fixed to decode Euler XYZ extrinsic (see `_euler_xyz_to_quat_wxyz` in each
file). `evaluate.py` and `sanity_check.py`'s rotation error metric had the
same rotvec assumption baked into `inference_utils.geodesic_angle_deg` --
switched to the new `geodesic_angle_deg_euler_xyz`; expect *larger* reported
angle errors after this fix (the old metric was undercounting real error
in the same tail-heavy way, not just at deployment time), not smaller.

**Two things this fix deliberately did NOT touch:** (1) `dataset.py`'s
default training path (`rotation_repr="rotvec"`) regresses the 3 raw
numbers verbatim with no geometric interpretation, so training itself was
never affected -- only code that *interprets* those numbers as a rotation
(deployment IK, error metrics) was wrong, which is also why no checkpoint
needed retraining. (2) `policies/act_eval_usb.py` / `act_eval_gear.py` /
`act_eval.py` and `collect_lerobot_v3.py`/`v4.py` -- confirmed true rotvec,
left alone.

**Still open / unverified:** `policies/diffusion_lerobot.py` and
`policies/pi05_lerobot.py` also decode their action rotation as rotvec, but
this repo has no record of which dataset their checkpoints were actually
trained/fine-tuned on -- both are flagged in-file. Confirm before trusting
either for a real deployment. `rotation_utils.py`'s `rotvec_to_rot6d` (used
only by `dataset.py`'s never-yet-trained `rotation_repr="rot6d"` branch) is
also wrong for `roco2026_by_part` as written -- fix it first if that branch
is ever actually used.

## One-time setup (run once, not per part)

```bash
cd training/diffusion_policy

# 1. Verify the right-arm action slice (index 7..13) really is ~constant
#    per part, and dump the fallback constant used to reassemble the full
#    14-D action at execution time. Reports, never silently changes design
#    (see the printed table -- as of the last run, every part is below the
#    1e-3 std threshold except `pin`, which the script flags rather than
#    hiding).
python precheck_right_arm.py            # -> right_arm_constants.json

# 2. Compute ONE shared normalizer across all 9 parts' train splits.
#    Every single-part run below reuses this file so a later
#    single-part-vs-multi-part ablation is never confounded by different
#    normalizers.
python compute_norm_stats.py            # -> norm_stats.json
```

## Train one part

```bash
python train.py --part gear_20teeth
python train.py --part gear_60teeth
python train.py --part rod_16mm
python train.py --part bolt_8mm
python train.py --part usb_a
python train.py --part hdmi
python train.py --part pin
python train.py --part battery_size1
python train.py --part battery_size5
```

`--part` is the only thing that changes across the 9 stage-1 runs. Useful
overrides: `--horizon`, `--batch-size`, `--num-epochs`, `--lr`, `--device`,
`--rotation-repr {rotvec,rot6d}`, `--no-ema`, `--config path/to/saved.json`
(round-trips `ExperimentConfig.save`/`.load` from `config.py`).

Checkpoints land in `outputs/<part>/{epoch_NNNN.pt, final.pt}`. Each
checkpoint is self-contained: model + EMA weights, the full `config.json`,
a copy of `norm_stats.json`, the part's `right_arm_constant` entry, and
`part_to_idx`.

## Module layout

| file | contents |
|---|---|
| `constants.py` | `PART_ORDER` / `PART_TO_IDX`, state (44-D) and action (14-D) field names, the left-arm index slices (`LEFT_STATE_IDX`, `LEFT_ACTION_IDX`) |
| `config.py` | `DataConfig` / `ModelConfig` / `DiffusionConfig` / `TrainConfig` dataclasses |
| `data_io.py` | parquet loading + the deterministic episode train/val split shared by `dataset.py` and `compute_norm_stats.py` |
| `normalization.py` | `NormStats` (state mean/std, action quantile-clipped min-max), normalize/unnormalize |
| `rotation_utils.py` | quat<->rotvec<->rot6d conversions, rotvec-jump detector |
| `dataset.py` | `PartSequenceDataset` -- takes a *list* of parts (stage-1 passes one), builds `(state, action_chunk, action_is_pad, task_idx)` samples |
| `model.py` | `StateEncoder`, `TaskEncoder`, `ConditionalUnet1D` (FiLM 1D temporal U-Net), `DiffusionPolicyNet` |
| `precheck_right_arm.py` | script: right-arm action std per part -> `right_arm_constants.json` |
| `compute_norm_stats.py` | script: pooled 9-part normalizer -> `norm_stats.json` |
| `train.py` | training loop (DDPM eps-prediction, masked MSE, EMA, checkpointing) |
| `rotation_convention_audit.py` | script: empirically settles rotvec-vs-Euler-XYZ for `roco2026_by_part`'s action rotation dims (see "Rotation conventions" above) |
| `check_deploy_consistency.py` | script: offline (no Isaac) dimension-by-dimension diff between `dataset.py`/`evaluate.py`'s state/action processing and `task/policies/diffusion_stateonly.py`'s adapter path -- catches the class of silent unit/convention bug this page documents |

## Design notes / known limits (carried over from the data exploration)

- **State = 22-D, left arm only.** Right arm is constant across all parts
  in `observation.state` (dropped entirely, not even conditioned on).
- **Action = 7-D, left arm only** (`xyz + rotvec + gripper`), and is an
  **absolute** Cartesian target, not a delta -- confirmed against the data
  (`action[t] ~= state_xyz[t+1]`, median error 1.3mm). The dataset does
  **no** frame differencing.
- **Rotation conventions differ between state and action, AND the action
  convention differs between this repo's two datasets.** State orientation
  is a unit quaternion (wxyz). Action orientation for
  `tools/roco2026_by_part` (what this whole `training/diffusion_policy/`
  package trains on) is **Euler XYZ extrinsic**, not a rotation vector --
  see "Rotation conventions: two datasets, two conventions" below for the
  full story and why this was gotten wrong for a while.
  `rotation_utils.py` is rotvec-only (correct for the *other*,
  self-collected dataset) and must not be applied to
  `roco2026_by_part` action data without an explicit conversion; the two
  representations never share normalization stats either way.
- **Normalization**: state uses per-dim mean/std; action uses per-dim
  min/max from the pooled **q0.01/q0.99** quantiles (not raw min/max) so
  the ~1% of rotvec outliers don't compress everyone else's range, then
  clips to [-1, 1].
- **`rotation_repr="rot6d"`** is wired through `config.py` / `dataset.py` /
  `model.py` (action dim becomes 10) but is *not* the default and hasn't
  been trained end-to-end -- treat it as a reserved branch for a later
  ablation, not a validated path.
- **`ConditionalUnet1D`** is a from-scratch, symmetric-downsampling
  reimplementation of the Diffusion Policy 1D temporal U-Net (every down
  stage halves the time axis, every up stage doubles it back, with
  defensive length-matching on skip connections). This makes it correct
  for arbitrary `horizon` values, including non-powers-of-2, unlike the
  original paper's asymmetric last-stage variant. See `model.py`'s module
  docstring.
- **Episode lengths vary a lot by part** (`bolt_8mm` mean ~41 frames /
  max 53, `rod_16mm` mean ~73 / max 183). `PartSequenceDataset` pads
  action chunks that run past an episode's end by repeating the last
  action, and `action_is_pad` marks those steps so `train.py`'s
  `masked_mse` excludes them from the loss.
- **Duplicate consecutive frames exist** (0.6% overall, concentrated in
  `gear_20teeth` and `battery_size5`). Harmless for this training loop
  (no differencing happens), left in intentionally rather than filtered,
  so anyone adding a delta-based loss later needs to handle them.
