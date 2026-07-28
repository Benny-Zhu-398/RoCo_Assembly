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

## Design notes / known limits (carried over from the data exploration)

- **State = 22-D, left arm only.** Right arm is constant across all parts
  in `observation.state` (dropped entirely, not even conditioned on).
- **Action = 7-D, left arm only** (`xyz + rotvec + gripper`), and is an
  **absolute** Cartesian target, not a delta -- confirmed against the data
  (`action[t] ~= state_xyz[t+1]`, median error 1.3mm). The dataset does
  **no** frame differencing.
- **Rotation conventions differ between state and action.** State
  orientation is a unit quaternion (wxyz); action orientation is a
  rotation vector that is *not* canonical-range (samples go up to 2.54*pi).
  `rotation_utils.py` is the only place conversions should happen; the two
  never share normalization stats.
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
