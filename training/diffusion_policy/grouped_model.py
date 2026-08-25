"""Multi-group FiLM-conditioned Diffusion Policy: one model spanning all 9
parts, with a shared vision+state trunk, a per-part task embedding used as a
FiLM condition, and deterministic hard routing (by part name, via
constants.PART_TO_GROUP) to one of GROUP_ORDER's skill heads. Design is the
2026-08-18 architecture discussion's diagram:

    RGB image ---> Vision encoder (ResNet-18) --+
                                                  +--> shared features f_t
    Joint state -> State encoder (MLP) ---------+

    Task spec (part id) -> Task encoder (embed) -> condition c
    c FiLM-modulates f_t (NOT the other way around -- c never carries
    position, f_t is where the vision pathway's spatial grounding lives;
    see the module-level note below on why that split matters)

    f_t' (FiLM-modulated) --route by part--> skill_heads[group] (its own
    ConditionalUnet1D, reusing model.py's DDPM/DDIM denoiser unchanged) -> a_t^BC

    snap-type groups (constants.SNAP_GROUPS) only:
        a_t^BC -> + residual_policies[group](a_t^BC) -> a_t^final
        open-type groups (constants.OPEN_GROUPS) skip this, a_t^final = a_t^BC

=== WHY c (task encoder output) MUST NOT be the thing carrying object
position ===

This split exists because of a diagnosed failure mode in the *single*-part,
state-only checkpoints (training/diffusion_policy/outputs/*): their
`observation.state` (see constants.STATE_NAMES_FULL) has no object-pose
field at all -- only the robot's own proprioception. Per-timestep eval error
(evaluate.py) on those checkpoints spikes hard (p95 angle error jumps from a
few degrees at the chunk's first predicted step to 44-56 degrees by t=1-8)
exactly where per-episode trajectories should diverge based on the (unseen)
part's actual placement -- consistent with a state-only model regressing to
an averaged, part-placement-blind trajectory. The task/part-id condition c
here is a category label ("which skill"), not a position signal, and adding
position info to it (e.g. by learning some proxy through the id embedding)
would silently re-derive the same blind-averaging failure one level up.
Object position must come from the vision pathway (f_t) or not at all.

=== ROUTING IS DETERMINISTIC, NOT LEARNED ===

Which skill head processes a sample is decided from the SAME externally-given
part label the harness already provides every step (task/policy_api.py's
PartTarget at deploy time; the dataset's own per-episode part/task_index
label at train time) via constants.PART_TO_GROUP -- see that constant's
docstring. The model is never asked to infer which part it's looking at from
pixels; routing and the task-embedding lookup are two independent table
lookups keyed by the same known part name.

=== SCOPE OF THIS PASS (2026-08-18) ===

Implemented: shared trunk, FiLM conditioning, per-group ConditionalUnet1D
skill heads with mixed-group-batch support (gather/compute/scatter by
group), residual-policy call sites for SNAP_GROUPS.

As of 2026-08-18, train.py IS wired to this module (`--group {gears,batteries,
connectors,fasteners,all}`, mutually exclusive with `--part`; see
config.py::DataConfig.group / resolved_parts()) -- dataset.py needed no
changes at all, PartSequenceDataset already accepted a list of parts (see
its own module docstring: "Built to take a *list* of parts"). train.py's
training LOOP body (noise scheduling, masked_mse loss, EMA, optimizer) is
UNCHANGED from the single-part path -- no per-group loss masking was needed,
because routing already happens inside forward()/predict_noise() per sample,
so whatever comes back is already the right shape/semantics for the same
masked_mse call every other checkpoint uses. Verified end-to-end against
real data (`--group gears --use-vision`): dataset construction, model
construction, a real training step with decreasing loss, all run clean.

Deliberately NOT implemented yet (tracked here, not silently assumed away):
  - ResidualPolicyStub.forward returns an all-zero correction. The real
    J^dagger_lambda . delta_x_t task-space correction (what delta_x_t is
    measured from, how J^dagger_lambda is computed, and whether pi_res is
    trained via RL or supervised) is an explicitly separate, later design
    decision -- see that class's docstring.
  - inference_utils.ddim_sample / dp_server_stateonly.py / evaluate.py are
    NOT adapted to this module yet -- they assume a checkpoint serves
    exactly one fixed part (loaded once from ckpt["part"]) and call a plain
    `model.unet(...)` / `model.global_cond(...)`, neither of which exists on
    GroupedDiffusionPolicyNet (it has predict_noise()/shared_condition()
    instead, and a checkpoint now spans multiple parts via ckpt["parts"]).
    A trained grouped checkpoint can't be evaluated or deployed until this
    adapter work happens -- training does not depend on it, but inference
    does. dp_server_stateonly.py's protocol in particular would need each
    query to carry which part it's for (today it's fixed once at process
    startup from the checkpoint), since one grouped checkpoint now answers
    for every part in its `parts` list, not just one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import ModelConfig  # noqa: E402
from constants import GROUP_ORDER, GROUP_TO_IDX, PART_TO_GROUP, PART_ORDER, SNAP_GROUPS  # noqa: E402
from model import ConditionalUnet1D, StateEncoder, TaskEncoder  # noqa: E402
from vision import MultiCameraEncoder  # noqa: E402

# constants.PART_TO_GROUP is keyed by part name; this is the same lookup
# re-expressed by task_idx (constants.PART_TO_IDX row number) since that's
# what flows through the model as a tensor. Built once here, not per-call.
_TASK_IDX_TO_GROUP_IDX = torch.tensor(
    [GROUP_TO_IDX[PART_TO_GROUP[part]] for part in PART_ORDER], dtype=torch.long
)


def task_idx_to_group_idx(task_idx: torch.Tensor) -> torch.Tensor:
    """(B,) task_idx (row into PART_TO_IDX) -> (B,) group_idx (row into
    GROUP_ORDER), via the fixed constants.PART_TO_GROUP table."""
    return _TASK_IDX_TO_GROUP_IDX.to(task_idx.device)[task_idx]


class FiLM(nn.Module):
    """out = f * (1 + scale(c)) + shift(c). The `1 +` centers the scale
    branch at identity, and both branches are zero-initialized, so a
    freshly-constructed FiLM layer starts as a no-op (out == f) -- the shared
    trunk's features aren't clobbered by an untrained conditioning head
    before the task embedding has learned anything useful."""

    def __init__(self, feature_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, feature_dim * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)
        self._feature_dim = feature_dim

    def forward(self, f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(c).split(self._feature_dim, dim=-1)
        return f * (1.0 + scale) + shift


class ResidualPolicyStub(nn.Module):
    """Placeholder for pi_res, the snap-group-only task-space residual
    correction (diagram: a_t^final = a_t^BC + J^dagger_lambda . delta_x_t).

    Scaffolding only -- see this module's SCOPE OF THIS PASS note. Returns an
    all-zero correction of shape (..., task_space_dim), i.e. every caller
    site behaves exactly like a_t^final = a_t^BC until this is filled in, so
    wiring this into GroupedDiffusionPolicyNet now costs nothing later:
    swapping in the real J^dagger_lambda . delta_x_t computation only
    requires replacing this class's forward(), no call-site changes.

    task_space_dim=6 (xyz + 3 rotation dims) deliberately excludes the
    gripper dim (constants.ACTION_GRIPPER_IDX) -- a Jacobian-pseudoinverse
    task-space correction has no meaning for the gripper's 1-DOF joint
    value, so GroupedDiffusionPolicyNet.apply_residual only ever adds this
    into a_t^BC's first 6 dims.

    NOT decided by this pass: what delta_x_t actually measures (force/torque
    residual? vision-estimated pose error against a target snap-site pose?
    something else?), how J^dagger_lambda (damped-least-squares Jacobian
    pseudoinverse) is obtained (Isaac Sim kinematics at deploy time -- there
    is no such thing offline), and whether pi_res is trained supervised or
    via residual RL. Any of those choices changes this class's __init__
    signature and forward()'s inputs, not just its body.
    """

    TASK_SPACE_DIM = 6

    def __init__(self, task_space_dim: int = TASK_SPACE_DIM) -> None:
        super().__init__()
        self.task_space_dim = task_space_dim

    def forward(self, bc_action: torch.Tensor, context: Optional[dict] = None) -> torch.Tensor:
        """bc_action: (..., action_dim), only used for shape/device/dtype.
        context: reserved for whatever delta_x_t / Jacobian inputs the real
        implementation ends up needing (force-torque reading, target-pose
        error, current Jacobian, ...) -- unused by this stub."""
        return torch.zeros(
            (*bc_action.shape[:-1], self.task_space_dim),
            dtype=bc_action.dtype, device=bc_action.device,
        )


class GroupedDiffusionPolicyNet(nn.Module):
    """Shared vision+state trunk -> FiLM(c) -> per-group ConditionalUnet1D
    skill head, chosen by constants.PART_TO_GROUP. See module docstring for
    the full diagram and what is/isn't implemented yet.

    Vision is required (not optional like model.py's DiffusionPolicyNet):
    this architecture exists specifically to give the shared trunk the
    object-position grounding a state-only model doesn't have (see module
    docstring's WHY note), so cfg.use_vision=False is rejected rather than
    silently reproducing that blind spot.
    """

    def __init__(self, cfg: ModelConfig, vision_image_hw: Tuple[int, int] = (240, 320)) -> None:
        super().__init__()
        if not cfg.use_vision:
            raise ValueError(
                "GroupedDiffusionPolicyNet requires cfg.use_vision=True -- the whole point "
                "of this architecture is giving the shared trunk object-position grounding "
                "through vision; a state-only variant would reproduce the blind-averaging "
                "failure this design is meant to fix (see module docstring)."
            )
        self.cfg = cfg

        self.state_encoder = StateEncoder(cfg.state_dim, cfg.state_hidden_dim, cfg.state_feature_dim)
        self.vision_encoder = MultiCameraEncoder(
            camera_keys=list(cfg.camera_keys),
            image_hw=vision_image_hw,
            backbone_name=cfg.vision_backbone,
            pretrained=cfg.vision_pretrained,
            use_group_norm=cfg.vision_use_group_norm,
            num_keypoints=cfg.vision_num_keypoints,
            crop_hw=cfg.vision_crop_hw,
            crop_is_random=cfg.vision_crop_is_random,
        )
        shared_dim = cfg.state_feature_dim + self.vision_encoder.feature_dim

        # Per-PART (not per-group) embedding: FiLM still needs to tell
        # gear_20teeth from gear_60teeth apart even though both route to the
        # "gears" skill head -- routing (coarse, which weights run) and
        # conditioning (fine, how those weights behave) are independently
        # keyed off the same part label, see module docstring's ROUTING note.
        self.task_encoder = TaskEncoder(cfg.num_parts, cfg.task_emb_dim)
        self.film = FiLM(shared_dim, cfg.task_emb_dim)

        # global_cond handed to each ConditionalUnet1D: FiLM-modulated
        # shared features concatenated with the raw task embedding c, so the
        # UNet's own internal per-block FiLM (model.py's
        # ConditionalResidualBlock1D) also has direct access to task
        # identity, not just the once-modulated trunk features.
        global_cond_dim = shared_dim + cfg.task_emb_dim
        self.skill_heads = nn.ModuleDict({
            group: ConditionalUnet1D(
                input_dim=cfg.action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
                down_dims=cfg.down_dims,
                kernel_size=cfg.kernel_size,
                n_groups=cfg.n_groups,
            )
            for group in GROUP_ORDER
        })
        self.residual_policies = nn.ModuleDict({
            group: ResidualPolicyStub() for group in SNAP_GROUPS
        })

    def shared_condition(
        self, state: torch.Tensor, task_idx: torch.Tensor, images: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (global_cond, group_idx). global_cond is (B, shared_dim + task_emb_dim);
        group_idx is (B,) long, one GROUP_TO_IDX value per sample."""
        state_feat = self.state_encoder(state)
        vision_feat = self.vision_encoder(images)
        f = torch.cat([state_feat, vision_feat], dim=-1)
        c = self.task_encoder(task_idx)
        f_modulated = self.film(f, c)
        global_cond = torch.cat([f_modulated, c], dim=-1)
        group_idx = task_idx_to_group_idx(task_idx)
        return global_cond, group_idx

    def predict_noise(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        group_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Route each sample to its group's skill head. Supports a batch
        spanning multiple groups (gather per group, run that group's UNet
        only on its subset, scatter back into the original sample order) so
        joint multi-group training batches don't have to be pre-sorted by
        group -- a single-group batch (e.g. one query at deploy time) just
        takes the len(present_groups) == 1 fast path below."""
        B = noisy_action.shape[0]
        out = torch.empty_like(noisy_action)
        present = torch.unique(group_idx).tolist()
        for g_idx in present:
            group = GROUP_ORDER[g_idx]
            mask = group_idx == g_idx
            out[mask] = self.skill_heads[group](
                noisy_action[mask], timestep[mask] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep,
                global_cond=global_cond[mask],
            )
        return out

    def apply_residual(
        self, bc_action: torch.Tensor, group_idx: torch.Tensor, context: Optional[dict] = None,
    ) -> torch.Tensor:
        """a_t^BC -> a_t^final. Open-type groups pass through unchanged;
        snap-type groups get + residual_policies[group](...) added into the
        first ResidualPolicyStub.TASK_SPACE_DIM dims only (see that class's
        docstring for why the gripper dim is excluded)."""
        out = bc_action.clone()
        for group in SNAP_GROUPS:
            g_idx = GROUP_TO_IDX[group]
            mask = group_idx == g_idx
            if not torch.any(mask):
                continue
            correction = self.residual_policies[group](bc_action[mask], context)
            d = correction.shape[-1]
            # Single combined-index assignment (mask, :, :d) -- NOT
            # out[mask][..., :d] = ..., which silently no-ops: boolean
            # indexing on the left of a chained `[][]` returns a copy, so an
            # assignment into that copy never reaches `out`.
            out[mask, :, :d] = out[mask, :, :d] + correction
        return out

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
        task_idx: torch.Tensor,
        images: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Training-time entry point: predicted noise for the diffusion loss
        (matches model.py::DiffusionPolicyNet.forward's role). Does NOT call
        apply_residual -- the residual policy corrects a *sampled* a_t^BC
        action (see predict_action below), it has nothing to add to a
        noise-prediction target during DDPM training."""
        global_cond, group_idx = self.shared_condition(state, task_idx, images)
        return self.predict_noise(noisy_action, timestep, global_cond, group_idx)

    def predict_action(
        self, a_t_bc: torch.Tensor, task_idx: torch.Tensor, context: Optional[dict] = None,
    ) -> torch.Tensor:
        """Inference-time entry point: a fully-denoised a_t^BC (already
        produced by a DDIM sampling loop over `forward`, e.g.
        inference_utils.ddim_sample) -> a_t^final, applying the residual
        correction for snap-type groups. Kept separate from `forward` so the
        denoising loop itself never has to know about routing/residual --
        it only ever calls `forward`/`predict_noise`."""
        group_idx = task_idx_to_group_idx(task_idx)
        return self.apply_residual(a_t_bc, group_idx, context)
