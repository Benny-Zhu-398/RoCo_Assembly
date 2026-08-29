"""Residual RL (TD3) on top of the frozen grouped-vision Diffusion Policy.

The Diffusion Policy stays frozen and keeps producing its action chunks.
Near the place target -- once the part is grasped and the endpoint
closed-loop of ``DiffusionVisionGroupedGripPolicy`` has armed (ee within
``policy._cl_trigger`` of ``grade_pos`` in XY) -- a TD3 agent emits a 2-D
xy correction in [-1, 1].  It is scaled to metres by ``policy._cl_max_step``
and injected through ``policy.set_residual()`` (the hook the policy exposes
for exactly this).  The policy's own lock-and-release then fires when the
xy error drops below ``policy._cl_lock``: it freezes the pose and opens the
gripper, the part settles onto the rack post by gravity, and the episode is
graded by the settled mesh position against ``grade_pos``.

Only the TD3 actor/critic learn; the Diffusion Policy weights never change
(it runs frozen in its own inference subprocess).

Launched from ``run_pick_place.py --residual-grip-train`` -- see that flag's
help.  Requires ``DP_CL_MODE=residual`` and exactly one part in
``ROCO_PART_ORDER``.  Reuses the ``--td3-*`` CLI knobs.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

import param_config as pc

# TD3 obs layout (14-D). Position terms are metres; the scale vector lifts
# the ~cm-magnitude ones close to O(1) before TD3's input LayerNorm.
#   0-1  ee xy  - grade_pos xy         2-3  ee dist_xy, ee z - grade_pos z
#   4-5  gripper pos, grasped flag     6-7  previous residual xy
#   8-9  BC target xy - grade_pos xy   10-12 part mesh xyz - grade_pos xyz
#   13   part mesh speed (m / control step)
OBS_DIM = 14
ACT_DIM = 2
_OBS_SCALE = np.array(
    [50.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0,
     50.0, 50.0, 50.0, 50.0, 50.0, 100.0],
    dtype=np.float64,
)

# Reward shaping. The signal is fully dense: every step is scored by the
# ee->target xy error AND the per-step progress toward it, and the locked
# episode gets a smooth terminal in the settled part-position error (no
# pass/fail cliff at the grade tolerance).
_STEP_PENALTY = 0.05          # per RL step, encourages fast alignment
_PROGRESS_W = 2.0            # weight on (prev_dist_xy - dist_xy) / trigger
_SEAT_BONUS = 100.0          # * (1 - tanh(settle_err / grade_tol)) at lock
_NEARMISS_W = 12.0          # at truncation: * max(0, 1 - min_dist_xy/(NEARMISS_FRAC*trigger))
_NEARMISS_FRAC = 0.35      # only pay the near-miss bonus once genuinely close
_FAIL_REWARD = -100.0        # part released from the hand / drifted out
_TRUNCATE_PENALTY = -10.0    # ran out of RL steps without locking
_IK_FAIL_PENALTY = -1.0     # this step's commanded pose had no IK solution
_DRIFT_STREAK = 3           # consecutive far-from-target steps -> drift_out
# "Released" = the part actually left the gripper. Measured as growth in the
# part<->ee separation vs its value at gate entry (invariant to the normal
# vertical placement descent), or the part falling well below the socket.
# A violent BC lurch can desync part<->ee by a few cm for a frame without
# dropping it, so require the condition to hold for _RELEASE_STREAK steps.
_RELEASE_SEP_M = 0.05
_RELEASE_FLOOR_M = 0.05
_RELEASE_STREAK = 3
_DRIFT_MULT = 1.5             # ee left 1.5 * trigger from place -> give up


def _deepest_mesh_world_xyz(stage, prim_path, use_aabb=False):
    """World position of the deepest Mesh under ``prim_path``.

    Mirrors ``run_pick_place._grade_task``: the "meshT" measure (mesh
    local-to-world translation) by default, or the world-axis-aligned AABB
    midpoint when ``use_aabb`` is set -- required for parts whose
    ``grade_pos`` was baselined as the AABB midpoint (axis-symmetric parts
    like the batteries, cfg["grade_use_aabb"] == True). Using the wrong
    measure puts the reward's target metres away from where the harness
    grades.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    deepest, depth = None, -1
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() != "Mesh":
            continue
        d = p.GetPath().pathString.count("/")
        if d > depth:
            depth, deepest = d, p
    if deepest is None:
        return None
    if use_aabb:
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeWorldBound(deepest)
        mid = bbox.ComputeAlignedRange().GetMidpoint()
        return np.array([float(mid[0]), float(mid[1]), float(mid[2])], dtype=np.float64)
    m = UsdGeom.XformCache().GetLocalToWorldTransform(deepest)
    t = m.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def run_grip_residual_training(
    *,
    policy,
    my_world,
    articulation_controller,
    build_observation,
    merge_left_with_right_hold,
    apply_init_joint_targets,
    restart_iteration,
    clear_snap_state,
    L_controller,
    R_controller,
    args,
    video_recorder_class=None,
):
    """Entry point invoked by ``run_pick_place.main()`` for --residual-grip-train.

    When ``args.residual_grip_eval_ckpt`` is set this runs EVAL ONLY: load
    that checkpoint's actor, roll ``--td3-eval-episodes`` greedy episodes,
    and (if ``--record-video`` is given) write one MP4 of the whole run.
    """
    from residual_td3 import TD3, TD3Config, ReplayBuffer

    if len(pc.part_order) != 1:
        raise ValueError(
            "--residual-grip-train needs exactly one ROCO_PART_ORDER part, got "
            f"{list(pc.part_order)}"
        )
    part_name = pc.part_order[0]

    if getattr(policy, "_cl_mode", None) != "residual":
        raise RuntimeError(
            "--residual-grip-train requires the grouped-vision DP policy with "
            "DP_CL_MODE=residual (got cl_mode="
            f"{getattr(policy, '_cl_mode', None)!r}). Set DP_CL_MODE=residual."
        )
    if not callable(getattr(policy, "set_residual", None)):
        raise RuntimeError("selected policy has no set_residual() hook")

    stage = omni.usd.get_context().get_stage()
    prim_path = f"/World/parts/{part_name}"

    cfg = pc.get_part_config(part_name)
    grade_pos = cfg.get("grade_pos")
    if grade_pos is None:
        grade_pos = cfg.get("place_pos")
    if grade_pos is None:
        raise ValueError(f"part {part_name!r} has no grade_pos / place_pos to grade against")
    grade_pos = np.asarray(grade_pos, dtype=np.float64).reshape(3)
    grade_tol = float(getattr(pc, "GRADE_POS_TOL_M", 0.01))
    # Axis-symmetric parts (batteries) baseline grade_pos as the AABB
    # midpoint; match that measure so the reward target and the harness
    # grade point at the same place.
    grade_use_aabb = bool(cfg.get("grade_use_aabb", False))

    def _part_xyz():
        return _deepest_mesh_world_xyz(stage, prim_path, use_aabb=grade_use_aabb)

    cl_max_step = float(getattr(policy, "_cl_max_step", 0.015))
    trigger = float(getattr(policy, "_cl_trigger", 0.02))

    rendering_dt = float(my_world.get_rendering_dt())
    if args.policy_control_hz:
        steps_per_action = max(
            1, int(round(1.0 / (float(args.policy_control_hz) * rendering_dt)))
        )
    else:
        steps_per_action = 1

    warmup_sim_steps = int(getattr(pc, "WARMUP_STEPS", 0))
    max_rl_steps = int(args.residual_env_max_steps)
    settle_steps = int(args.residual_grip_settle_steps)
    prefix_cap = int(args.residual_grip_prefix_steps)
    max_reset_attempts = int(args.residual_grip_max_reset_attempts)

    print(
        f"[grip.train] part={part_name} grade_pos={grade_pos.round(4).tolist()} "
        f"grade_tol_m={grade_tol} cl_trigger_m={trigger} cl_max_step_m={cl_max_step} "
        f"steps_per_action={steps_per_action} max_rl_steps={max_rl_steps} "
        f"settle_steps={settle_steps}",
        flush=True,
    )

    # ------------------------------------------------------------------ sim io
    _video_rec = None   # set in the eval-only branch; captures every step
    _video_cam = getattr(args, "record_video_camera", "head") or "head"

    def _advance():
        for k in range(steps_per_action):
            my_world.step(render=(k == steps_per_action - 1))

    def _roll_frozen(residual_xy=(0.0, 0.0)):
        obs = build_observation()
        policy.set_residual((float(residual_xy[0]), float(residual_xy[1]), 0.0))
        l_action = policy.act(obs)
        articulation_controller.apply_action(merge_left_with_right_hold(l_action))
        _advance()
        nxt = build_observation()
        if _video_rec is not None:
            _video_rec.write(nxt.rgb.get(_video_cam))
        return nxt

    def _ee_dist_xy(obs):
        ee = np.asarray(obs.ee_pose_L[0], dtype=np.float64)
        return float(np.linalg.norm(ee[:2] - grade_pos[:2])), ee

    def _obs_vector(obs, dist_xy, ee, prev_res, gear_xyz, gear_speed):
        bc = np.asarray(getattr(policy, "_last_bc_pos", None)
                        if getattr(policy, "_last_bc_pos", None) is not None
                        else ee, dtype=np.float64)
        if gear_xyz is None:
            gear_err = np.zeros(3, dtype=np.float64)
        else:
            gear_err = np.asarray(gear_xyz, dtype=np.float64) - grade_pos
        raw = np.array(
            [
                ee[0] - grade_pos[0],
                ee[1] - grade_pos[1],
                dist_xy,
                ee[2] - grade_pos[2],
                float(obs.L_gripper_position),
                1.0 if getattr(policy, "_grasped_once", False) else 0.0,
                float(prev_res[0]),
                float(prev_res[1]),
                bc[0] - grade_pos[0],
                bc[1] - grade_pos[1],
                gear_err[0],
                gear_err[1],
                gear_err[2],
                float(gear_speed),
            ],
            dtype=np.float64,
        )
        return (raw * _OBS_SCALE).astype(np.float32)

    def _reset_and_arm():
        """Full sim reset, then roll the frozen DP (zero residual) until the
        endpoint loop is armed.

        Returns ``(armed_obs, attempt, diag)``. ``armed_obs`` is None when no
        attempt produced a trainable insertion window -- ``diag`` then says why
        ("never_grasped" / "grasped_no_window:<min mm>" / "dp_self_solved").
        The residual only has authority in the near-place window, so a frozen
        DP that fails to pick or fails to bring the part close is not something
        the RL can fix -- the episode is skipped, not crashed.
        """
        worst_diag = "never_grasped"
        best_min_dist = float("inf")

        def _grasp_commit_err():
            hx = getattr(policy, "_grasp_hold_xy", None)
            px = getattr(policy, "_pick_ee_xy", None)
            if hx is None or px is None:
                return None
            return float(np.linalg.norm(np.asarray(hx) - np.asarray(px)))

        for attempt in range(1, max_reset_attempts + 1):
            clear_snap_state()
            try:
                my_world.stop()
            except Exception:
                pass
            my_world.reset()
            apply_init_joint_targets()
            L_controller.reset()
            R_controller.reset()
            restart_iteration()  # -> _start_next_part() -> policy.reset(obs, target)
            my_world.play()
            for _ in range(warmup_sim_steps):
                apply_init_joint_targets()
                my_world.step(render=True)

            obs = build_observation()
            ever_grasped = False
            min_dist = float("inf")
            dp_self_solved = False
            for _ in range(prefix_cap):
                obs = _roll_frozen()
                if getattr(policy, "_place_locked", False):
                    dp_self_solved = True
                    break  # DP solved it alone; nothing for RL to correct here
                grasped = bool(getattr(policy, "_grasped_once", False))
                ever_grasped = ever_grasped or grasped
                dist_xy, _ = _ee_dist_xy(obs)
                if grasped:
                    min_dist = min(min_dist, dist_xy)
                if grasped and dist_xy <= trigger:
                    return obs, attempt, "armed", _grasp_commit_err()
            if dp_self_solved:
                worst_diag = "dp_self_solved"
            elif ever_grasped:
                worst_diag = f"grasped_no_window:{min_dist * 1000:.0f}mm"
                best_min_dist = min(best_min_dist, min_dist)
            print(
                f"[grip.reset] attempt {attempt}/{max_reset_attempts}: "
                f"grasped={ever_grasped} min_dist_xy="
                f"{'--' if min_dist == float('inf') else f'{min_dist*1000:.1f}mm'} "
                f"self_solved={dp_self_solved}",
                flush=True,
            )
        return None, max_reset_attempts, worst_diag, _grasp_commit_err()

    # --------------------------------------------------------------- one episode
    total_steps = 0
    cam_warned = False

    def _episode(agent, buffer, rng, *, train, explore_noise):
        nonlocal total_steps, cam_warned
        obs, reset_attempt, arm_diag, grasp_err = _reset_and_arm()
        if obs is None:
            # Frozen DP did not present a trainable insertion window. No
            # transitions to store; log it and move on so a flaky pick just
            # costs wall-clock instead of crashing the run.
            return {
                "return": 0.0,
                "steps": 0,
                "done_reason": f"no_arm:{arm_diag}",
                "settle_err_m": None,
                "grasp_commit_err_m": grasp_err,
                "reset_attempt": reset_attempt,
                "ik_fail_steps": 0,
                "mean_critic_loss": None,
            }
        if not cam_warned:
            cam_warned = True
            head = obs.rgb.get("head")
            head_mean = (
                None if head is None
                else float(np.asarray(head, dtype=np.float64).mean())
            )
            if head is None or (head_mean is not None and head_mean < 1.0):
                print(
                    "[grip.train] WARNING: head camera frame is "
                    f"{'absent' if head is None else f'near-black (mean px {head_mean:.2f})'}"
                    " -- the frozen vision DP is effectively blind, so BC will "
                    "land far off and the residual cannot recover it. Fix camera "
                    "output before training.",
                    flush=True,
                )
            else:
                print(
                    f"[grip.train] head camera OK (mean px {head_mean:.1f})",
                    flush=True,
                )
        ee_ref = np.asarray(obs.ee_pose_L[0], dtype=np.float64)
        gear_ref = _part_xyz()
        grip_sep_ref = (
            None if gear_ref is None
            else float(np.linalg.norm(gear_ref - ee_ref))
        )
        prev_res = np.zeros(ACT_DIM, dtype=np.float64)
        gear_prev = None if gear_ref is None else np.asarray(gear_ref, dtype=np.float64)
        ep_return = 0.0
        critic_losses = []
        drift_streak = 0
        release_streak = 0
        ik_fail_count = 0
        ep_min_dist = float("inf")

        for rl_step in range(1, max_rl_steps + 1):
            dist_xy, ee = _ee_dist_xy(obs)
            gear_now = _part_xyz()
            gear_speed = (
                0.0
                if (gear_now is None or gear_prev is None)
                else float(np.linalg.norm(gear_now - gear_prev))
            )
            state = _obs_vector(obs, dist_xy, ee, prev_res, gear_now, gear_speed)

            warmup_random = train and total_steps < int(args.td3_warmup_steps)
            if warmup_random:
                # Standard TD3 warm-up: uniform over the action box, decoupled
                # from the post-warm-up actor exploration noise.
                action = rng.uniform(-1.0, 1.0, size=ACT_DIM)
            else:
                action = agent.select_action(state)
                if train:
                    action = action + rng.normal(0.0, explore_noise, size=ACT_DIM)
            action = np.clip(action, -1.0, 1.0)

            nxt = _roll_frozen(action * cl_max_step)
            ndist_xy, nee = _ee_dist_xy(nxt)
            locked = bool(getattr(policy, "_place_locked", False))
            ep_min_dist = min(ep_min_dist, ndist_xy)
            # BC re-pulls hard to its own landing every action chunk, so a
            # single-step excursion is normal. Only call it drift-out if the
            # ee stays far from the target for several consecutive steps.
            if not locked and ndist_xy > _DRIFT_MULT * trigger:
                drift_streak += 1
            else:
                drift_streak = 0

            gear_xyz = _part_xyz()
            nspeed = (
                0.0
                if (gear_xyz is None or gear_now is None)
                else float(np.linalg.norm(np.asarray(gear_xyz) - gear_now))
            )

            # Dense: absolute xy error + per-step progress toward the target.
            progress = (dist_xy - ndist_xy) / trigger
            reward = (
                -(ndist_xy / trigger)
                + _PROGRESS_W * progress
                - _STEP_PENALTY
            )
            # IK failure: L.forward re-issued the last joint targets, so the
            # arm did not move to the commanded anchor. Small penalty so the
            # agent learns to keep the anchor reachable (the anchor clamp in
            # the policy keeps it bounded regardless).
            ik_failed = not bool(getattr(L_controller, "ik_ok", True))
            if ik_failed:
                reward += _IK_FAIL_PENALTY
                ik_fail_count += 1
            terminated = False
            truncated = False
            done_reason = None
            settle_err = None

            release_now = False
            if not locked and gear_xyz is not None:
                sep = (
                    None if grip_sep_ref is None
                    else float(np.linalg.norm(gear_xyz - nee))
                )
                release_now = bool(
                    (sep is not None and sep > grip_sep_ref + _RELEASE_SEP_M)
                    or gear_xyz[2] < grade_pos[2] - _RELEASE_FLOOR_M
                )
            release_streak = release_streak + 1 if release_now else 0

            if not locked and release_streak >= _RELEASE_STREAK:
                reward += _FAIL_REWARD
                terminated = True
                done_reason = "released"
            elif not locked and drift_streak >= _DRIFT_STREAK:
                reward += _FAIL_REWARD
                terminated = True
                done_reason = "drift_out"
            elif locked:
                # Policy has frozen its pose and opened the gripper. Let the
                # part fall onto the post and settle, then grade.
                for _ in range(settle_steps):
                    _roll_frozen((0.0, 0.0))
                settled = _part_xyz()
                settle_err = (
                    math.inf
                    if settled is None
                    else float(np.linalg.norm(settled - grade_pos))
                )
                seated = settle_err < grade_tol
                # Smooth terminal in the settled part-position error: full
                # bonus at err=0, ~0.24*bonus at err=grade_tol, ~0 beyond 3x.
                # No discontinuity at the pass/fail boundary.
                reward += _SEAT_BONUS * (
                    1.0 - math.tanh(min(settle_err, 10.0 * grade_tol) / grade_tol)
                )
                terminated = True
                done_reason = "seated" if seated else "settle_miss"
            elif rl_step >= max_rl_steps:
                # Shaped by how close it ever got: a run that repeatedly
                # touched ~1-2 mm but never caught a lock still beats one
                # that stayed 15 mm out.
                reward += _TRUNCATE_PENALTY
                reward += _NEARMISS_W * max(
                    0.0, 1.0 - ep_min_dist / (_NEARMISS_FRAC * trigger)
                )
                truncated = True
                done_reason = "max_steps"

            done = terminated or truncated
            nstate = _obs_vector(nxt, ndist_xy, nee, action, gear_xyz, nspeed)

            if train:
                buffer.add(state, action, nstate, reward, done)
                total_steps += 1
                if (
                    not warmup_random
                    and total_steps >= int(args.td3_warmup_steps)
                    and len(buffer) >= int(args.td3_batch_size)
                ):
                    losses = agent.train(buffer, batch_size=int(args.td3_batch_size))
                    critic_losses.append(losses["critic_loss"])

            ep_return += reward
            prev_res = action
            obs = nxt
            gear_prev = None if gear_xyz is None else np.asarray(gear_xyz, dtype=np.float64)
            if done:
                return {
                    "return": ep_return,
                    "steps": rl_step,
                    "done_reason": done_reason,
                    "settle_err_m": settle_err,
                    "grasp_commit_err_m": grasp_err,
                    "reset_attempt": reset_attempt,
                    "ik_fail_steps": ik_fail_count,
                    "mean_critic_loss": (
                        float(np.mean(critic_losses)) if critic_losses else None
                    ),
                }

        # max_rl_steps is handled inside the loop; this is a safety net.
        return {
            "return": ep_return,
            "steps": max_rl_steps,
            "done_reason": "max_steps",
            "settle_err_m": None,
            "grasp_commit_err_m": grasp_err,
            "reset_attempt": reset_attempt,
            "ik_fail_steps": ik_fail_count,
            "mean_critic_loss": (
                float(np.mean(critic_losses)) if critic_losses else None
            ),
        }

    # ------------------------------------------------------------------- driver
    agent = TD3(
        observation_dim=OBS_DIM,
        action_dim=ACT_DIM,
        config=TD3Config(gamma=0.99, tau=0.005, learning_rate=3e-4, policy_freq=2),
        device=args.td3_device,
        seed=args.residual_seed,
    )
    buffer = ReplayBuffer(
        OBS_DIM, ACT_DIM, capacity=int(args.td3_buffer_capacity), seed=args.residual_seed
    )
    rng = np.random.default_rng(args.residual_seed)
    os.makedirs(os.path.dirname(args.residual_grip_log) or ".", exist_ok=True)
    os.makedirs(args.residual_grip_checkpoint_dir, exist_ok=True)

    def _run_eval(train_ep, log_f):
        seated = 0
        reasons = {}
        returns = []
        errs = []
        for i in range(1, int(args.td3_eval_episodes) + 1):
            res = _episode(agent, buffer, rng, train=False, explore_noise=0.0)
            reasons[res["done_reason"]] = reasons.get(res["done_reason"], 0) + 1
            seated += int(res["done_reason"] == "seated")
            returns.append(res["return"])
            if res["settle_err_m"] is not None and math.isfinite(res["settle_err_m"]):
                errs.append(res["settle_err_m"])
            log_f.write(
                json.dumps({"type": "eval_episode", "train_episode": train_ep,
                            "eval_episode": i, **res})
                + "\n"
            )
            log_f.flush()
        summary = {
            "type": "eval_summary",
            "train_episode": train_ep,
            "episodes": int(args.td3_eval_episodes),
            "seated": seated,
            "seat_rate": seated / max(1, int(args.td3_eval_episodes)),
            "done_reasons": reasons,
            "mean_return": float(np.mean(returns)) if returns else None,
            "mean_settle_err_m": float(np.mean(errs)) if errs else None,
            "total_steps": total_steps,
        }
        log_f.write(json.dumps(summary) + "\n")
        log_f.flush()
        ckpt = os.path.join(
            args.residual_grip_checkpoint_dir, f"grip_td3_ep{train_ep:04d}.pt"
        )
        agent.save(ckpt, metadata=summary)
        print(
            f"[grip.eval] train_ep={train_ep} "
            f"seated={seated}/{int(args.td3_eval_episodes)} "
            f"rate={summary['seat_rate']:.3f} reasons={reasons} "
            f"mean_settle_err_mm="
            f"{None if not errs else round(1000.0 * float(np.mean(errs)), 2)}",
            flush=True,
        )
        return summary

    # ---------------------------------------------------------------- eval only
    eval_ckpt = getattr(args, "residual_grip_eval_ckpt", None)
    if eval_ckpt:
        import torch

        ck = torch.load(eval_ckpt, map_location=args.td3_device)
        agent.actor.load_state_dict(ck["actor"])
        agent.actor.eval()
        print(f"[grip.eval] loaded {eval_ckpt}", flush=True)

        if getattr(args, "record_video", None) and video_recorder_class is not None:
            _video_rec = video_recorder_class(
                args.record_video,
                fps=getattr(args, "record_video_fps", 30),
                camera=_video_cam,
            )
            print(f"[grip.eval] recording -> {args.record_video} ({_video_cam})",
                  flush=True)

        n = int(args.td3_eval_episodes)
        seated = 0
        reasons = {}
        returns = []
        errs = []
        eval_log = os.path.splitext(args.residual_grip_log)[0] + "_evalonly.jsonl"
        with open(eval_log, "w", encoding="utf-8") as f:
            for i in range(1, n + 1):
                res = _episode(agent, buffer, rng, train=False, explore_noise=0.0)
                reasons[res["done_reason"]] = reasons.get(res["done_reason"], 0) + 1
                seated += int(res["done_reason"] == "seated")
                returns.append(res["return"])
                if res["settle_err_m"] is not None and math.isfinite(res["settle_err_m"]):
                    errs.append(res["settle_err_m"])
                f.write(json.dumps({"type": "eval_episode", "episode": i, **res}) + "\n")
                f.flush()
                print(
                    f"[grip.eval] ep={i:02d}/{n} return={res['return']:+.1f} "
                    f"done={res['done_reason']} "
                    f"settle_mm={None if res['settle_err_m'] is None else round(1000*res['settle_err_m'],1)}",
                    flush=True,
                )
            summary = {
                "type": "eval_summary", "checkpoint": eval_ckpt, "episodes": n,
                "seated": seated, "seat_rate": seated / max(1, n),
                "done_reasons": reasons,
                "mean_return": float(np.mean(returns)) if returns else None,
                "mean_settle_err_m": float(np.mean(errs)) if errs else None,
            }
            f.write(json.dumps(summary) + "\n")
        if _video_rec is not None:
            _video_rec.close()
        print(
            f"[grip.eval] DONE seated={seated}/{n} rate={summary['seat_rate']:.3f} "
            f"reasons={reasons} "
            f"mean_settle_mm={None if not errs else round(1000*float(np.mean(errs)),2)} "
            f"-> {eval_log}",
            flush=True,
        )
        return

    with open(args.residual_grip_log, "w", encoding="utf-8") as log_f:
        log_f.write(
            json.dumps(
                {
                    "type": "configuration",
                    "algorithm": "TD3",
                    "mode": "grip_residual",
                    "part": part_name,
                    "obs_dim": OBS_DIM,
                    "act_dim": ACT_DIM,
                    "cl_max_step_m": cl_max_step,
                    "cl_trigger_m": trigger,
                    "grade_tol_m": grade_tol,
                    "grade_pos": grade_pos.tolist(),
                    "train_episodes": int(args.td3_train_episodes),
                    "warmup_steps": int(args.td3_warmup_steps),
                    "warmup_distribution": "uniform",
                    "exploration_noise": float(args.td3_exploration_noise),
                    "batch_size": int(args.td3_batch_size),
                    "buffer_capacity": int(args.td3_buffer_capacity),
                    "eval_interval": int(args.td3_eval_interval),
                    "eval_episodes": int(args.td3_eval_episodes),
                    "settle_steps": settle_steps,
                    "max_rl_steps": max_rl_steps,
                    "steps_per_action": steps_per_action,
                    "reward": {
                        "dense": "-(dist_xy/trigger) + "
                                 f"{_PROGRESS_W}*progress - {_STEP_PENALTY}",
                        "terminal_lock": f"{_SEAT_BONUS}*(1 - tanh(settle_err/grade_tol))",
                        "fail": _FAIL_REWARD,
                        "truncate": _TRUNCATE_PENALTY,
                    },
                    "device": args.td3_device,
                    "seed": int(args.residual_seed),
                }
            )
            + "\n"
        )
        log_f.flush()

        reasons_tot = {}
        consec_no_arm = 0
        for ep in range(1, int(args.td3_train_episodes) + 1):
            res = _episode(
                agent,
                buffer,
                rng,
                train=True,
                explore_noise=float(args.td3_exploration_noise),
            )
            if str(res["done_reason"]).startswith("no_arm:"):
                consec_no_arm += 1
                if consec_no_arm >= 10:
                    raise RuntimeError(
                        "10 consecutive episodes with no insertion window "
                        f"(last: {res['done_reason']}). The frozen DP is not "
                        "picking/placing gear_60teeth. Check the head camera "
                        "frame, DP_GRIP_BIAS, and DP_CL_TRIGGER_M before "
                        "retrying -- the residual cannot fix the pick."
                    )
            else:
                consec_no_arm = 0
            reasons_tot[res["done_reason"]] = reasons_tot.get(res["done_reason"], 0) + 1
            log_f.write(
                json.dumps(
                    {
                        "type": "train_episode",
                        "episode": ep,
                        "total_steps": total_steps,
                        "buffer": len(buffer),
                        **res,
                    }
                )
                + "\n"
            )
            log_f.flush()
            _gce = res.get("grasp_commit_err_m")
            print(
                f"[grip.train] ep={ep:03d} steps={res['steps']:02d} "
                f"return={res['return']:+.2f} done={res['done_reason']} "
                f"grasp_err_mm={'--' if _gce is None else round(1000*_gce, 1)} "
                f"buffer={len(buffer)} "
                f"seat_rate={reasons_tot.get('seated', 0) / ep:.3f}",
                flush=True,
            )
            if ep % int(args.td3_eval_interval) == 0:
                _run_eval(ep, log_f)

        log_f.write(
            json.dumps(
                {
                    "type": "training_summary",
                    "episodes": int(args.td3_train_episodes),
                    "total_steps": total_steps,
                    "done_reasons": reasons_tot,
                }
            )
            + "\n"
        )
        log_f.flush()

    print("[grip.train] done", flush=True)
