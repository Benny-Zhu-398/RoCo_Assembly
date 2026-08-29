"""Grasp-phase residual RL (TD3) on top of the frozen grouped-vision DP.

One job: align the gripper xy with the gear centre while the arm is still
high, within a capped range. At the first real Z-descent the policy PINS
the xy target absolutely (BC then supplies only z), so the close happens on
centre and the descent is vertical. Lifting and closing are BC's; the place
phase is left entirely to BC (run with DP_CL_MODE=off).

Canonical grasp xy = pick_pos.xy + ee_offset.xy (baseline convention).

Reward -- alignment + a clean "moved together" check, nothing else:
  per step   -(d_xy / grasp_res_m) + 2*progress            (+ -1 on IK fail)
  at grasp   + CENTER_BONUS * (1 - tanh(center_err / center_tol))
  after lift  + HELD_BONUS  if |gear_rise - ee_rise| <= MOVE_TOL_M
             + HELD_FAIL   otherwise
             (scored only once ee has risen >= MOVE_MIN_EE_M; else no term)
  if seated  + SEAT_BONUS                                   (bonus only)
  knocked the ungrasped part off its rest pose -> KNOCK_PENALTY, end
  never closed within the step budget           -> NO_GRASP_PENALTY, end

Launched from run_pick_place.py --residual-grasp-train. Requires
DP_GRASP_RES_M > 0 and exactly one ROCO_PART_ORDER part. Reuses --td3-*.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

import param_config as pc
from residual_grip_train import _deepest_mesh_world_xyz  # shared mesh helper

OBS_DIM = 11
ACT_DIM = 2
_OBS_SCALE = np.array(
    [50.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 1.0], dtype=np.float64
)

# Reward = xy alignment + a clean "gripper and gear moved together" check.
# Nothing about close force, lift height, or seating is shaped.
_PROGRESS_W = 3.0           # dense: reward closing the xy gap during approach
_IK_FAIL_PENALTY = -1.0
_CENTER_BONUS = 80.0        # * (1 - tanh(center_err / tol)) at the close
_CENTER_TOL_M = 0.010       # shaping scale (also the eval "centered" threshold)
_HELD_BONUS = 40.0          # gear tracked the gripper up -> grasp held
_HELD_FAIL = -40.0          # gear did NOT track the gripper -> not held
_MOVE_TOL_M = 0.010         # |gear_rise - ee_rise| within this = "together"
# how far the ee must rise before the move-together check is scored is a CLI
# knob: args.residual_grasp_lift_rise_m (default 0.03).
_SEAT_BONUS = 100.0         # BC seated it afterwards (bonus only, optional)
_KNOCK_PENALTY = -50.0
_KNOCK_TOL_M = 0.008
_NO_GRASP_PENALTY = -20.0


def run_grasp_residual_training(
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
    """Entry point invoked by run_pick_place.main() for --residual-grasp-train.

    When ``args.residual_grasp_eval_ckpt`` is set this runs EVAL ONLY: load
    that checkpoint's actor, roll ``--td3-eval-episodes`` greedy episodes,
    and (with ``--record-video``) write one MP4 of the whole run.
    """
    from residual_td3 import TD3, TD3Config, ReplayBuffer

    if len(pc.part_order) != 1:
        raise ValueError(
            "--residual-grasp-train needs exactly one ROCO_PART_ORDER part, got "
            f"{list(pc.part_order)}"
        )
    part_name = pc.part_order[0]

    if float(getattr(policy, "_grasp_res_m", 0.0)) <= 0.0:
        raise RuntimeError(
            "--residual-grasp-train requires DP_GRASP_RES_M > 0 (the grasp "
            "residual window on the grouped-vision DP policy)."
        )
    if not callable(getattr(policy, "set_grasp_residual", None)):
        raise RuntimeError("selected policy has no set_grasp_residual() hook")

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim_path = f"/World/parts/{part_name}"

    cfg = pc.get_part_config(part_name)
    pick_pos = np.asarray(cfg["pick_pos"], dtype=np.float64).reshape(3)
    ee_off = np.asarray(cfg.get("ee_offset", pc.PART_DEFAULTS["ee_offset"]),
                        dtype=np.float64).reshape(3)
    init_height = float(
        cfg["init_height"] if cfg.get("init_height") is not None else pc.INIT_HEIGHT
    )
    grasp_ee_xy = (pick_pos[:2] + ee_off[:2]).copy()
    grasp_ee_z = float(pick_pos[2] + ee_off[2])

    grade_pos = cfg.get("grade_pos")
    if grade_pos is None:
        grade_pos = cfg.get("place_pos")
    grade_pos = None if grade_pos is None else np.asarray(grade_pos, dtype=np.float64).reshape(3)
    grade_tol = float(getattr(pc, "GRADE_POS_TOL_M", 0.01))
    grade_use_aabb = bool(cfg.get("grade_use_aabb", False))

    def _part_xyz():
        return _deepest_mesh_world_xyz(stage, prim_path, use_aabb=grade_use_aabb)

    res_m = float(policy._grasp_res_m)
    max_step = float(getattr(policy, "_grasp_max_step", 0.005))
    warmup_sim_steps = int(getattr(pc, "WARMUP_STEPS", 0))
    max_grasp_steps = int(args.residual_grasp_max_steps)
    lift_steps = int(args.residual_grasp_lift_steps)
    lift_rise_m = float(args.residual_grasp_lift_rise_m)
    place_cap = int(args.residual_grasp_place_cap)
    prefix_cap = int(args.residual_grasp_prefix_steps)
    max_reset_attempts = int(args.residual_grasp_max_reset_attempts)

    rendering_dt = float(my_world.get_rendering_dt())
    if args.policy_control_hz:
        spa = max(1, int(round(1.0 / (float(args.policy_control_hz) * rendering_dt))))
    else:
        spa = 1

    print(
        f"[grasp.train] part={part_name} grasp_ee_xy={grasp_ee_xy.round(4).tolist()} "
        f"grasp_ee_z={grasp_ee_z:.4f} res_m={res_m} max_step={max_step} "
        f"max_grasp_steps={max_grasp_steps} move_tol_m={_MOVE_TOL_M}",
        flush=True,
    )

    # ------------------------------------------------------------------ sim io
    _video_rec = None   # set in the eval-only branch; captures every step
    _video_cam = getattr(args, "record_video_camera", "head") or "head"

    def _advance():
        for k in range(spa):
            my_world.step(render=(k == spa - 1))

    def _roll(grasp_xy=(0.0, 0.0)):
        obs = build_observation()
        policy.set_grasp_residual((float(grasp_xy[0]), float(grasp_xy[1])))
        l_action = policy.act(obs)
        articulation_controller.apply_action(merge_left_with_right_hold(l_action))
        _advance()
        nxt = build_observation()
        if _video_rec is not None:
            _video_rec.write(nxt.rgb.get(_video_cam))
        return nxt

    def _ee_dxy(obs):
        ee = np.asarray(obs.ee_pose_L[0], dtype=np.float64)
        return float(np.linalg.norm(ee[:2] - grasp_ee_xy)), ee

    def _obs_vector(obs, d_xy, ee, prev_res, gear_off_xy):
        bc = np.asarray(getattr(policy, "_last_bc_pos", None)
                        if getattr(policy, "_last_bc_pos", None) is not None
                        else ee, dtype=np.float64)
        raw = np.array([
            ee[0] - grasp_ee_xy[0],
            ee[1] - grasp_ee_xy[1],
            d_xy,
            ee[2] - grasp_ee_z,
            float(obs.L_gripper_position),
            float(prev_res[0]),
            float(prev_res[1]),
            bc[0] - grasp_ee_xy[0],
            bc[1] - grasp_ee_xy[1],
            float(gear_off_xy),
            1.0 if bool(getattr(policy, "_grasp_committed", False)) else 0.0,
        ], dtype=np.float64)
        return (raw * _OBS_SCALE).astype(np.float32)

    def _reset_and_arm():
        """Reset, roll BC (zero residual) until ungrasped + ee xy within
        res_m of the grasp xy. Returns (obs, attempt, diag, gear_rest_xyz)
        or (None, attempts, diag, None)."""
        worst = "never_near"
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
            restart_iteration()
            my_world.play()
            for _ in range(warmup_sim_steps):
                apply_init_joint_targets()
                my_world.step(render=True)

            gear_rest = _part_xyz()
            obs = build_observation()
            grasped_early = False
            for _ in range(prefix_cap):
                obs = _roll()
                if bool(getattr(policy, "_grasped_once", False)):
                    grasped_early = True
                    break
                d_xy, _ = _ee_dxy(obs)
                if d_xy <= res_m:
                    return obs, attempt, "armed", gear_rest
            worst = "grasped_before_window" if grasped_early else "never_near"
            print(f"[grasp.reset] attempt {attempt}/{max_reset_attempts}: {worst}",
                  flush=True)
        return None, max_reset_attempts, worst, None

    # --------------------------------------------------------------- one episode
    total_steps = 0
    cam_warned = False

    def _episode(agent, buffer, rng, *, train, explore_noise):
        nonlocal total_steps, cam_warned
        obs, reset_attempt, diag, gear_rest = _reset_and_arm()
        if obs is None:
            return {"return": 0.0, "steps": 0, "done_reason": f"no_arm:{diag}",
                    "center_err_m": None, "lift_ok": False,
                    "gear_rose_m": 0.0, "ee_rose_m": 0.0, "seated": False,
                    "reset_attempt": reset_attempt, "ik_fail_steps": 0,
                    "mean_critic_loss": None}
        if not cam_warned:
            cam_warned = True
            head = obs.rgb.get("head")
            hm = None if head is None else float(np.asarray(head, np.float64).mean())
            print(f"[grasp.train] head camera {'ABSENT' if head is None else f'mean px {hm:.1f}'}",
                  flush=True)
        gear_rest_xy = (None if gear_rest is None
                        else np.asarray(gear_rest[:2], dtype=np.float64))
        gear_rest_z = None if gear_rest is None else float(gear_rest[2])
        prev_res = np.zeros(ACT_DIM, dtype=np.float64)
        ep_return = 0.0
        critic_losses = []
        ik_fail_count = 0

        pending = None  # (state, action, nstate, dense_reward) for the grasp step
        done_reason = None
        center_err = None

        for step_i in range(1, max_grasp_steps + 1):
            d_xy, ee = _ee_dxy(obs)
            gear_now = _part_xyz()
            gear_off = (0.0 if (gear_now is None or gear_rest_xy is None)
                        else float(np.linalg.norm(gear_now[:2] - gear_rest_xy)))
            state = _obs_vector(obs, d_xy, ee, prev_res, gear_off)

            if train and total_steps < int(args.td3_warmup_steps):
                action = rng.uniform(-1.0, 1.0, size=ACT_DIM)
            else:
                action = agent.select_action(state)
                if train:
                    action = action + rng.normal(0.0, explore_noise, size=ACT_DIM)
            action = np.clip(action, -1.0, 1.0)

            nxt = _roll(action * max_step)
            nd_xy, nee = _ee_dxy(nxt)
            ngear = _part_xyz()
            ngear_off = (0.0 if (ngear is None or gear_rest_xy is None)
                         else float(np.linalg.norm(ngear[:2] - gear_rest_xy)))

            reward = -(nd_xy / res_m) + _PROGRESS_W * (d_xy - nd_xy) / res_m
            if not bool(getattr(L_controller, "ik_ok", True)):
                reward += _IK_FAIL_PENALTY
                ik_fail_count += 1

            nstate = _obs_vector(nxt, nd_xy, nee, action, ngear_off)
            grasped = bool(getattr(policy, "_grasped_once", False))

            if grasped:
                center_err = nd_xy
                ee_z_grasp = float(nee[2])
                gear_z_grasp = None if ngear is None else float(ngear[2])
                pending = (state, action, nstate, reward, ee_z_grasp,
                           gear_z_grasp, ngear_off)
                done_reason = "grasped"
                break
            if ngear_off > _KNOCK_TOL_M:
                reward += _KNOCK_PENALTY
                done_reason = "knocked"
            elif step_i >= max_grasp_steps:
                reward += _NO_GRASP_PENALTY
                done_reason = "no_grasp"

            if train:
                buffer.add(state, action, nstate, reward,
                           done_reason is not None)
                total_steps += 1
                if (total_steps >= int(args.td3_warmup_steps)
                        and len(buffer) >= int(args.td3_batch_size)):
                    critic_losses.append(
                        agent.train(buffer, batch_size=int(args.td3_batch_size))["critic_loss"]
                    )
            ep_return += reward
            prev_res = action
            obs = nxt
            if done_reason is not None:
                break

        lift_ok = False
        seated = False
        gear_rose = ee_rose = 0.0
        if done_reason == "grasped":
            (state, action, nstate, dense_r, ee_z_grasp, gear_z_grasp,
             grasp_gear_off) = pending
            terminal = dense_r + _CENTER_BONUS * (
                1.0 - math.tanh(center_err / _CENTER_TOL_M)
            )
            # --- move-together check: roll BC (zero residual) until the ee
            # has risen a meaningful amount, then ask whether the part rose
            # WITH it. Binary, and independent of how far BC chooses to lift.
            lift_obs = obs
            for _ in range(lift_steps):
                lift_obs = _roll((0.0, 0.0))
                ee_z_now = float(np.asarray(lift_obs.ee_pose_L[0], np.float64)[2])
                if (ee_z_grasp is not None
                        and ee_z_now - ee_z_grasp >= lift_rise_m):
                    break
            g = _part_xyz()
            ee_now_z = float(np.asarray(lift_obs.ee_pose_L[0], dtype=np.float64)[2])
            ee_rose = ee_now_z - ee_z_grasp
            gear_rose = (0.0 if (g is None or gear_z_grasp is None)
                         else float(g[2] - gear_z_grasp))
            if ee_rose >= lift_rise_m:
                lift_ok = abs(gear_rose - ee_rose) <= _MOVE_TOL_M
                terminal += _HELD_BONUS if lift_ok else _HELD_FAIL
            # else: BC never lifted enough this episode -> inconclusive,
            # no bonus or penalty; the centering term still scores it.
            # --- optional seat check: only worth it if the grasp held ---
            if lift_ok and grade_pos is not None and place_cap > 0:
                for _ in range(place_cap):
                    _roll((0.0, 0.0))
                    gg = _part_xyz()
                    if gg is not None and float(np.linalg.norm(gg - grade_pos)) < grade_tol:
                        break
                gg = _part_xyz()
                if gg is not None:
                    seated = float(np.linalg.norm(gg - grade_pos)) < grade_tol
                if seated:
                    terminal += _SEAT_BONUS
            if train:
                buffer.add(state, action, nstate, terminal, True)
                total_steps += 1
                if (total_steps >= int(args.td3_warmup_steps)
                        and len(buffer) >= int(args.td3_batch_size)):
                    critic_losses.append(
                        agent.train(buffer, batch_size=int(args.td3_batch_size))["critic_loss"]
                    )
            ep_return += terminal

        return {
            "return": ep_return,
            "steps": step_i,
            "done_reason": done_reason or "no_grasp",
            "center_err_m": center_err,
            "lift_ok": lift_ok,
            "gear_rose_m": round(gear_rose, 4),
            "ee_rose_m": round(ee_rose, 4),
            "seated": seated,
            "reset_attempt": reset_attempt,
            "ik_fail_steps": ik_fail_count,
            "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else None,
        }

    # ------------------------------------------------------------------- driver
    agent = TD3(observation_dim=OBS_DIM, action_dim=ACT_DIM,
                config=TD3Config(gamma=0.99, tau=0.005, learning_rate=3e-4, policy_freq=2),
                device=args.td3_device, seed=args.residual_seed)
    buffer = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=int(args.td3_buffer_capacity),
                          seed=args.residual_seed)
    rng = np.random.default_rng(args.residual_seed)
    os.makedirs(os.path.dirname(args.residual_grasp_log) or ".", exist_ok=True)
    os.makedirs(args.residual_grasp_checkpoint_dir, exist_ok=True)

    def _run_eval(train_ep, log_f):
        grasped = centered = lifted = seat = 0
        reasons = {}
        errs = []
        for i in range(1, int(args.td3_eval_episodes) + 1):
            r = _episode(agent, buffer, rng, train=False, explore_noise=0.0)
            reasons[r["done_reason"]] = reasons.get(r["done_reason"], 0) + 1
            grasped += int(r["done_reason"] == "grasped")
            lifted += int(r["lift_ok"])
            seat += int(r["seated"])
            if r["center_err_m"] is not None:
                errs.append(r["center_err_m"])
                centered += int(r["center_err_m"] < _CENTER_TOL_M)
            log_f.write(json.dumps({"type": "eval_episode", "train_episode": train_ep,
                                    "eval_episode": i, **r}) + "\n")
            log_f.flush()
        n = int(args.td3_eval_episodes)
        summary = {"type": "eval_summary", "train_episode": train_ep, "episodes": n,
                   "grasp_rate": grasped / n, "center_rate": centered / n,
                   "lift_rate": lifted / n, "seat_rate": seat / n,
                   "mean_center_err_m": float(np.mean(errs)) if errs else None,
                   "done_reasons": reasons, "total_steps": total_steps}
        log_f.write(json.dumps(summary) + "\n")
        log_f.flush()
        ckpt_every = int(getattr(args, "residual_grasp_ckpt_interval", 0)
                         or args.td3_eval_interval)
        if train_ep % ckpt_every == 0 or train_ep >= int(args.td3_train_episodes):
            agent.save(os.path.join(args.residual_grasp_checkpoint_dir,
                                    f"grasp_td3_ep{train_ep:04d}.pt"), metadata=summary)
        print(f"[grasp.eval] ep{train_ep} grasp={grasped}/{n} centered={centered}/{n} "
              f"lift={lifted}/{n} seat={seat}/{n} "
              f"err_mm={None if not errs else round(1000*float(np.mean(errs)),2)} "
              f"reasons={reasons}", flush=True)
        return summary

    # ---------------------------------------------------------------- eval only
    eval_ckpt = getattr(args, "residual_grasp_eval_ckpt", None)
    if eval_ckpt:
        import torch

        ck = torch.load(eval_ckpt, map_location=args.td3_device)
        agent.actor.load_state_dict(ck["actor"])
        agent.actor.eval()
        print(f"[grasp.eval] loaded {eval_ckpt}", flush=True)

        if getattr(args, "record_video", None) and video_recorder_class is not None:
            _video_rec = video_recorder_class(
                args.record_video,
                fps=getattr(args, "record_video_fps", 30),
                camera=_video_cam,
            )
            print(f"[grasp.eval] recording -> {args.record_video} ({_video_cam})",
                  flush=True)

        n = int(args.td3_eval_episodes)
        grasped = centered = lifted = seat = 0
        reasons = {}
        errs = []
        eval_log = os.path.splitext(args.residual_grasp_log)[0] + "_evalonly.jsonl"
        with open(eval_log, "w", encoding="utf-8") as f:
            for i in range(1, n + 1):
                r = _episode(agent, buffer, rng, train=False, explore_noise=0.0)
                reasons[r["done_reason"]] = reasons.get(r["done_reason"], 0) + 1
                grasped += int(r["done_reason"] == "grasped")
                lifted += int(r["lift_ok"])
                seat += int(r["seated"])
                if r["center_err_m"] is not None:
                    errs.append(r["center_err_m"])
                    centered += int(r["center_err_m"] < _CENTER_TOL_M)
                f.write(json.dumps({"type": "eval_episode", "episode": i, **r}) + "\n")
                f.flush()
                print(
                    f"[grasp.eval] ep={i:02d}/{n} done={r['done_reason']} "
                    f"center_mm={None if r['center_err_m'] is None else round(1000*r['center_err_m'],1)} "
                    f"lift={int(r['lift_ok'])}",
                    flush=True,
                )
            summary = {
                "type": "eval_summary", "checkpoint": eval_ckpt, "episodes": n,
                "grasp_rate": grasped / max(1, n), "center_rate": centered / max(1, n),
                "lift_rate": lifted / max(1, n), "seat_rate": seat / max(1, n),
                "mean_center_err_m": float(np.mean(errs)) if errs else None,
                "done_reasons": reasons,
            }
            f.write(json.dumps(summary) + "\n")
        if _video_rec is not None:
            _video_rec.close()
        print(
            f"[grasp.eval] DONE grasp={grasped}/{n} centered={centered}/{n} "
            f"lift={lifted}/{n} seat={seat}/{n} "
            f"mean_center_mm={None if not errs else round(1000*float(np.mean(errs)),2)} "
            f"reasons={reasons} -> {eval_log}",
            flush=True,
        )
        return

    with open(args.residual_grasp_log, "w", encoding="utf-8") as log_f:
        log_f.write(json.dumps({
            "type": "configuration", "algorithm": "TD3", "mode": "grasp_residual",
            "part": part_name, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "grasp_ee_xy": grasp_ee_xy.tolist(), "grasp_ee_z": grasp_ee_z,
            "res_m": res_m, "max_step": max_step, "max_grasp_steps": max_grasp_steps,
            "lift_steps": lift_steps, "place_cap": place_cap,
            "train_episodes": int(args.td3_train_episodes),
            "warmup_steps": int(args.td3_warmup_steps),
            "exploration_noise": float(args.td3_exploration_noise),
            "reward": {"center_bonus": _CENTER_BONUS, "center_tol_m": _CENTER_TOL_M,
                       "held_bonus": _HELD_BONUS, "held_fail": _HELD_FAIL,
                       "move_tol_m": _MOVE_TOL_M, "lift_rise_m": lift_rise_m,
                       "seat_bonus": _SEAT_BONUS, "knock_penalty": _KNOCK_PENALTY,
                       "no_grasp_penalty": _NO_GRASP_PENALTY},
        }) + "\n")
        log_f.flush()

        consec_no_arm = 0
        reasons_tot = {}
        for ep in range(1, int(args.td3_train_episodes) + 1):
            r = _episode(agent, buffer, rng, train=True,
                         explore_noise=float(args.td3_exploration_noise))
            if str(r["done_reason"]).startswith("no_arm:"):
                consec_no_arm += 1
                if consec_no_arm >= 10:
                    raise RuntimeError(
                        "10 consecutive episodes with no grasp window (last: "
                        f"{r['done_reason']}). The frozen DP is not reaching the "
                        "pick pose -- check the head camera and DP_GRASP_RES_M."
                    )
            else:
                consec_no_arm = 0
            reasons_tot[r["done_reason"]] = reasons_tot.get(r["done_reason"], 0) + 1
            log_f.write(json.dumps({"type": "train_episode", "episode": ep,
                                    "total_steps": total_steps, "buffer": len(buffer),
                                    **r}) + "\n")
            log_f.flush()
            print(f"[grasp.train] ep={ep:03d} steps={r['steps']:02d} "
                  f"return={r['return']:+.2f} done={r['done_reason']} "
                  f"center_mm={None if r['center_err_m'] is None else round(1000*r['center_err_m'],1)} "
                  f"gear_rose_mm={round(1000*r['gear_rose_m'],1)} "
                  f"ee_rose_mm={round(1000*r['ee_rose_m'],1)} "
                  f"lift={int(r['lift_ok'])} seat={int(r['seated'])} "
                  f"grasp_rate={reasons_tot.get('grasped',0)/ep:.3f}", flush=True)
            if ep % int(args.td3_eval_interval) == 0:
                _run_eval(ep, log_f)

        log_f.write(json.dumps({"type": "training_summary",
                                "episodes": int(args.td3_train_episodes),
                                "total_steps": total_steps,
                                "done_reasons": reasons_tot}) + "\n")
        log_f.flush()
    print("[grasp.train] done", flush=True)
