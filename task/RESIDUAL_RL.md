# Residual RL architecture

The residual stack is split into four independent layers:

1. `residual_policy.py` adapts a frozen base policy to canonical physical
   actions: `[x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]`.
   Built-in adapters support `Pi05LeRobotPolicy`, vision LeRobot DP, and the
   repository's state-only DP. New policies can implement
   `ResidualPolicyAdapter` or use `CallableResidualPolicyAdapter`.
2. `residual_task.py` owns task semantics: error queries, gate conditions,
   success, bottleneck diagnostics, and reward. `SnapInsertionTask` covers the
   task-board snap parts. Tasks and rewards have separate registries so reward
   ablations do not require environment changes.
3. `residual_env.py` owns only the Gym state machine: prefix replay, residual
   history, gripper hold, stuck detection, observation assembly, and episode
   transitions.
4. `residual_td3.py` contains generic TD3 networks, replay buffer, update, and
   checkpoint code. It depends only on observation/action dimensions.

## Add a base policy

Implement `ResidualPolicyAdapter` with:

- `observation_features(observation)`
- `predict_actions(observation, horizon=..., replan=...)`
- `clear_cache()` and `close()`

Return canonical 7-D physical actions. `replan=True` must remove action-chunk
hidden state so every RL transition has execution horizon one.

## Add a task or reward

Implement `ResidualTask` and/or `ResidualReward`, then register a factory with
`register_residual_task()` or `register_residual_reward()`. A task receives the
simulator backend, while a reward receives a `RewardContext`; neither imports
TD3 or a frozen policy implementation.

The runner exposes `--residual-task` and `--residual-reward`. The defaults are
`snap_insertion` and `bounded_snap`, which preserve the existing HDMI setup.

## Compatibility

`ResidualEnvConfig` retains its original gate/reward fields. If no explicit
task is supplied, the environment constructs the legacy-equivalent
`SnapInsertionTask` and `BoundedSnapReward`. Existing HDMI scripts and TD3
checkpoints therefore keep the same 70-D observation when π0.5 is used.
