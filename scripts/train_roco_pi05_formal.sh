#!/usr/bin/env bash
set -euo pipefail

: "${PI05_ROOT:?source ~/pi05_env.sh first}"
train_bin="$PI05_ROOT/envs/lerobot-py312/bin/lerobot-train"
dataset_root="$PI05_ROOT/datasets/rocochallenge2026_Industrial_Assembly"
run_stamp="${PI05_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
output_root="${PI05_FORMAL_OUTPUT:-$PI05_ROOT/outputs/roco_pi05_expert_3000_${run_stamp}}"
steps=3000

test -x "$train_bin" || { echo "ERROR: missing $train_bin" >&2; exit 1; }
test -f "$dataset_root/meta/info.json" || { echo "ERROR: dataset missing at $dataset_root" >&2; exit 1; }
test ! -e "$output_root" || { echo "ERROR: output already exists: $output_root" >&2; exit 1; }
mkdir -p "$(dirname "$output_root")" "$PI05_ROOT/logs"
export CUDA_VISIBLE_DEVICES=2
printf '%s\n' "$output_root" > "$PI05_ROOT/logs/current_formal_output.txt"

exec "$train_bin" \
  --dataset.repo_id=rocochallenge2025/rocochallenge2026_Industrial_Assembly \
  --dataset.root="$dataset_root" \
  --dataset.revision=main \
  --rename_map='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_hand":"observation.images.left_wrist_0_rgb","observation.images.right_hand":"observation.images.right_wrist_0_rgb"}' \
  --policy.path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --output_dir="$output_root" \
  --steps="$steps" \
  --batch_size=1 \
  --num_workers=0 \
  --log_freq=5 \
  --save_freq="$steps" \
  --save_checkpoint=true \
  --env_eval_freq=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false
