#!/usr/bin/env bash
set -euo pipefail

: "${PI05_ROOT:?source ~/pi05_env.sh first}"
output_root="${1:?usage: $0 /path/to/formal/output}"
checkpoint="$output_root/checkpoints/003000/pretrained_model"
python="$PI05_ROOT/envs/lerobot-py312/bin/python"
server=/home/yudongluo/user/Roco/RoCo_Assembly/task/pi05_server.py
client="$PI05_ROOT/scripts/pi05_protocol_smoke_client.py"
load_log="$PI05_ROOT/logs/pi05_formal_load_checkpoint.log"
smoke_log="$PI05_ROOT/logs/pi05_formal_protocol_smoke.log"
server_log="$PI05_ROOT/logs/pi05_server_formal_protocol.log"
latest="$PI05_ROOT/logs/latest_checkpoint_path.txt"

test -f "$checkpoint/config.json" || { echo "ERROR: missing config: $checkpoint" >&2; exit 1; }
test -f "$checkpoint/model.safetensors" || { echo "ERROR: missing weights: $checkpoint" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=2
export CHECKPOINT="$checkpoint"

"$python" -c 'import os, torch; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; p=PI05Policy.from_pretrained(os.environ["CHECKPOINT"]); p.eval().to("cuda"); print("checkpoint load OK", next(p.parameters()).device, next(p.parameters()).dtype, p.config.chunk_size, p.config.n_action_steps)' \
  >"$load_log" 2>&1

"$python" "$client" "$checkpoint" --server "$server" --server-log "$server_log" \
  >"$smoke_log" 2>&1

latest_tmp="$PI05_ROOT/logs/.latest_checkpoint_path.$$.tmp"
printf '%s\n' "$checkpoint" > "$latest_tmp"
mv "$latest_tmp" "$latest"
printf 'VALIDATION PASSED; latest checkpoint updated: %s\n' "$checkpoint"
