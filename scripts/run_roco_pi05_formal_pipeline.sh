#!/usr/bin/env bash
set -euo pipefail

source /home/yudongluo/pi05_env.sh
: "${PI05_ROOT:?missing PI05_ROOT}"
output_root="${1:?usage: $0 /path/to/new/output}"
export PI05_FORMAL_OUTPUT="$output_root"
export CUDA_VISIBLE_DEVICES=2

"$PI05_ROOT/scripts/train_roco_pi05_formal.sh"
"$PI05_ROOT/scripts/validate_roco_pi05_formal.sh" "$output_root"
