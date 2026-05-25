#!/bin/bash
# Run 6 long evals (num_seconds=21600 → ~21 ambulances) on best.pth checkpoints.
# Output: <ckpt_dir>/eval_summary.json (eval_unified.py default).
# Run sequentially to avoid SUMO port conflicts.
set -e
cd "$(dirname "$0")"

# (agent_type, config, ckpt_glob)
declare -a JOBS=(
  "orico  |configs/evaluation/eval_exp126_NS20bus_1amb_U.yaml|models/exp126_*/*/best.pth"
  "dqn    |configs/evaluation/eval_exp127_NS20bus_1amb_U.yaml|models/exp127_*/*/best.pth"
  "orico  |configs/evaluation/eval_exp134_NS20bus_1amb_U.yaml|models/exp134_*/*/best.pth"
  "dqn    |configs/evaluation/eval_exp136_NS20bus_1amb_U.yaml|models/exp136_*/*/best.pth"
  "orico  |configs/evaluation/eval_exp135_NS20bus_1amb_U.yaml|models/exp135_*/*/best.pth"
  "dqn    |configs/evaluation/eval_exp137_NS20bus_1amb_U.yaml|models/exp137_*/*/best.pth"
)

for j in "${JOBS[@]}"; do
  IFS='|' read -r agent cfg ckpt_glob <<< "$j"
  agent=$(echo "$agent" | xargs)        # trim
  ckpt=$(ls $ckpt_glob | head -1)
  echo ""
  echo "=================================================="
  echo "  $cfg  [$agent]"
  echo "  ckpt: $ckpt"
  echo "=================================================="
  python eval_unified.py \
    --agent "$agent" \
    --config "$cfg" \
    --ckpt "$ckpt" \
    --episodes 1 \
    --warmup 0 \
    --fixed-seed
done

echo ""
echo "All 6 long evals done."
