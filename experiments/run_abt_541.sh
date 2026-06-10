#!/usr/bin/env bash
# Run the A/B/T 5-4-1 vanilla-DQN ablation (exp172-182) in three batches:
#   Batch 1: exp172-175  (A / B-cq / T-transformer / PriorityBCA)  ── parallel, wait all
#   Batch 2: exp180-182  (tuned: A-hd256 / B-hd256 / T-d256)       ── after Batch 1
#   Batch 3: exp176-179  (B fields: c / cqm / cqmm / qm)           ── after Batch 2
#
# Usage:
#   ./run_abt_541.sh                 # default: WANDB online (logs to cloud), GPU 0
#   WANDB_MODE=offline ./run_abt_541.sh  # local only, no cloud (sync later with `wandb sync`)
#   GPU=1 ./run_abt_541.sh           # pick GPU
#   PARALLEL=0 ./run_abt_541.sh      # run each config sequentially instead of parallel-in-batch
#
# Logs: experiments/logs/abt_541/exp<NNN>.log   (full per-run output)
set -uo pipefail
cd "$(dirname "$0")"                              # → experiments/

export WANDB_MODE="${WANDB_MODE:-online}"         # online = log live to wandb cloud (project per config)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"    # cap torch CPU threads per proc (4 parallel runs)
GPU="${GPU:-0}"
PARALLEL="${PARALLEL:-1}"
LOGDIR="logs/abt_541"
mkdir -p "$LOGDIR"

echo "WANDB_MODE=$WANDB_MODE  GPU=$GPU  PARALLEL=$PARALLEL  logs→$LOGDIR/"
echo

run_batch () {
  local nums=("$@") pids=() launched=() rc=0 i
  for n in "${nums[@]}"; do
    local cfg=( configs/exp${n}_1x1_541_NS20bus_vanilla_*.yaml )
    if [[ ! -f "${cfg[0]}" ]]; then
      echo "[$(date +%H:%M:%S)] ✗ exp${n}: config not found"; rc=1; continue
    fi
    echo "[$(date +%H:%M:%S)] start exp${n}  →  $(basename "${cfg[0]}")"
    if [[ "$PARALLEL" == "1" ]]; then
      python train.py --config "${cfg[0]}" --gpu "$GPU" > "$LOGDIR/exp${n}.log" 2>&1 &
      pids+=("$!"); launched+=("$n")          # keep pid↔exp aligned even if a config was skipped
    else
      python train.py --config "${cfg[0]}" --gpu "$GPU" > "$LOGDIR/exp${n}.log" 2>&1 \
        && echo "[$(date +%H:%M:%S)] ✓ exp${n} done" \
        || { echo "[$(date +%H:%M:%S)] ✗ exp${n} FAILED — see $LOGDIR/exp${n}.log"; rc=1; }
    fi
  done
  if [[ "$PARALLEL" == "1" ]]; then
    for i in "${!launched[@]}"; do
      if wait "${pids[$i]}"; then
        echo "[$(date +%H:%M:%S)] ✓ exp${launched[$i]} done"
      else
        echo "[$(date +%H:%M:%S)] ✗ exp${launched[$i]} FAILED — see $LOGDIR/exp${launched[$i]}.log"; rc=1
      fi
    done
  fi
  return $rc
}

echo "===== Batch 1: exp172-175 (base A/B/T + PriorityBCA) ====="
run_batch 172 173 174 175
echo
echo "===== Batch 1 complete — starting Batch 2: exp180-182 (tuned A/B/T) ====="
run_batch 180 181 182
echo
echo "===== Batch 2 complete — starting Batch 3: exp176-179 (B field ablation) ====="
run_batch 176 177 178 179
echo
echo "===== All done. Per-run logs in $LOGDIR/ ====="
