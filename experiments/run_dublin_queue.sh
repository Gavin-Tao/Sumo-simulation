#!/usr/bin/env bash
# Dublin + 1x3 training queue (2026-06-11).
#
# Schedule (user-specified):
#   Phase 1: exp192 + exp197 + exp189 run IN PARALLEL, wait for ALL three.
#   Phase 2: then 193 -> 191 -> 194 -> 195 -> 196 run ONE AT A TIME
#            (each must finish before the next starts).
#
# trainer per exp:
#   189/191/192/193 = train.py        (B-movement MLP; 191/192/193 masked, 189 is 1x3)
#   194         = trainorico.py       (CoLightOrig GAT, masked)
#   195/196     = train.py            (T lane-token transformer, masked)
#   197         = trainorico.py       (1x3 CoLightOrig, the B-coord arm)
#
# Usage:
#   ./run_dublin_queue.sh                 # GPU 0, WANDB online
#   GPU=1 ./run_dublin_queue.sh
#   WANDB_MODE=offline ./run_dublin_queue.sh
set -uo pipefail
cd "$(dirname "$0")"                                # -> experiments/

export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
GPU="${GPU:-0}"
LOGDIR="logs/dublin_queue"
mkdir -p "$LOGDIR"

# exp -> (trainer, config) ----------------------------------------------------
declare -A TRAINER=(
  [189]=train.py [191]=train.py [192]=train.py [193]=train.py
  [194]=trainorico.py [195]=train.py [196]=train.py [197]=trainorico.py
)
cfg() { ls configs/exp${1}_*.yaml; }

run_one() {  # blocking: run a single exp to completion
  local n="$1" tr c rc
  tr="${TRAINER[$n]}"; c="$(cfg "$n")"
  echo "[$(date '+%F %T')] ▶ START exp${n}  ($tr)  $(basename "$c")"
  python "$tr" --config "$c" --gpu "$GPU" > "$LOGDIR/exp${n}.log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date '+%F %T')] ✓ DONE  exp${n}"
  else
    echo "[$(date '+%F %T')] ✗ FAIL  exp${n} (rc=$rc) — see $LOGDIR/exp${n}.log"
  fi
  return $rc
}

echo "===== queue start | GPU=$GPU WANDB=$WANDB_MODE logs=$LOGDIR/ ====="

# ── Phase 1: 192 + 197 + 189 in parallel, wait for all three ────────────────
echo "[$(date '+%F %T')] ▶ PHASE 1: exp192 ∥ exp197 ∥ exp189"
run_one 192 & P192=$!
run_one 197 & P197=$!
run_one 189 & P189=$!
wait $P192; R192=$?
wait $P197; R197=$?
wait $P189; R189=$?
echo "[$(date '+%F %T')] PHASE 1 done (exp192 rc=$R192, exp197 rc=$R197, exp189 rc=$R189)"

# ── Phase 2: strictly sequential 193 -> 191 -> 194 -> 195 -> 196 ─────────────
echo "[$(date '+%F %T')] ▶ PHASE 2: sequential 193 191 194 195 196"
for n in 193 191 194 195 196; do
  run_one "$n" || echo "[$(date '+%F %T')] (continuing despite exp${n} failure)"
done

echo "===== queue finished | $(date '+%F %T') ====="
