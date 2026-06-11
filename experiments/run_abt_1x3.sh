#!/usr/bin/env bash
# Run the 1x3 coordination-ladder comparison (exp184-188) IN PARALLEL.
# Each arm uses a different trainer, mapped explicitly. All share: B/T-cqm state,
# NS-20bus-1amb-U scenario, 5-4-1 (table-driven priority-avg-waiting), train_only.sumocfg.
#
#   exp184  B-cqm   no-coord       (L0)   train.py
#   exp185  T-cqm   no-coord       (L0)   train.py        ← T baseline
#   exp186  B-cqm   CoLightOrig    (L1)   trainorico.py
#   exp187  B-cqm   CoeffDQN β     (L2)   traincoeff.py
#   exp188  T-cqm   boundary-token (T-native coord) train.py  ← vs exp185
#
# Usage:
#   ./run_abt_1x3.sh                 # all 5 in PARALLEL, WANDB online, GPU 0
#   WANDB_MODE=offline ./run_abt_1x3.sh
#   GPU=1 ./run_abt_1x3.sh
#   ONLY="185 188" ./run_abt_1x3.sh  # subset (still parallel)
#   PARALLEL=0 ./run_abt_1x3.sh      # fall back to sequential (if GPU mem tight)
#
# Logs: experiments/logs/abt_1x3/exp<NNN>.log
set -uo pipefail
cd "$(dirname "$0")"                               # → experiments/

export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"     # 5 parallel procs → cap CPU threads each
GPU="${GPU:-0}"
PARALLEL="${PARALLEL:-1}"
ONLY="${ONLY:-184 185 186 187 188}"
LOGDIR="logs/abt_1x3"
mkdir -p "$LOGDIR"

declare -A TRAINER=(
  [184]=train.py [185]=train.py [186]=trainorico.py [187]=traincoeff.py [188]=train.py
)

echo "WANDB_MODE=$WANDB_MODE  GPU=$GPU  PARALLEL=$PARALLEL  runs=[$ONLY]  logs→$LOGDIR/"
echo

pids=(); names=(); rc=0
for n in $ONLY; do
  tr="${TRAINER[$n]:-}"
  if [[ -z "$tr" ]]; then echo "[$(date +%H:%M:%S)] ✗ exp${n}: no trainer mapping"; rc=1; continue; fi
  cfg=( configs/exp${n}_1x3_541_*.yaml )
  if [[ ! -f "${cfg[0]}" ]]; then echo "[$(date +%H:%M:%S)] ✗ exp${n}: config not found"; rc=1; continue; fi
  echo "[$(date +%H:%M:%S)] ▶ start exp${n}  ($tr)  $(basename "${cfg[0]}")"
  if [[ "$PARALLEL" == "1" ]]; then
    python "$tr" --config "${cfg[0]}" --gpu "$GPU" > "$LOGDIR/exp${n}.log" 2>&1 &
    pids+=("$!"); names+=("$n")
  else
    python "$tr" --config "${cfg[0]}" --gpu "$GPU" > "$LOGDIR/exp${n}.log" 2>&1 \
      && echo "[$(date +%H:%M:%S)] ✓ exp${n} done" \
      || { echo "[$(date +%H:%M:%S)] ✗ exp${n} FAILED — see $LOGDIR/exp${n}.log"; rc=1; }
  fi
done

if [[ "$PARALLEL" == "1" ]]; then
  echo "[$(date +%H:%M:%S)] all launched (${#pids[@]} procs); waiting…"
  for i in "${!names[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[$(date +%H:%M:%S)] ✓ exp${names[$i]} done"
    else
      echo "[$(date +%H:%M:%S)] ✗ exp${names[$i]} FAILED — see $LOGDIR/exp${names[$i]}.log"; rc=1
    fi
  done
fi

echo
echo "===== All done (rc=$rc). Per-run logs in $LOGDIR/ ====="
exit $rc
