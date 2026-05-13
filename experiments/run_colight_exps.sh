#!/usr/bin/env bash
# Run CoLight (multi-head GAT) exps only — 6 exps total.
# Pair with run_all_exps.sh which runs the DQN side.
#
# CoLight model is bigger than DQN (~3GB GPU each). On 8GB VRAM (3060 Ti),
# 2 in parallel is the safe limit, with 3-sec stagger between launches to
# avoid CUDA context-init race condition.
#
# IMPORTANT: invoke from experiments/ directory (yamls use ../nets/... paths).
#
# Usage:
#   ./experiments/run_colight_exps.sh           # default GPU=0
#   GPU=1 ./experiments/run_colight_exps.sh     # use GPU 1
#
# Logs:    ../logs/runs/<exp_name>.log
# Models:  experiments/models/<exp_name>/<timestamp>/
# Plots:   experiments/figures/  (auto-generated after all exps finish)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"  # = experiments/

GPU="${GPU:-0}"
CFG_DIR="configs"
LOG_DIR="../logs/runs"
mkdir -p "$LOG_DIR"

run_one() {
    local yaml="$1"
    local name; name=$(basename "$yaml" .yaml)
    # Skip if best_metrics.json already saved (from previous training)
    if ls models/"$name"/*/best_metrics.json 1>/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] SKIP   $name  (best_metrics.json already exists)"
        return 0
    fi
    local start; start=$(date +%H:%M:%S)
    echo "[$start] START  $name  (trainorico.py, gpu=$GPU)"
    python trainorico.py --config "$yaml" --gpu "$GPU" > "$LOG_DIR/$name.log" 2>&1
    local rc=$?
    local end; end=$(date +%H:%M:%S)
    if [[ $rc -eq 0 ]]; then
        echo "[$end] DONE   $name"
    else
        echo "[$end] FAIL   $name (rc=$rc)  → $LOG_DIR/$name.log"
    fi
}

# 6 CoLight exps, 2 at a time (3 batches). exp130 NS 5% bus dropped per user.
BATCHES=(
    "exp122 exp124"     # CoLight: NS 100% + 50% bus
    "exp126 exp128"     # CoLight: NS 20% + 10% bus
    "exp134 exp135"     # CoLight: weight 5-3-1 + 5-4-1 (both at NS 20% bus)
    "exp138 exp139"     # CoLight: weight 5-3-1 + 5-4-1 (both at NS 100% bus)
)

T0=$SECONDS
for i in "${!BATCHES[@]}"; do
    batch="${BATCHES[$i]}"
    echo ""
    echo "================================================================"
    echo "Batch $((i+1)) / ${#BATCHES[@]}:  $batch"
    echo "================================================================"
    pids=()
    first=1
    for prefix in $batch; do
        yaml=$(ls "$CFG_DIR"/"$prefix"_*.yaml 2>/dev/null | head -1)
        if [[ -z "$yaml" ]]; then
            echo "  ⚠️  No yaml found matching $prefix in $CFG_DIR/"
            continue
        fi
        # Stagger by 3 seconds between launches to avoid CUDA context-init race
        if [[ $first -eq 0 ]]; then
            sleep 3
        fi
        first=0
        run_one "$yaml" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
done

elapsed=$(( SECONDS - T0 ))
echo ""
echo "================================================================"
echo "ALL CoLight exps complete in ${elapsed}s ($((elapsed/3600))h$(((elapsed/60)%60))m)"
echo "Logs:   $LOG_DIR/"
echo "Models: experiments/models/"
echo "================================================================"

# ── Auto-generate sensitivity plots (uses ALL best_metrics.json: DQN + CoLight) ──
echo ""
echo "================================================================"
echo "Generating sensitivity plots from best_metrics.json ..."
echo "================================================================"
plot_log="$LOG_DIR/plot_best_metrics.log"
python plot_best_metrics.py > "$plot_log" 2>&1
plot_rc=$?
if [[ $plot_rc -eq 0 ]]; then
    cat "$plot_log"
    echo ""
    echo "✓ Plots saved to experiments/figures/"
    ls -la figures/ 2>/dev/null
else
    echo "✗ Plotting failed (rc=$plot_rc) — see $plot_log"
    tail -30 "$plot_log"
fi
echo "================================================================"
