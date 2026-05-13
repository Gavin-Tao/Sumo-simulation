#!/usr/bin/env bash
# Run exp122 - exp137 in batches of 4 (to bound concurrent GPU/RAM usage).
# Each batch launches up to 4 training jobs in parallel; the next batch
# starts only after all jobs in the current batch finish.
#
# Usage:
#   ./experiments/run_all_exps.sh           # default GPU=0
#   GPU=1 ./experiments/run_all_exps.sh     # use GPU 1
#   GPU=-1 ./experiments/run_all_exps.sh    # CPU
#
# Logs go to ../logs/runs/<exp_name>.log (= repo_root/logs/runs)
# Trained checkpoints go under experiments/models/<exp_name>/<timestamp>/
#
# IMPORTANT: This script is invoked from the experiments/ directory.
# Yaml configs use "../nets/..." paths which are relative to experiments/,
# so we cd into experiments/ before launching python (idempotent if already there).

set -u  # don't exit on errors so one failed job doesn't kill the rest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"  # always run from experiments/  (yamls' "../nets/..." needs this CWD)

GPU="${GPU:-0}"
CFG_DIR="configs"        # relative to experiments/
LOG_DIR="../logs/runs"   # = repo_root/logs/runs
mkdir -p "$LOG_DIR"

run_one() {
    local yaml="$1"
    local name; name=$(basename "$yaml" .yaml)
    # Skip if this exp already has a best_metrics.json saved (from a previous
    # training run, e.g., killed mid-train but best ckpt was saved). The plot
    # script reads the latest timestamp's best_metrics.json automatically.
    # To force re-run: delete the exp's model dir.
    if ls models/"$name"/*/best_metrics.json 1>/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] SKIP   $name  (best_metrics.json already exists)"
        return 0
    fi
    local script
    if [[ "$yaml" == *colightorig* ]]; then
        script="trainorico.py"
    else
        script="train.py"
    fi
    local start; start=$(date +%H:%M:%S)
    echo "[$start] START  $name  ($script, gpu=$GPU)"
    python "$script" --config "$yaml" --gpu "$GPU" > "$LOG_DIR/$name.log" 2>&1
    local rc=$?
    local end; end=$(date +%H:%M:%S)
    if [[ $rc -eq 0 ]]; then
        echo "[$end] DONE   $name"
    else
        echo "[$end] FAIL   $name (rc=$rc)  → $LOG_DIR/$name.log"
    fi
}

# Strategy: DQN first (small, 4-parallel), CoLight after (big, 1-at-a-time).
# RTX 3060 Ti has 8GB VRAM — testing showed 4-mixed-parallel causes CUDA "device
# busy or unavailable" errors (only 1 of 4 survives the GPU memory race).
# DQN models are tiny (~65KB ckpt, ~1GB GPU runtime), 4 in parallel fits.
# CoLight (multi-head GAT) is ~3GB each, even 2-parallel risks OOM.
#
# 7 exps total — DQN only (CoLight exps 122/124/126/128/134/135 to be run later).
# (exp130/exp131 at NS 5% bus also dropped per user request.)
BATCHES=(
    # ─── DQN + baseline ───
    "exp125 exp127 exp129 exp133"                # DQN 5-2-1: NS 100/50/20/10 bus
    "exp136 exp137 exp140 exp141"         # DQN: baseline + weights 5-3-1/5-4-1 (NS 20% + NS 100%)
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
        # Stagger by 3 seconds between launches so CUDA contexts initialize
        # serially, not in parallel. Avoids "device busy or unavailable" race
        # that happens when 4 PyTorch processes call torch.cuda.init() at the
        # same instant.
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
echo "ALL 15 experiments complete in ${elapsed}s ($((elapsed/3600))h$(((elapsed/60)%60))m)"
echo "Logs:   $LOG_DIR/"
echo "Models: ./models/"
echo "================================================================"

# ── Auto-generate sensitivity plots from best_metrics.json ────────────────
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

# ── Chain: run CoLight exps next, then re-plot (CoLight script auto-plots at end) ──
echo ""
echo "================================================================"
echo "DQN phase done. Launching CoLight exps via run_colight_exps.sh ..."
echo "(CoLight script will auto-regenerate plots with both DQN + CoLight data)"
echo "================================================================"
bash "$SCRIPT_DIR/run_colight_exps.sh"
