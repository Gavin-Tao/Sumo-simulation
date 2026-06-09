#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 2-phase ambmax 实验串行 (4 个 CoeffDQN + amb-max reward)
#
# PHASE 1 (优先, 2 个并行):
#   exp167 = CoeffDQN 1x3 5-4-1 ambmax  (跟 exp147 对照, 看 max vs avg 在协同场景)
#   exp168 = CoeffDQN 1x1 5-4-1 ambmax  (无协同 baseline, 看 reward 本身)
#
# PHASE 2 (Phase 1 完成后, 2 个并行):
#   exp169 = CoeffDQN 1x1 5-3-1 ambmax  (1x1 weight 敏感性)
#   exp170 = CoeffDQN 1x1 5-2-1 ambmax  (1x1 weight 敏感性)
#
# 用法:
#   cd experiments && bash run_exp167_170_ambmax.sh
#   GPU=1 STAGGER=15 bash run_exp167_170_ambmax.sh
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")" || { echo "cd 失败"; exit 1; }

PHASE1=(
    "configs/exp167_coeff_1x3_541_avg_waiting_NS20bus_1amb_U_ambmax_tuned.yaml"
    "configs/exp168_coeff_1x1_541_avg_waiting_NSamb_ambmax_tuned.yaml"
)
PHASE2=(
    "configs/exp169_coeff_1x1_531_avg_waiting_NSamb_ambmax_tuned.yaml"
    "configs/exp170_coeff_1x1_521_avg_waiting_NSamb_ambmax_tuned.yaml"
)

GPU="${GPU:-0}"
STAGGER="${STAGGER:-15}"
LOG_DIR="./logs/run_ambmax_$(date +%Y%m%d_%H%M%S)"
SCRIPT="traincoeff.py"
mkdir -p "$LOG_DIR"

ALL_PIDS=()
declare -A NAME_OF_PID
declare -A EXIT_CODES

trap '
    echo
    echo "[$(date)] 收到中断, 杀掉子进程..."
    for pid in "${ALL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  killed PID $pid"
    done
    sleep 2
    for pid in "${ALL_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    exit 130
' INT TERM

run_phase() {
    local phase_name=$1
    shift
    local items=("$@")
    local pids=()

    echo "════════════════════════════════════════════════════════════════════════════"
    echo "[$(date)] ▶▶▶ $phase_name 启动 (${#items[@]} 并行)"
    echo "════════════════════════════════════════════════════════════════════════════"

    for i in "${!items[@]}"; do
        local cfg="${items[$i]}"
        local name=$(basename "$cfg" .yaml)
        local log="$LOG_DIR/${name}.log"

        echo "[$(date)] ▶ LAUNCH  $name"
        echo "  log: $log"

        python3 "$SCRIPT" --config "$cfg" --gpu "$GPU" > "$log" 2>&1 &
        local pid=$!
        pids+=("$pid")
        ALL_PIDS+=("$pid")
        NAME_OF_PID[$pid]=$name
        echo "  PID: $pid"
        echo

        if [ "$i" -lt "$((${#items[@]} - 1))" ]; then
            echo "[$(date)] stagger ${STAGGER}s ..."
            sleep "$STAGGER"
            echo
        fi
    done

    echo "[$(date)] $phase_name 全部启动, 等结束 ..."
    echo "  实时看: tail -f $LOG_DIR/*.log"
    echo

    for pid in "${pids[@]}"; do
        wait "$pid"
        local rc=$?
        local name="${NAME_OF_PID[$pid]}"
        EXIT_CODES[$name]=$rc
        if [ "$rc" -eq 0 ]; then
            echo "[$(date)] ✓ $name 结束 (exit 0)"
        else
            echo "[$(date)] ✗ $name 失败 (exit $rc)"
            echo "  最后 5 行 log:"
            tail -5 "$LOG_DIR/${name}.log" | sed 's/^/    /'
        fi
    done
    echo
    echo "[$(date)] $phase_name 全部完成"
    echo
}

START_TIME=$SECONDS

echo "总览: PHASE 1 (${#PHASE1[@]}) → PHASE 2 (${#PHASE2[@]})  共 $((${#PHASE1[@]} + ${#PHASE2[@]})) runs"
echo "  GPU: $GPU, STAGGER: ${STAGGER}s, LOG_DIR: $LOG_DIR"
echo "  PHASE 1 RAM: 2 × ~2GB ≈ 4GB"
echo "  PHASE 2 RAM: 2 × ~1.5GB ≈ 3GB (1x1 内存少)"
echo "  ⚠ 留意跟其它 run (exp157/159/160 等)共存的 RAM"
echo

run_phase "PHASE 1 (5-4-1 ambmax: exp167 1x3 + exp168 1x1)" "${PHASE1[@]}"
run_phase "PHASE 2 (1x1 ambmax 敏感性: exp169 5-3-1 + exp170 5-2-1)" "${PHASE2[@]}"

TOTAL=$((SECONDS - START_TIME))
echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 全部 4 个 run 完成,总挂钟 ${TOTAL}s (≈ $((TOTAL/60))min)"
echo
echo "PHASE 1:"
for cfg in "${PHASE1[@]}"; do
    name=$(basename "$cfg" .yaml)
    rc=${EXIT_CODES[$name]}
    icon="✓"; [ "$rc" -ne 0 ] && icon="✗"
    echo "  $icon  $name  (exit $rc)"
done
echo
echo "PHASE 2:"
for cfg in "${PHASE2[@]}"; do
    name=$(basename "$cfg" .yaml)
    rc=${EXIT_CODES[$name]}
    icon="✓"; [ "$rc" -ne 0 ] && icon="✗"
    echo "  $icon  $name  (exit $rc)"
done
echo
echo "  log 目录: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════════════════"
