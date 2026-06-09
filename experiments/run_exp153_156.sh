#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 只跑 exp153 + exp156 (5-4-1 上的 PER / PER+Double 实验)
#
# 测试 PER 能否补回 CoeffDQN 在 5-4-1 上 amb 退化的问题:
#   exp153 = CoeffDQN + PER          (5-4-1)
#   exp156 = CoeffDQN + PER + Double (5-4-1)
#
# 都基于 exp147 (plain CoeffDQN controlled hd=128),所以 use_per=true (+ exp156 use_double=true).
#
# 用法:
#   cd experiments && bash run_exp153_156.sh
#   GPU=1 STAGGER=20 bash run_exp153_156.sh
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")" || { echo "cd 失败"; exit 1; }

CONFIGS=(
    "configs/exp153_coeff_1x3_541_avg_waiting_NS20bus_1amb_U_PER.yaml"
    "configs/exp156_coeff_1x3_541_avg_waiting_NS20bus_1amb_U_PER_Double.yaml"
)

GPU="${GPU:-0}"
STAGGER="${STAGGER:-15}"
LOG_DIR="./logs/run_per_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PIDS=()
declare -A NAME_OF_PID
declare -A EXIT_CODES

trap '
    echo
    echo "[$(date)] 收到中断,杀掉子进程..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  killed PID $pid"
    done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    exit 130
' INT TERM

echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 启动 ${#CONFIGS[@]} 个并行 CoeffDQN+PER 实验"
echo "  GPU: $GPU, STAGGER: ${STAGGER}s"
echo "  LOG_DIR: $LOG_DIR"
echo "  注意: BATCH 2 (exp148-150) 可能还在跑,本脚本会和它共存"
echo "════════════════════════════════════════════════════════════════════════════"
echo

START_TIME=$SECONDS

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    log="$LOG_DIR/${name}.log"

    echo "[$(date)] ▶ LAUNCH  $name"
    echo "  cmd: python3 traincoeff.py --config $cfg --gpu $GPU"
    echo "  log: $log"

    python3 traincoeff.py --config "$cfg" --gpu "$GPU" > "$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    NAME_OF_PID[$pid]=$name
    echo "  PID: $pid"
    echo

    if [ "$i" -lt "$((${#CONFIGS[@]} - 1))" ]; then
        echo "[$(date)] stagger ${STAGGER}s ..."
        sleep "$STAGGER"
        echo
    fi
done

echo "[$(date)] 全部启动,等结束 ..."
echo "  实时看: tail -f $LOG_DIR/*.log"
echo

for pid in "${PIDS[@]}"; do
    wait "$pid"
    rc=$?
    name="${NAME_OF_PID[$pid]}"
    EXIT_CODES[$name]=$rc
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] ✓ $name 结束 (exit 0)"
    else
        echo "[$(date)] ✗ $name 失败 (exit $rc)"
        echo "  最后 5 行 log:"
        tail -5 "$LOG_DIR/${name}.log" | sed 's/^/    /'
    fi
done

TOTAL=$((SECONDS - START_TIME))
echo
echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 全部完成,总挂钟 ${TOTAL}s (≈ $((TOTAL/60))min)"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    rc=${EXIT_CODES[$name]}
    icon="✓"; [ "$rc" -ne 0 ] && icon="✗"
    echo "  $icon  $name  (exit $rc)"
done
echo "  log 目录: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════════════════"
