#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 后续 run: exp161/162/164/165 (5-2-1 + 5-3-1 ablation, 4 个并行)
#
#   exp161 = GATCoeffDQN   5-2-1   (traincoeff_gat.py)
#   exp162 = GATCoeffDQN   5-3-1   (traincoeff_gat.py)
#   exp164 = FixedCoeffDQN 5-2-1   (traincoeff_fixed.py)
#   exp165 = FixedCoeffDQN 5-3-1   (traincoeff_fixed.py)
#
# 建议先跑 run_exp163_166.sh (优先 5-4-1), 跑完后再跑这个.
#
# 用法:
#   cd experiments && bash run_exp161_162_164_165.sh
#   GPU=1 STAGGER=15 bash run_exp161_162_164_165.sh
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")" || { echo "cd 失败"; exit 1; }

CONFIGS=(
    "configs/exp161_gatcoeff_1x3_521_avg_waiting_NS20bus_1amb_U_tuned.yaml:traincoeff_gat.py"
    "configs/exp162_gatcoeff_1x3_531_avg_waiting_NS20bus_1amb_U_tuned.yaml:traincoeff_gat.py"
    "configs/exp164_fixedcoeff_1x3_521_avg_waiting_NS20bus_1amb_U_tuned.yaml:traincoeff_fixed.py"
    "configs/exp165_fixedcoeff_1x3_531_avg_waiting_NS20bus_1amb_U_tuned.yaml:traincoeff_fixed.py"
)

GPU="${GPU:-0}"
STAGGER="${STAGGER:-15}"
LOG_DIR="./logs/run_coord521_531_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PIDS=()
declare -A NAME_OF_PID
declare -A EXIT_CODES

trap '
    echo
    echo "[$(date)] 收到中断, 杀掉子进程..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  killed PID $pid"
    done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    exit 130
' INT TERM

echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 启动 ${#CONFIGS[@]} 个并行 (5-2-1 + 5-3-1) 实验"
echo "  GPU: $GPU, STAGGER: ${STAGGER}s, LOG_DIR: $LOG_DIR"
echo "  RAM 估算: 4 × ~2GB ≈ 8GB"
echo "  ⚠ 如果还有其它 run 在跑 (exp159/160 等), 注意监控 RAM"
echo "════════════════════════════════════════════════════════════════════════════"
echo

START_TIME=$SECONDS

for i in "${!CONFIGS[@]}"; do
    item="${CONFIGS[$i]}"
    cfg="${item%%:*}"
    script="${item##*:}"
    name=$(basename "$cfg" .yaml)
    log="$LOG_DIR/${name}.log"

    echo "[$(date)] ▶ LAUNCH  $name  (用 $script)"
    echo "  log: $log"

    python3 "$script" --config "$cfg" --gpu "$GPU" > "$log" 2>&1 &
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

echo "[$(date)] 全部启动, 等结束 ..."
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
echo "[$(date)] exp161/162/164/165 全部完成, 总挂钟 ${TOTAL}s (≈ $((TOTAL/60))min)"
for item in "${CONFIGS[@]}"; do
    name=$(basename "${item%%:*}" .yaml)
    rc=${EXIT_CODES[$name]}
    icon="✓"; [ "$rc" -ne 0 ] && icon="✗"
    echo "  $icon  $name  (exit $rc)"
done
echo "  log 目录: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════════════════"
