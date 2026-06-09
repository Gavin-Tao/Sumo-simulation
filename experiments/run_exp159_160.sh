#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 跑 exp159 + exp160 (CoLightOrig + PER / + PER+Double, 5-4-1)
#
# 测试 PER 能否救 CoLight 在 5-4-1 上的 amb 退化:
#   exp135 = CoLightOrig plain         (amb=4.54, priority 颠倒)
#   exp159 = CoLightOrig + PER         ← 测 PER 是否能让 GAT 学到 amb 协同
#   exp160 = CoLightOrig + PER + Double ← 预期跟 1x1 一样抵消
#
# 都基于 exp135 (CoLightOrig 5-4-1 controlled),所以 use_per=true (+ exp160 use_double=true).
# wandb 会自动 log GAT attention (train/gat_attn_up/down/left/right + entropy).
#
# 用法:
#   cd experiments && bash run_exp159_160.sh
#   GPU=1 STAGGER=20 bash run_exp159_160.sh
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")" || { echo "cd 失败"; exit 1; }

CONFIGS=(
    "configs/exp159_colightorig_1x3_541_avg_waiting_NS20bus_1amb_U_PER.yaml"
    "configs/exp160_colightorig_1x3_541_avg_waiting_NS20bus_1amb_U_PER_Double.yaml"
)

GPU="${GPU:-0}"
STAGGER="${STAGGER:-15}"
LOG_DIR="./logs/run_colight_per_$(date +%Y%m%d_%H%M%S)"
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
echo "[$(date)] 启动 ${#CONFIGS[@]} 个并行 CoLightOrig+PER 实验"
echo "  GPU: $GPU, STAGGER: ${STAGGER}s"
echo "  LOG_DIR: $LOG_DIR"
echo "  ⚠ 注意: 当前 ~7.5GB RAM 已用,加 2 个 CoLight (~3GB) 会到 ~10.5GB / 15GB,紧但够"
echo "  ⚠ 如果跑爆,kill 后改成串行:bash 等第 1 个完成再启第 2 个"
echo "  ⚠ 跑的脚本是 trainorico.py (CoLightOrig 用这个,不是 traincoeff.py)"
echo "════════════════════════════════════════════════════════════════════════════"
echo

START_TIME=$SECONDS

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    log="$LOG_DIR/${name}.log"

    echo "[$(date)] ▶ LAUNCH  $name"
    echo "  cmd: python3 trainorico.py --config $cfg --gpu $GPU"
    echo "  log: $log"

    python3 trainorico.py --config "$cfg" --gpu "$GPU" > "$log" 2>&1 &
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
echo "  内存监控: watch -n 10 free -h"
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
