#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 并行跑 exp142 / exp143 / exp144(Double / PER / Double+PER 三组 ablation)
#
# 用法:
#   cd experiments && bash run_exp142_143_144.sh           # GPU 0, 启动间隔 15s
#   GPU=1 STAGGER=20 bash run_exp142_143_144.sh
#   GPU=-1 bash run_exp142_143_144.sh                       # CPU
#
# 为什么要 stagger 不能"哗"一下三个同时启:
#   - 多个 wandb.init 同一毫秒触发可能撞 auth / cache 锁
#   - 多个 SUMO 进程同时初始化也偶发卡(端口/共享文件)
#   - 串行启,但每个启动后立刻进入并行训练 → 整体跑时间 ≈ 单个 + 2×stagger,
#     比纯串行省 2/3 时间
#
# 一个崩了不会拉爆其他 — 用后台进程 + wait,各自捕 exit code。
# Ctrl+C 会 kill 掉所有子进程清理退出,不留僵尸。
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")" || { echo "cd 到脚本目录失败"; exit 1; }

CONFIGS=(
    "configs/exp142_1x1_521_avg_waiting_NSamb_tuned_double.yaml"
    "configs/exp143_1x1_521_avg_waiting_NSamb_tuned_per.yaml"
    "configs/exp144_1x1_521_avg_waiting_NSamb_tuned_double_per.yaml"
)

GPU="${GPU:-0}"
STAGGER="${STAGGER:-15}"
LOG_DIR="./logs/run_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PIDS=()
NAMES=()

# Ctrl+C 时 kill 所有子进程
trap '
    echo
    echo "[$(date)] 收到中断,杀掉所有训练子进程..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  killed PID $pid"
    done
    # 等几秒让子进程真退,然后 SIGKILL 兜底
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    exit 130
' INT TERM

echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 并行 ablation: ${#CONFIGS[@]} 个实验"
echo "  GPU       : $GPU"
echo "  启动间隔  : ${STAGGER}s(避免 wandb/SUMO 初始化并发抢资源)"
echo "  脚本日志  : $LOG_DIR"
echo "  实时监控  : tail -f $LOG_DIR/*.log"
echo "════════════════════════════════════════════════════════════════════════════"
echo

# 依次启动(每个之间 STAGGER sleep),后台跑
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    log="$LOG_DIR/${name}.log"

    echo "[$(date)] ▶ LAUNCH  $name"
    echo "  cmd: python3 train.py --config $cfg --gpu $GPU"
    echo "  log: $log"

    python3 train.py --config "$cfg" --gpu "$GPU" > "$log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    NAMES+=("$name")
    echo "  PID: $pid"
    echo

    # 不是最后一个就 sleep STAGGER
    if [ "$i" -lt "$((${#CONFIGS[@]} - 1))" ]; then
        echo "[$(date)] stagger ${STAGGER}s 再启下一个..."
        sleep "$STAGGER"
        echo
    fi
done

echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 全部 ${#PIDS[@]} 个已在后台跑,等它们结束..."
echo "  PIDs : ${PIDS[@]}"
echo "  实时看: tail -f $LOG_DIR/*.log"
echo "  单看 1 个: tail -f $LOG_DIR/${NAMES[0]}.log"
echo "════════════════════════════════════════════════════════════════════════════"
echo

# 等每个进程结束,记录 exit code
declare -A EXIT_CODES
declare -A ELAPSED
START_TIME=$SECONDS
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    wait "$pid"
    rc=$?
    EXIT_CODES[$name]=$rc
    ELAPSED[$name]=$((SECONDS - START_TIME))
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] ✓ $name 结束 (exit 0, 总耗时 ${ELAPSED[$name]}s)"
    else
        echo "[$(date)] ✗ $name 失败 (exit $rc, 总耗时 ${ELAPSED[$name]}s)"
        echo "  最后 5 行 log:"
        tail -5 "$LOG_DIR/${name}.log" | sed 's/^/    /'
    fi
done

# 总览
TOTAL=$((SECONDS - START_TIME))
echo
echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 全部完成,总挂钟时间 ${TOTAL}s。总览:"
for name in "${NAMES[@]}"; do
    rc=${EXIT_CODES[$name]}
    icon="✓"; [ "$rc" -ne 0 ] && icon="✗"
    echo "  $icon  $name  (exit $rc)"
done
echo "  log 目录: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════════════════"
