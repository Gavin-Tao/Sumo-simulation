#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 分两批跑 exp145-150 (CoeffDQN ablation 矩阵, NS20bus_1amb_U 1x3)
#
# 6 个 config 全部 plain CoeffDQN (use_per=0, use_double=0):
#   BATCH 1: exp145/146/147 = controlled (hd=128, 与 CoLightOrig 同 hyperparam)
#   BATCH 2: exp148/149/150 = CoeffDQN-tuned (hd=256, capacity=80000)
#
# 设计:
#   - 批内并行(3 个一起跑,STAGGER 启动间隔)
#   - 批间串行(BATCH1 全部结束才启动 BATCH2)
#   - 控制 RAM 峰值: 3 × ~2GB ≈ 6GB,留足余量
#
# 用法:
#   cd experiments && bash run_exp145_150.sh                # GPU 0, 启动间隔 15s
#   GPU=1 STAGGER=20 bash run_exp145_150.sh
#   GPU=-1 bash run_exp145_150.sh                            # CPU
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")" || { echo "cd 到脚本目录失败"; exit 1; }

BATCH1=(
    "configs/exp145_coeff_1x3_521_avg_waiting_NS20bus_1amb_U_tuned.yaml"
    "configs/exp146_coeff_1x3_531_avg_waiting_NS20bus_1amb_U_tuned.yaml"
    "configs/exp147_coeff_1x3_541_avg_waiting_NS20bus_1amb_U_tuned.yaml"
)
BATCH2=(
    "configs/exp148_coeff_1x3_521_avg_waiting_NS20bus_1amb_U_tunedCoeff.yaml"
    "configs/exp149_coeff_1x3_531_avg_waiting_NS20bus_1amb_U_tunedCoeff.yaml"
    "configs/exp150_coeff_1x3_541_avg_waiting_NS20bus_1amb_U_tunedCoeff.yaml"
)

GPU="${GPU:-0}"
STAGGER="${STAGGER:-15}"
LOG_DIR="./logs/run_coeff_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 所有 PID 集中收集(用于 Ctrl+C 清理)
ALL_PIDS=()
declare -A NAME_OF_PID
declare -A EXIT_CODES

# Ctrl+C 拦截
trap '
    echo
    echo "[$(date)] 收到中断,杀掉所有 traincoeff 子进程..."
    for pid in "${ALL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  killed PID $pid"
    done
    sleep 2
    for pid in "${ALL_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    exit 130
' INT TERM

# 启一批 + 等它全部结束的函数
run_batch() {
    local batch_name=$1
    shift
    local cfgs=("$@")
    local pids=()

    echo "════════════════════════════════════════════════════════════════════════════"
    echo "[$(date)] ▶▶▶ $batch_name 启动 (${#cfgs[@]} 个并行)"
    echo "════════════════════════════════════════════════════════════════════════════"

    for i in "${!cfgs[@]}"; do
        cfg="${cfgs[$i]}"
        name=$(basename "$cfg" .yaml)
        log="$LOG_DIR/${name}.log"

        echo "[$(date)] ▶ LAUNCH  $name"
        echo "  cmd: python3 traincoeff.py --config $cfg --gpu $GPU"
        echo "  log: $log"

        python3 traincoeff.py --config "$cfg" --gpu "$GPU" > "$log" 2>&1 &
        pid=$!
        pids+=("$pid")
        ALL_PIDS+=("$pid")
        NAME_OF_PID[$pid]=$name
        echo "  PID: $pid"
        echo

        if [ "$i" -lt "$((${#cfgs[@]} - 1))" ]; then
            echo "[$(date)] stagger ${STAGGER}s ..."
            sleep "$STAGGER"
            echo
        fi
    done

    echo "[$(date)] $batch_name 全部启动,等结束 ..."
    echo "  实时看: tail -f $LOG_DIR/*.log"
    echo

    # 等本批所有 pid
    for pid in "${pids[@]}"; do
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
    echo
    echo "[$(date)] $batch_name 全部完成"
    echo
}

START_TIME=$SECONDS

echo "总览: ${#BATCH1[@]} + ${#BATCH2[@]} 个 run,分两批跑"
echo "  GPU: $GPU, STAGGER: ${STAGGER}s, LOG_DIR: $LOG_DIR"
echo "  RAM: 每批 3 × ~2GB ≈ 6GB (安全)"
echo

# 跑 BATCH 1 (controlled)
run_batch "BATCH 1 (controlled, hd=128)" "${BATCH1[@]}"

# 跑 BATCH 2 (tuned)
run_batch "BATCH 2 (tuned, hd=256)" "${BATCH2[@]}"

# 总览
TOTAL=$((SECONDS - START_TIME))
echo "════════════════════════════════════════════════════════════════════════════"
echo "[$(date)] 6 个 run 全部完成,总挂钟 ${TOTAL}s (≈ $((TOTAL/60))min)"
echo
echo "BATCH 1 (controlled):"
for cfg in "${BATCH1[@]}"; do
    name=$(basename "$cfg" .yaml)
    rc=${EXIT_CODES[$name]}
    icon="✓"; [ "$rc" -ne 0 ] && icon="✗"
    echo "  $icon  $name  (exit $rc)"
done
echo
echo "BATCH 2 (tuned):"
for cfg in "${BATCH2[@]}"; do
    name=$(basename "$cfg" .yaml)
    rc=${EXIT_CODES[$name]}
    icon="✓"; [ "$rc" -ne 0 ] && icon="✗"
    echo "  $icon  $name  (exit $rc)"
done
echo
echo "  log 目录: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════════════════"
