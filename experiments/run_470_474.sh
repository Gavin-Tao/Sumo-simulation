#!/bin/bash
# 2026-10-28 用户令 "等建立完新场景以后, 在 461 基础上做新场景的 531 351 111 配置, 3 个用 CPU 跑, 能同时跑就同时跑, 不能就先跑 531 351 最后跑 111;
#   然后建立 lanefix 的 531 111 配置, 用 CPU 跑; 不要浪费计算资源但是不要 OOM":
#   exp470/471/472 = 配对场景 (weekday_pairs50_lanefix) 531/351/111; exp473/474 = 18h 定稿 lanefix 场景 (weekday_18h_nocg_claude) 531/111。
#   五份配置都与 exp461 逐位相同, 只改 name/net_file/action_meta_file/route_file/cfg_file/priority_source。
# 本脚本 = run_exact_466.sh 改名字与默认臂, 调度逻辑不变: 内存门控 (BUDGET/RESERVE/投影增长), 按 ARMS 顺序放行 (470→471→472→473→474),
#   MAXPAR=5 只是上限, 实际并行数由内存决定; 默认 NOEVAL=1 (只训练; 配对场景的评估方案 [镜像/留出时刻表/排空] 用户另定)。
# 用法:  cd /home/xiaowen/sumo-rl/experiments && nohup bash run_470_474.sh > logs/run_470_474.out 2>&1 &
set -u
# 2026-10-26 22:37 事故: 训练一上 GPU, 内核即报 Xid 79 "GPU has fallen off the bus" (与 10-09 同), 黑屏死机只能强制重启。
#   → 训练一律 CPU (--gpu -1; 实测 CPU 与 GPU 每集耗时相当), 并用 systemd-inhibit 阻止锁屏后 5 分钟自动休眠 (电源设置 sleep-inactive-ac 300 s)。
if [ -z "${INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
  INHIBITED=1 exec systemd-inhibit --what=sleep:idle --mode=block --who=run_470_474 --why="训练中, 勿休眠" bash "$0" "$@"
fi
cd /home/xiaowen/sumo-rl/experiments
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARMS=${ARMS:-"470 471 472 473 474"}
BUDGET=${BUDGET:-2200}; RESERVE=${RESERVE:-2500}; MAXPAR=${MAXPAR:-5}
DRY=${DRY:-0}; NOEVAL=${NOEVAL:-1}
ST=logs/run_470_474_status.txt; mkdir -p logs analysis/data/pervehicle
DATE=$(date +%F)
declare -A PID NAME
log() { echo "$(date '+%F %T') $*" >> $ST; }
mem() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
rss_mb() { ps -o rss= -p "$1" 2>/dev/null | awk '{print int($1/1024)}'; }
running() { local n=0; for e in "${!PID[@]}"; do kill -0 "${PID[$e]}" 2>/dev/null && n=$((n+1)); done; echo $n; }
projected_growth() {   # 已在跑的臂还可能增长的内存 (MB)
  local g=0 r
  for e in "${!PID[@]}"; do
    kill -0 "${PID[$e]}" 2>/dev/null || continue
    r=$(rss_mb "${PID[$e]}"); r=${r:-0}
    [ "$r" -lt "$BUDGET" ] && g=$((g + BUDGET - r))
  done
  echo $g
}
can_start() { [ "$(running)" -lt "$MAXPAR" ] && [ $(( $(mem) - $(projected_growth) )) -ge $((BUDGET + RESERVE)) ]; }

posttrain() {   # $1=exp $2=配置名: 冻结 best, best × 10 种子逐秒评估 (内存门控, 两个并行)
  local e=$1 n=$2 d fr ep
  d=$(ls -d models/$n/2026-*/ 2>/dev/null | tail -1); [ -z "$d" ] && { log "EVAL exp$e 跳过: 没有模型目录"; return; }
  fr=models/$n/frozen_best_$DATE; mkdir -p $fr; cp $d/best.pth $fr/best_snapshot.pth; cp $d/best_metrics.json $fr/ 2>/dev/null
  ep=$(python -c "import json;print(json.load(open('$fr/best_metrics.json'))['_meta']['episode'])" 2>/dev/null || echo "?")
  log "EVAL exp$e best=ep$ep 开始 (10 种子, 逐秒口径)"
  ( cd /home/xiaowen/sumo-rl
    for s in $(seq 123 132); do
      while [ $(mem) -lt 1500 ]; do sleep 30; done
      python experiments/tools/dublin/eval_pervehicle.py $e --ckpt experiments/$fr/best_snapshot.pth --seed $s --collector --collector-1s --rl-delay --tag _best_coll1s \
        > experiments/logs/pv_exp${e}_best_s$s.log 2>&1 &
      [ $(( (s - 123) % 2 )) -eq 1 ] && wait     # 两个一组
    done; wait )
  log "EVAL exp$e best=ep$ep 完成: coll1s=$(ls analysis/data/pervehicle | grep -cE "^exp${e}_ep[0-9]+_seed[0-9]+_best_coll1s.json")"
}

launch() {   # $1=exp
  local e=$1 f n
  f=$(ls configs/exp${e}_*.yaml | head -1); n=$(basename $f .yaml)
  log "START $n (MemAvailable=$(mem)MB, 在跑 $(running) 臂, 投影增长 $(projected_growth)MB)"
  if [ "$DRY" = 1 ]; then
    ( sleep 20; log "DONE $n (演练)" ) & PID[$e]=$!
  else
    ( python train.py --config $f --gpu -1 > logs/${n}_train.log 2>&1; r=$?; log "DONE $n exited rc=$r"
      [ "$NOEVAL" = 1 ] || posttrain $e $n ) & PID[$e]=$!
  fi
  NAME[$e]=$n
}

# 启动前体检 (2026-10-26 用户令 "每次跑之前检查 GPU nvidia-smi 是否正常"): 训练虽走 CPU, 但这块卡掉总线会把整机带黑屏; 不正常就不起训练。
if timeout 20 nvidia-smi --query-gpu=name,temperature.gpu,power.draw,pcie.link.width.current --format=csv,noheader > /tmp/nvsmi_preflight.txt 2>&1; then
  log "PREFLIGHT nvidia-smi 正常: $(tr -d '\n' < /tmp/nvsmi_preflight.txt)"
else
  log "PREFLIGHT nvidia-smi 异常 (退出码 $? 或 20 秒无响应): $(head -c 200 /tmp/nvsmi_preflight.txt) —— 不启动训练, 请先处理 GPU"; exit 2
fi
log "SCHED run_470_474 启动 (CPU 训练, systemd-inhibit=${INHIBITED:-0}): ARMS=[$ARMS] BUDGET=${BUDGET}MB RESERVE=${RESERVE}MB MAXPAR=$MAXPAR DRY=$DRY NOEVAL=$NOEVAL MemAvailable=$(mem)MB"
for e in $ARMS; do
  while ! can_start; do
    sleep 30
    [ $(mem) -lt 1000 ] && log "WARN MemAvailable=$(mem)MB 低于 1 GB, 在跑 $(running) 臂"
  done
  launch $e
  sleep ${STAGGER:-120}      # 错开启动, 让前一臂的 SUMO 与 wandb 初始化完成再评估内存
done
wait
log "RUN_470_474_DONE 训练$([ "$NOEVAL" = 1 ] && echo '' || echo '与 best×10 逐秒评估')全部完成"
