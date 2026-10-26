#!/bin/bash
# 新口径四臂训练调度 (2026-10-26, 用户令 "写个 BASH 运行 459-462, 不要 OOM, 也不要浪费计算资源"):
#   exp459 (541) / exp460 (111) / exp461 (531) / exp462 (521), 都柏林 18h 改道第二版, 分槽网, visit 口径, 800 集。
# 并行策略: 内存门控而不是写死 2 个或 4 个。
#   - 每臂预算 BUDGET=2200 MB (实测: torch+DQN 433 MB + 回放池装满 883 MB + SUMO/收集器 ≈ 1.6 GB, 留 35% 余量);
#   - 回放池要 ~25 集 (~10 分钟) 才装满, 所以不能只看当前 MemAvailable: 按 "已在跑的臂还会长多少" 投影后再判断:
#       可用 − Σ max(0, BUDGET − 该臂当前 RSS) ≥ BUDGET + RESERVE  才起下一臂;
#   - RESERVE=2500 MB 留给系统与评估; 最多 MAXPAR=4 臂 (16 核, 每进程单线程, CPU 不是瓶颈)。
#   VSCode 关掉后可用约 12-13 GB → 四臂通常会全部并行; 内存紧时自动只跑 2-3 臂, 等前面的跑完再补。
# 训练后: 冻结 best → best × 10 种子逐秒评估 (eval_pervehicle --collector --collector-1s --rl-delay, 与 449/456/457/458 的报表同口径)。
# 用法:  cd /home/xiaowen/sumo-rl/experiments && nohup bash run_exact_459_462.sh > logs/run_exact_459_462.out 2>&1 &
#   环境变量: ARMS="459 460 461 462" (默认四臂), BUDGET/RESERVE/MAXPAR 可覆盖, DRY=1 只演练调度不训练, NOEVAL=1 跳过训练后评估。
set -u
# 2026-10-26 22:37 事故: 训练一上 GPU, 内核即报 Xid 79 "GPU has fallen off the bus" (与 10-09 同), 黑屏死机只能强制重启。
#   → 训练一律 CPU (--gpu -1; 实测 CPU 与 GPU 每集耗时相当), 并用 systemd-inhibit 阻止锁屏后 5 分钟自动休眠 (电源设置 sleep-inactive-ac 300 s)。
if [ -z "${INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
  INHIBITED=1 exec systemd-inhibit --what=sleep:idle --mode=block --who=run_exact_459_462 --why="训练中, 勿休眠" bash "$0" "$@"
fi
cd /home/xiaowen/sumo-rl/experiments
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARMS=${ARMS:-"459 460 461 462"}
BUDGET=${BUDGET:-2200}; RESERVE=${RESERVE:-2500}; MAXPAR=${MAXPAR:-4}
DRY=${DRY:-0}; NOEVAL=${NOEVAL:-0}
ST=logs/exact_459_462_status.txt; mkdir -p logs analysis/data/pervehicle
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

log "SCHED exact_459_462 启动 (CPU 训练, systemd-inhibit=${INHIBITED:-0}): ARMS=[$ARMS] BUDGET=${BUDGET}MB RESERVE=${RESERVE}MB MAXPAR=$MAXPAR DRY=$DRY NOEVAL=$NOEVAL MemAvailable=$(mem)MB"
for e in $ARMS; do
  while ! can_start; do
    sleep 30
    [ $(mem) -lt 1000 ] && log "WARN MemAvailable=$(mem)MB 低于 1 GB, 在跑 $(running) 臂"
  done
  launch $e
  sleep ${STAGGER:-120}      # 错开启动, 让前一臂的 SUMO 与 wandb 初始化完成再评估内存
done
wait
log "EXACT_459_462_DONE 四臂训练$([ "$NOEVAL" = 1 ] && echo '' || echo '与 best×10 逐秒评估')全部完成"
