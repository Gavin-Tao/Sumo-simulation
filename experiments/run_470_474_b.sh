#!/bin/bash
# 2026-10-28 补充调度: run_470_474.sh 已起 470/471 后卡住 —— 它的 "投影增长" 读的是外层子 shell 的 RSS (约 4 MB) 而不是 python 训练进程 (约 1.6 GB),
#   每个在跑的臂都被当成还要再涨 BUDGET, 重复计算, 只放行 2 臂。本脚本接着放行 472 → 473 → 474 (同一顺序, 同一 BUDGET/RESERVE),
#   内存按 "所有 train.py 进程连同其直接子进程 (SUMO) 的实际 RSS" 计算 (含 470/471), 并在全部训练结束前持有防休眠锁。
# 第二版 (23:45): 第一版用 pgrep -f 'python train.py --config' 会匹配到启动命令的 shell, 递归统计子孙时把调度器自己算进去, 派生失控, 已整组杀掉; 改为行首匹配、不递归。
# 用法:  cd /home/xiaowen/sumo-rl/experiments && nohup bash run_470_474_b.sh > logs/run_470_474_b.out 2>&1 &
set -u
if [ -z "${INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
  INHIBITED=1 exec systemd-inhibit --what=sleep:idle --mode=block --who=run_470_474_b --why="训练中, 勿休眠" bash "$0" "$@"
fi
cd /home/xiaowen/sumo-rl/experiments
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARMS=${ARMS:-"472 473 474"}
BUDGET=${BUDGET:-2200}; RESERVE=${RESERVE:-2500}; MAXPAR=${MAXPAR:-5}
ST=logs/run_470_474_status.txt
log() { echo "$(date '+%F %T') [b] $*" >> $ST; }
mem() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
tree_rss_mb() {   # 训练进程 + 它的直接子进程 (SUMO) 的 RSS 之和 (MB); 不递归 (第一版递归 + 宽松匹配导致派生失控, 2026-10-28 23:43)
  local p=$1
  ps -o rss= -p "$p" --ppid "$p" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}'
}
trainers() { pgrep -f '^python train.py --config'; }     # 只认命令行以 python train.py 开头的进程 (不会匹配到含这段文字的 shell)
running() { trainers | wc -l; }
projected_growth() { local g=0 r p; for p in $(trainers); do r=$(tree_rss_mb "$p"); r=${r:-0}; [ "$r" -lt "$BUDGET" ] && g=$((g + BUDGET - r)); done; echo $g; }
can_start() { [ "$(running)" -lt "$MAXPAR" ] && [ $(( $(mem) - $(projected_growth) )) -ge $((BUDGET + RESERVE)) ]; }
if timeout 20 nvidia-smi --query-gpu=name,temperature.gpu,power.draw,pcie.link.width.current --format=csv,noheader > /tmp/nvsmi_preflight_b.txt 2>&1; then
  log "PREFLIGHT nvidia-smi 正常: $(tr -d '\n' < /tmp/nvsmi_preflight_b.txt)"
else
  log "PREFLIGHT nvidia-smi 异常: $(head -c 200 /tmp/nvsmi_preflight_b.txt) —— 不启动训练"; exit 2
fi
log "SCHED 补充调度启动: ARMS=[$ARMS] BUDGET=${BUDGET}MB RESERVE=${RESERVE}MB MAXPAR=$MAXPAR MemAvailable=$(mem)MB 在跑 $(running) 臂 投影增长 $(projected_growth)MB"
for e in $ARMS; do
  while ! can_start; do sleep 30; [ $(mem) -lt 1000 ] && log "WARN MemAvailable=$(mem)MB 低于 1 GB, 在跑 $(running) 臂"; done
  f=$(ls configs/exp${e}_*.yaml | head -1); n=$(basename $f .yaml)
  log "START $n (MemAvailable=$(mem)MB, 在跑 $(running) 臂, 投影增长 $(projected_growth)MB)"
  ( python train.py --config $f --gpu -1 > logs/${n}_train.log 2>&1; log "DONE $n exited rc=$?" ) &
  sleep ${STAGGER:-120}
done
while [ "$(running)" -gt 0 ]; do sleep 60; done     # 全部训练 (含 470/471) 结束前不释放防休眠锁
log "RUN_470_474_DONE 训练全部完成 (补充调度)"
