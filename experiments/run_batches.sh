#!/bin/bash
# 2026-10-29 用户令 "记得是用 CPU 跑, 3 个 3 个跑吧": 分三批, 每批最多 3 个 (CPU, --gpu -1, 内存门控同 run_470_474_b.sh):
#   第 1 批 470/471/472 (修改后, 已在跑) → 全部结束后第 2 批 475/476/477 (修改前, weekday_pairs50_lanefix_v1) → 全部结束后第 3 批 473/474 (18h lanefix)。
#   每批结束后 post_batch.sh 评估该批 (best × 10 种子逐秒) 并出 per-visit 三指标表与优先级反转检查, 与下一批训练并行。
# 用法:  cd /home/xiaowen/sumo-rl/experiments && nohup bash run_batches.sh > logs/run_batches.out 2>&1 &
set -u
if [ -z "${INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
  export INHIBITED=1; exec systemd-inhibit --what=sleep:idle --mode=block --who=run_batches --why="训练中, 勿休眠" bash "$0" "$@"
fi
cd /home/xiaowen/sumo-rl/experiments
ST=logs/run_470_474_status.txt
log() { echo "$(date '+%F %T') [批] $*" >> $ST; }
log "分批调度启动: 等第 1 批 (470/471/472) 跑完"
while pgrep -f '^python train.py --config' > /dev/null; do sleep 120; done
log "第 1 批结束, 开始第 2 批 475/476/477; 同时评估第 1 批并出表"
bash post_batch.sh 第1批_修改后_470-472 470 471 472 &      # 2026-10-29 用户令 "跑完一批就整理结果表": best × 10 种子逐秒评估 + per-visit 三指标表 + 反转检查
ARMS="475 476 477" MAXPAR=3 bash run_470_474_b.sh      # 该脚本在本批全部训练结束后才返回
wait
log "第 2 批结束, 开始第 3 批 473/474; 同时评估第 2 批并出表"
bash post_batch.sh 第2批_修改前_475-477 475 476 477 &
ARMS="473 474" MAXPAR=3 bash run_470_474_b.sh
wait
log "第 3 批结束, 评估第 3 批并出表"
bash post_batch.sh 第3批_18h_473-474 473 474
log "三批训练与评估全部完成"
