#!/bin/bash
# 2026-10-29 用户令 "先跑新场景, 等新场景跑完了再跑 18 的, 避免内存不够或者存储空间不够":
#   2026-10-29 用户令 "修改之前跟修改之后都跑一下, 跑完了以后再跑 18": 等新版 exp470/471/472 与旧版 exp475/476/477 的训练进程全部结束后, 才按内存门控启动 473/474 (调度逻辑同 run_470_474_b.sh)。
cd /home/xiaowen/sumo-rl/experiments
ST=logs/run_470_474_status.txt
until [ "$(grep -c 'START exp472' $ST)" -ge 2 ]; do sleep 60; done                       # 472 已重新启动
until [ "$(grep -c 'START exp477' $ST)" -ge 1 ]; do sleep 60; done                       # 旧版 477 已启动
while pgrep -f '^python train.py --config configs/exp47[012567]_' > /dev/null; do sleep 120; done   # 等新版 470–472 与旧版 475–477 全部跑完
echo "$(date '+%F %T') [d] exp470–472 与 475–477 全部结束, 开始跑 473/474 (磁盘可用 $(df -h --output=avail /home | tail -1 | tr -d ' '))" >> $ST
ARMS="473 474" exec bash run_470_474_b.sh
