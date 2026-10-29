#!/bin/bash
# 2026-10-29 用户令 "跑完一批就整理结果表": 对一批实验冻结 best → best × 10 种子逐秒评估 (与 run_exact_466.sh 的 posttrain 同口径, CPU) → 出表 + 优先级反转检查。
# 用法: bash post_batch.sh <表名> <exp> [<exp> ...]      (在 experiments/ 下)
set -u
cd /home/xiaowen/sumo-rl/experiments
TAG=$1; shift; ST=logs/run_470_474_status.txt; DATE=$(date +%F)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "$(date '+%F %T') [评估] $*" >> $ST; }
mem() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
for e in "$@"; do
  f=$(ls configs/exp${e}_*.yaml | head -1); n=$(basename $f .yaml)
  d=$(ls -d models/$n/2026-*/ 2>/dev/null | tail -1); [ -z "$d" ] && { log "exp$e 跳过: 没有模型目录"; continue; }
  [ -f $d/best.pth ] || { log "exp$e 跳过: $d 里没有 best.pth"; continue; }
  fr=models/$n/frozen_best_$DATE; mkdir -p $fr; cp $d/best.pth $fr/best_snapshot.pth; cp $d/best_metrics.json $fr/ 2>/dev/null
  ep=$(python -c "import json;print(json.load(open('$fr/best_metrics.json'))['_meta']['episode'])" 2>/dev/null || echo "?")
  log "exp$e best=ep$ep 开始 10 种子逐秒评估 (模型 $d)"
  ( cd /home/xiaowen/sumo-rl
    for s in $(seq 123 132); do
      while [ $(mem) -lt 2500 ]; do sleep 30; done
      python experiments/tools/dublin/eval_pervehicle.py $e --ckpt experiments/$fr/best_snapshot.pth --seed $s --collector --collector-1s --rl-delay --tag _best_coll1s \
        > experiments/logs/pv_exp${e}_best_s$s.log 2>&1 &
      [ $(( (s - 123) % 3 )) -eq 2 ] && wait     # 三个一组
    done; wait )
  log "exp$e 评估完成: $(ls analysis/data/pervehicle | grep -cE "^exp${e}_ep[0-9]+_seed[0-9]+_best_coll1s.json") 个结果文件"
done
( cd /home/xiaowen/sumo-rl && python experiments/tools/dublin/batch_tables.py --tag "$TAG" "$@" > experiments/logs/batch_tables_${TAG}.log 2>&1 )
log "$TAG 结果表: experiments/analysis/data/batch_tables/$TAG/tables.md (反转检查见表内); 日志 logs/batch_tables_${TAG}.log"
