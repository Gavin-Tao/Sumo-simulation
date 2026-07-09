#!/bin/bash
# 修正版拟合链: kan_distill 需要绝对路径 (相对路径解析到仓库根而非 experiments/)
set -uo pipefail
cd /home/xiaowen/sumo-rl/experiments
V2=/home/xiaowen/sumo-rl/experiments/analysis/kan_data_v2
SP=/tmp/claude-1000/-home-xiaowen-sumo-rl/3a4321a6-ff88-4ed2-9027-c983fee3d07f/scratchpad
step(){ echo; echo "### [$(date +%H:%M:%S)] $1"; }

for t in x1_220 x1_228a x1_229a x1_228d x1_229b; do
  step "fit g $t (abs)"
  python tools/kan/kan_distill.py --data $V2/$t --target g --shared-levels \
    --out $V2/$t/fit_shared || echo "!!! fit $t FAILED"
done

step "wait dublin211_best extraction"
while [ ! -f $V2/dublin211_best/manifest.json ]; do sleep 120; done
step "fit g dublin211_best"
python tools/kan/kan_distill.py --data $V2/dublin211_best --target g --shared-levels \
  --out $V2/dublin211_best/fit_shared || echo "!!! fit dublin FAILED"

for tag in ep215 ep350 ep800; do
  step "wait vintage $tag extraction"
  while [ ! -f $V2/tgt_a211$tag/manifest.json ]; do sleep 120; done
  step "fit g vintage $tag"
  python tools/kan/kan_distill.py --data $V2/tgt_a211$tag --target g --shared-levels \
    --out $V2/tgt_a211$tag/fit_shared || echo "!!! fit $tag FAILED"
done

step "dublin s readout (idempotent)"
python $SP/dublin_s_readout.py || echo "!!! dublin s readout FAILED"
step "FITS ALL DONE"
