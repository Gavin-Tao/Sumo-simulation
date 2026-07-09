#!/bin/bash
# 2026-07-09 重验编排: 昨天 KAN 文档的全部数字从 ckpt 重新跑出 (fresh kan_data_v2/)
# 只读分析, 不碰训练码/旧数据(kan_data/ 保留作 diff 基线; mh219 例外先备份)
set -uo pipefail
cd /home/xiaowen/sumo-rl/experiments
V2=analysis/kan_data_v2
SP=/tmp/claude-1000/-home-xiaowen-sumo-rl/3a4321a6-ff88-4ed2-9027-c983fee3d07f/scratchpad
mkdir -p $V2
step(){ echo; echo "### [$(date +%H:%M:%S)] $1"; }

# ---------- A1: 1x1 五臂重提取 (与昨天同 cfg/ckpt/seeds) ----------
declare -A CFG CKPT
CFG[x1_220]=configs/exp220_1x1_531_NS20bus_enumfrap_cqm.yaml
CKPT[x1_220]=models/exp220_1x1_531_NS20bus_enumfrap_cqm/2026-07-03T23-26-15/best.pth
CFG[x1_228a]=configs/exp228a_1x1_ft421_physfrozen.yaml
CKPT[x1_228a]=models/exp228a_1x1_ft421_physfrozen/2026-07-07T17-39-16/best.pth
CFG[x1_229a]=configs/exp229a_1x1_gold421_scratch.yaml
CKPT[x1_229a]=models/exp229a_1x1_gold421_scratch/2026-07-07T18-26-23/best.pth
CFG[x1_228d]=configs/exp228d_1x1_ft351_physfrozen.yaml
CKPT[x1_228d]=models/exp228d_1x1_ft351_physfrozen/2026-07-07T17-39-16/best.pth
CFG[x1_229b]=configs/exp229b_1x1_gold351_scratch.yaml
CKPT[x1_229b]=models/exp229b_1x1_gold351_scratch/2026-07-07T18-26-23/best.pth

for t in x1_220 x1_228a x1_229a x1_228d x1_229b; do
  step "extract $t"
  python tools/kan/extract_frap_targets.py --config ${CFG[$t]} --ckpt ${CKPT[$t]} \
    --seeds 123,2000,2001,2002,2003 --eps-seeds 3000,3001 --out $V2/$t \
    || echo "!!! extract $t FAILED"
done

# ---------- A2: 1x1 五臂 g 拟合 (shared-levels, 与昨天同法) ----------
for t in x1_220 x1_228a x1_229a x1_228d x1_229b; do
  step "fit g $t"
  python tools/kan/kan_distill.py --data $V2/$t --target g --shared-levels \
    --out $V2/$t/fit_shared || echo "!!! fit $t FAILED"
done

# ---------- A3: x1 S 通道读出 (原法+修正法双跑) ----------
step "s readout x1 (v2 data)"
python $SP/s_readout_v2.py || echo "!!! s_readout FAILED"

# ---------- A4: mh219 重跑 (脚本硬编码输出路径, 先备份旧) ----------
step "mh219 re-run (8 seeds)"
cp analysis/kan_data/mh219/distortion.json analysis/kan_data/mh219/distortion_0708_orig.json
python tools/kan/multihead_distortion.py 123,2000,2001,2002,2003,2004,2005,2006 \
  || echo "!!! mh219 FAILED"
if [ -f analysis/kan_data/mh219/distortion.json ]; then
  cp analysis/kan_data/mh219/distortion.json $V2/mh219_distortion.json
  cp analysis/kan_data/mh219/distortion_0708_orig.json analysis/kan_data/mh219/distortion.json
fi

# ---------- A5: Dublin 主定理数据重提取 (最重, 8 runs) ----------
step "extract dublin211 best (8 runs, slow)"
python tools/kan/extract_frap_targets.py --config configs/exp211_dublin11h_531_enumfrap.yaml \
  --ckpt models/exp211_dublin11h_531_enumfrap/2026-07-02T17-30-29/best.pth \
  --seeds 123,2000,2001,2002,2003,2004 --eps-seeds 3000,3001 \
  --out $V2/dublin211_best || echo "!!! extract dublin FAILED"
step "fit g dublin211 best (shared-levels core16)"
python tools/kan/kan_distill.py --data $V2/dublin211_best --target g --shared-levels \
  --out $V2/dublin211_best/fit_shared || echo "!!! fit dublin FAILED"

# ---------- A6: vintage 三点重提取 (统一 4 runs) + 拟合 ----------
for pair in "ep215 best.pth" "ep350 ckpt_ep00350.pth" "ep800 ckpt_ep00800.pth"; do
  set -- $pair; tag=$1; ck=$2
  step "extract vintage $tag"
  python tools/kan/extract_frap_targets.py --config configs/exp211_dublin11h_531_enumfrap.yaml \
    --ckpt models/exp211_dublin11h_531_enumfrap/2026-07-02T17-30-29/$ck \
    --seeds 123,2000,2001 --eps-seeds 3000 --out $V2/tgt_a211$tag \
    || echo "!!! extract $tag FAILED"
  step "fit g vintage $tag"
  python tools/kan/kan_distill.py --data $V2/tgt_a211$tag --target g --shared-levels \
    --out $V2/tgt_a211$tag/fit_shared || echo "!!! fit $tag FAILED"
done

# ---------- A7: Dublin S 通道读出 (主 + 三 vintage; amb恒0 主张检验) ----------
step "dublin s readout"
python $SP/dublin_s_readout.py || echo "!!! dublin s readout FAILED"

step "ALL DONE"
