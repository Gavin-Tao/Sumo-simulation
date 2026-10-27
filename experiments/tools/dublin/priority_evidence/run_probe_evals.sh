#!/bin/bash
# 换车型探针评估: 臂 × 种子 123-132 × 车型 (amb/bus/car), best 检查点, 逐秒收集器; 3 并行, 内存门控。输出 analysis/data/pervehicle_probe18h/
cd /home/xiaowen/sumo-rl/experiments; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARMS=${ARMS:-"459 460 461 462 457"}; SEEDS=${SEEDS:-"123 124 125 126 127 128 129 130 131 132"}; PAR=${PAR:-3}
R=../nets/dublin/weekday_18h_reroute_v2; mkdir -p analysis/data/pervehicle_probe18h logs/probe18h
jobs=(); for e in $ARMS; do for s in $SEEDS; do for v in amb bus car; do jobs+=("$e $s $v"); done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P $PAR -L 1 bash -c '
  e=$0; s=$1; v=$2; cd /home/xiaowen/sumo-rl/experiments
  ck=$(ls models/exp${e}_*/frozen_best_*/best_snapshot.pth | tail -1)
  while [ $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) -lt 1800 ]; do sleep 20; done
  python tools/dublin/eval_pervehicle.py $e --ckpt $ck --seed $s --routes "'$R'/car_weekday_18h.rou.xml,'$R'/bus_weekday_18h.rou.xml,'$R'/probe_${v}_18h.rou.xml" --collector --collector-1s --out analysis/data/pervehicle_probe18h --tag _probe${v}_coll1s > logs/probe18h/exp${e}_s${s}_${v}.log 2>&1
  tail -1 logs/probe18h/exp${e}_s${s}_${v}.log | cut -c1-120'
echo PROBE_EVALS_DONE
