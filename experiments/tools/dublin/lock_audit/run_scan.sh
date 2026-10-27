# 锁死审计工具集 (2026-10-27): scan_locks.py 全检查点锁死扫描; force_lock_debug.py 强制/自然锁死逐步核对 (独立重算状态与奖励); stuck_trace.py 卡死车辆 SUMO 细节; batch_loss_probe.py 长等待样本对批损失/梯度的影响; analyze_force.py 读 force_lock_debug 输出。
#!/bin/bash
# 全检查点锁死扫描: 459 461 462 460 × ep50..800 × seed 123, 两并行, 内存门控
cd /home/xiaowen/sumo-rl/experiments
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
S=/tmp/claude-1000/-home-xiaowen-sumo-rl/0cdb38c0-320d-4e7a-96ec-a2c15b468974/scratchpad
SEEDS=${SEEDS:-123}
jobs_list=()
for e in ${ARMS:-459 461 462 460}; do
  n=$(ls -d models/exp${e}_*/ | head -1); d=$(ls -d ${n}2026-*/ | tail -1)
  for ck in $(ls $d/ckpt_ep*.pth); do for s in $SEEDS; do jobs_list+=("$e $ck $s"); done; done
done
printf '%s\n' "${jobs_list[@]}" | xargs -P ${PAR:-2} -L 1 bash -c '
  e=$0; ck=$1; s=$2; ep=$(basename $ck .pth | sed "s/ckpt_ep//")
  while [ $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) -lt 1800 ]; do sleep 20; done
  python '"$S"'/scan_locks.py $e $ck $s '"$S"'/lockscan/exp${e}_ep${ep}_s${s}.json > '"$S"'/lockscan/exp${e}_ep${ep}_s${s}.log 2>&1
  tail -1 '"$S"'/lockscan/exp${e}_ep${ep}_s${s}.log'
echo SCAN_DONE
