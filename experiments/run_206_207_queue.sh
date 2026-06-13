#!/bin/bash
# 内存安全排队 (2026-06-12): 等 205/208 (PID 108244/108245) 退出释放 ~10.4GB
# 后, 自动并行启动 206+207。用户命令: "留着现在正在跑的, 然后后台一起跑206 207"。
cd "$(dirname "$0")"
echo "[queue] waiting for 108244 (exp205) and 108245 (exp208) to exit..."
while ps -p 108244 >/dev/null 2>&1 || ps -p 108245 >/dev/null 2>&1; do
  sleep 60
done
sleep 30   # 等内存归还
avail=$(free -g | awk 'NR==2{print $7}')
echo "[queue] both exited at $(date +%H:%M), available mem ${avail}G"
if [ "$avail" -lt 11 ]; then
  echo "[queue] ABORT: available ${avail}G < 11G, not launching (manual check needed)"
  exit 1
fi
nohup python3 train.py --config configs/exp206_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_rfloor.yaml > logs/dublin_queue/exp206.log 2>&1 &
echo "[queue] exp206 PID $!"
nohup python3 train.py --config configs/exp207_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095_rfloor.yaml > logs/dublin_queue/exp207.log 2>&1 &
echo "[queue] exp207 PID $!"
echo "[queue] done $(date +%H:%M)"
