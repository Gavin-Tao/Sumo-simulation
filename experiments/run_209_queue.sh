#!/bin/bash
# 第三棒 (2026-06-12 用户命令: "全部的都运行完了, 机器空余了再运行209"):
# 等 206/207 (由 run_206_207_queue.sh 拉起) 全部退出 → 内存校验 → 单独运行 209。
cd "$(dirname "$0")"
QLOG=logs/dublin_queue/queue_206_207.log

# 阶段1: 等第一棒交接完成 (206/207 已启动)
echo "[queue209] waiting for 206/207 to be launched..."
while ! grep -qE "\[queue\] (done|ABORT)" "$QLOG" 2>/dev/null; do sleep 120; done
if grep -q "ABORT" "$QLOG"; then
  echo "[queue209] upstream queue ABORTED — not proceeding (manual check)"
  exit 1
fi
P206=$(grep -oE "exp206 PID [0-9]+" "$QLOG" | grep -oE "[0-9]+")
P207=$(grep -oE "exp207 PID [0-9]+" "$QLOG" | grep -oE "[0-9]+")
echo "[queue209] watching exp206 ($P206) and exp207 ($P207)..."

# 阶段2: 等 206/207 全部退出
while ps -p "$P206" >/dev/null 2>&1 || ps -p "$P207" >/dev/null 2>&1; do
  sleep 120
done
sleep 30
avail=$(free -g | awk 'NR==2{print $7}')
echo "[queue209] 206/207 exited at $(date +%H:%M), available mem ${avail}G"
# colight buffer @300k ≈ 9.4G + 运行时开销 → 需要 ≥11G
if [ "$avail" -lt 11 ]; then
  echo "[queue209] ABORT: available ${avail}G < 11G — not launching 209"
  exit 1
fi
nohup python3 trainorico.py --config configs/exp209_dublin11h_531_colightorig_B_movement_legacy_cqm_outcq_mask_stab_obsfix_clamp_eps05_g095.yaml > logs/dublin_queue/exp209.log 2>&1 &
echo "[queue209] exp209 PID $! launched at $(date +%H:%M)"
echo "[queue209] done"
