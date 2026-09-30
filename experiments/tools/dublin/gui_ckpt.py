"""在 sumo-gui 里观看某个检查点的贪心策略 (只读; 需要桌面 DISPLAY)。
用法: DISPLAY=:1 python experiments/tools/dublin/gui_ckpt.py 290 --ckpt <best.pth> --seed 132 [--delay 50] [--track amb_1] [--routes ...]
GUI 会自动开始运行 (环境用 --start --quit-on-end 启动); 用界面上的 Delay 调速; 关闭窗口即结束。
"""
import argparse, os, sys
os.environ["SUMO_RL_LIBSUMO"] = "0"        # GUI 必须走 TraCI, 且要在导入 ckpt_env 之前设置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_env

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int); ap.add_argument("--delay", type=int, default=50)
    ap.add_argument("--track", default="amb_1", help="出现后自动跟踪的车辆 id"); ap.add_argument("--zoom", type=float, default=1200.0); ap.add_argument("--routes")
    a = ap.parse_args()
    B = ckpt_env.build(a.exp, a.ckpt, a.seed, extra_sumo=f"--delay {a.delay}", route_override=a.routes, use_gui=True)
    env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo
    print(f"[gui] exp{a.exp} {B['kind']} ckpt ep{B['ckpt_episode']} seed {B['seed']} — GUI 已启动, 跟踪车辆 {a.track}", flush=True)
    tracked = False; done = {"__all__": False}
    while not done["__all__"]:
        if not tracked and a.track in sumo.vehicle.getIDList():
            try:
                sumo.gui.trackVehicle("View #0", a.track); sumo.gui.setZoom("View #0", a.zoom); tracked = True
                print(f"[gui] t={sumo.simulation.getTime():.0f}s 开始跟踪 {a.track}", flush=True)
            except Exception as e: print("[gui] 跟踪失败:", e, flush=True); tracked = True
        states, _, done, _ = env.step(action={t: act(t, states[t])[0] for t in env.ts_ids})
    env.close()
if __name__ == "__main__": main()
