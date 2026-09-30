"""逐车评估 (2026-09-30, 用户批准 "新表, 不影响旧表与旧代码"): 用训练好的检查点在评估种子上跑一集, 导出每辆车的
行程指标 (SUMO tripinfo: duration / timeLoss / waitingTime / routeLength) 与逐路口进口道记录 (进入/离开时刻, 进入时连接灯色,
首次变绿时刻, 停车秒数, movement 连接序号)。只读分析, 不训练。
用法: python experiments/tools/dublin/eval_pervehicle.py <exp> [--ckpt path] [--seed 123] [--out dir]
输出: <out>/exp<exp>_ep<ckpt集>_seed<seed>.json
"""
import argparse, collections, json, os, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ckpt_env

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int); ap.add_argument("--out", default="analysis/data/pervehicle")
    a = ap.parse_args()
    tmp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "experiments", f"logs/tripinfo_tmp_exp{a.exp}_{os.getpid()}.xml"))
    B = ckpt_env.build(a.exp, a.ckpt, a.seed, extra_sumo=f"--tripinfo-output {tmp} --tripinfo-output.write-unfinished true")
    env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo
    print(f"[pervehicle] exp{a.exp} {B['kind']} ckpt ep{B['ckpt_episode']} seed {B['seed']} n_ts {len(env.ts_ids)}")
    lanes = {t: [l for l in dict.fromkeys(sumo.trafficlight.getControlledLanes(t))] for t in env.ts_ids}
    lane2tls = {l: t for t, ls in lanes.items() for l in ls}
    veh = {}                                  # vid -> 基本信息 + 逐口记录
    on = {}                                   # (vid, tls) -> 当前进口段记录
    def rec():
        now = sumo.simulation.getTime()
        for vid in sumo.simulation.getDepartedIDList():
            veh.setdefault(vid, {"type": sumo.vehicle.getTypeID(vid), "depart": now, "tls": {}})
        present = set()
        for t, ls in lanes.items():
            for l in ls:
                for vid in sumo.lane.getLastStepVehicleIDs(l):
                    key = (vid, t); present.add(key); v = veh.setdefault(vid, {"type": sumo.vehicle.getTypeID(vid), "depart": now, "tls": {}})
                    sp = sumo.vehicle.getSpeed(vid)
                    if key not in on:
                        nt = sumo.vehicle.getNextTLS(vid); link, st, dist = (None, None, None)
                        for x in nt:
                            if x[0] == t: link, dist, st = int(x[1]), round(float(x[2]), 1), x[3]; break
                        on[key] = {"entry": now, "entry_state": st, "entry_dist": dist, "link": link, "lane": l, "first_green": None, "stopped": 0, "min_speed": sp, "exit": None}
                    r = on[key]
                    if sp < 0.1: r["stopped"] += 1
                    r["min_speed"] = min(r["min_speed"], sp)
                    if r["first_green"] is None and r["link"] is not None:
                        for x in sumo.vehicle.getNextTLS(vid):
                            if x[0] == t and x[3] in ("G", "g"): r["first_green"] = now; break
        for key in [k for k in on if k not in present]:
            r = on.pop(key); r["exit"] = now; vid, t = key
            veh[vid]["tls"].setdefault(t, []).append(r)
    _orig = env._sumo_step
    def hooked(): _orig(); rec()
    env._sumo_step = hooked
    rec(); done = {"__all__": False}
    while not done["__all__"]:
        states, _, done, _ = env.step(action={t: act(t, states[t])[0] for t in env.ts_ids})
    for key, r in on.items(): r["exit"] = None; veh[key[0]]["tls"].setdefault(key[1], []).append(r)
    env.close()
    trip = {}
    if os.path.exists(tmp):
        for ti in ET.parse(tmp).getroot().iter("tripinfo"):
            trip[ti.get("id")] = {k: float(ti.get(k)) for k in ("depart", "arrival", "duration", "routeLength", "waitingTime", "waitingCount", "stopTime", "timeLoss") if ti.get(k) is not None}
            trip[ti.get("id")]["vType"] = ti.get("vType")
        os.remove(tmp)
    for vid, v in veh.items():
        v["trip"] = trip.get(vid)
    out = dict(exp=a.exp, kind=B["kind"], ckpt=B["ckpt"], ckpt_episode=B["ckpt_episode"], seed=B["seed"], cfg=B["cfg_path"], n_tripinfo=len(trip), n_vehicles=len(veh), vehicles=veh)
    os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, f"exp{a.exp}_ep{B['ckpt_episode']:05d}_seed{B['seed']}.json")
    json.dump(out, open(path, "w")); print("saved", path, "| vehicles", len(veh), "tripinfo", len(trip))
if __name__ == "__main__": main()
