"""只算 RL 路口的延误 (2026-10-09 用户令): 用检查点在评估种子上跑一集, 让 SUMO 按车类输出逐边 timeLoss (edgeData, vTypes 过滤, 含路口内部边),
只累加 18 个 RL 路口的进口道边 + 路口内部边上的时间损失。
  RL 延误/车   = Σ_RL边 timeLoss_c / 该类出发车辆数
  RL 延误/visit = Σ_RL进口道 timeLoss_c (含对应内部边) / Σ_RL进口道 entered_c   (分母 = 精确的进口道进入次数, 不是收集器采样)
同时记录整趟 timeLoss (tripinfo) 作对照。用法: python experiments/tools/dublin/eval_rl_delay.py <exp> --ckpt <pth> --seed <s> [--tag _rldelay]
输出: experiments/analysis/data/pervehicle/exp<exp>_ep<ep>_seed<seed><tag>.json"""
import argparse, json, os, sys, collections, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_env

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int); ap.add_argument("--out", default="analysis/data/pervehicle"); ap.add_argument("--tag", default="_rldelay")
    a = ap.parse_args(); pid = os.getpid()
    LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs")); os.makedirs(LOG, exist_ok=True)
    add = f"{LOG}/edgedata_add_exp{a.exp}_{pid}.xml"; trip = f"{LOG}/tripinfo_tmp_exp{a.exp}_{pid}.xml"; outs = {c: f"{LOG}/edgedata_{c}_exp{a.exp}_{pid}.xml" for c in ("car", "bus", "ambulance")}
    open(add, "w").write("<additional>" + "".join(f'<edgeData id="ed_{c}" file="{outs[c]}" freq="3600" vTypes="{c}" withInternal="true" excludeEmpty="true"/>' for c in outs) + "</additional>")
    B = ckpt_env.build(a.exp, a.ckpt, a.seed, extra_sumo=f"--additional-files {add} --tripinfo-output {trip} --tripinfo-output.write-unfinished true")
    env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo
    rl = list(env.ts_ids); approach = {t: sorted({l.rsplit("_", 1)[0] for l in dict.fromkeys(sumo.trafficlight.getControlledLanes(t))}) for t in rl}
    done = {"__all__": False}
    while not done["__all__"]:
        states, _, done, _ = env.step(action={t: act(t, states[t])[0] for t in rl})
    env.close()
    res = {}
    wt = {}
    if os.path.exists(trip):
        for ti in ET.parse(trip).getroot().iter("tripinfo"): wt.setdefault(ti.get("vType"), []).append(float(ti.get("timeLoss")))
        os.remove(trip)
    for c, f in outs.items():
        ed = {e.get("id"): e for iv in ET.parse(f).getroot().iter("interval") for e in iv.iter("edge")} if os.path.exists(f) else {}
        per_j = {}
        for t, eds in approach.items():
            tl = sum(float(ed[e].get("timeLoss", 0)) for e in eds if e in ed) + sum(float(e.get("timeLoss", 0)) for k, e in ed.items() if k.startswith(f":{t}_"))
            ent = sum(float(ed[e].get("entered", 0)) + float(ed[e].get("departed", 0)) for e in eds if e in ed)   # 进口道上出发的车也算一次 visit
            per_j[t] = dict(timeloss=tl, entered=ent)
        TL = sum(x["timeloss"] for x in per_j.values()); EN = sum(x["entered"] for x in per_j.values()); n = len(wt.get(c, []))   # 车辆数 = tripinfo 条数 (含未完成)
        all_tl = sum(float(e.get("timeLoss", 0)) for e in ed.values())   # 全网全部边 (含内部边) 的 timeLoss, 作对照
        res[c] = dict(n_vehicles=n, rl_timeloss=TL, rl_entered=EN, rl_per_vehicle=(TL / n if n else None), rl_per_visit=(TL / EN if EN else None), net_timeloss=all_tl, net_per_vehicle=(all_tl / n if n else None), per_junction=per_j)
        os.remove(f)
    os.remove(add)
    for c in res: res[c]["trip_timeloss_per_vehicle"] = (sum(wt[c]) / len(wt[c]) if wt.get(c) else None); res[c]["n_tripinfo"] = len(wt.get(c, []))
    out = dict(exp=a.exp, kind=B["kind"], ckpt=B["ckpt"], ckpt_episode=B["ckpt_episode"], seed=B["seed"], cfg=B["cfg_path"], rl_junctions=rl, approach_edges=approach, classes=res)
    os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, f"exp{a.exp}_ep{B['ckpt_episode']:05d}_seed{B['seed']}{a.tag}.json"); json.dump(out, open(path, "w"), indent=1)
    print("saved", path, "|", {c: (round(r["rl_per_vehicle"] or 0, 1), round(r["trip_timeloss_per_vehicle"] or 0, 1)) for c, r in res.items()})
if __name__ == "__main__": main()
