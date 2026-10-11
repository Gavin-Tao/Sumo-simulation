"""固定周期基线评估 (2026-10-11, 只读, 不训练): 按某个 8STD 配置构造环境, 每个路口按自己的绿相位顺序轮转 (或用第 4 个参数指定相位列表, 单路口场景), 每相位 G 秒;
收集器口径 (停车时间/次数 per visit) + tripinfo 吞吐 (进入/完成/出发延迟/整趟 timeLoss); 另存 eval_pervehicle 同构 JSON expfix<exp>g<G>_ep00000_seed<seed>_best_coll.json 供出图脚本当一臂读。
用法: python experiments/tools/dublin/fixed_cycle_eval.py <exp> <green_s> <seed> [phase,list]"""
用法: python fixed_cycle_any.py <exp> <green_s> <seed>"""
import os, sys, functools, yaml, glob, numpy as np, xml.etree.ElementTree as ET
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO="/home/xiaowen/sumo-rl"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments"); os.chdir(f"{REPO}/experiments")
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.environment.metrics import EpisodeMetricsCollector
exp, G, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]); CYC=[int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
TMP=f"/home/xiaowen/sumo-rl/experiments/logs/tripinfo_fixed_{os.getpid()}.xml"   # CYC 可选: 只轮转给定相位 (单路口场景)
cfg=yaml.safe_load(open([c for c in sorted(glob.glob(f"configs/exp{exp}_*.yaml")) if "traindry" not in c and "smoke" not in c][0]))
kw={}
if "obs_fields" in cfg: kw["fields"]=tuple(cfg["obs_fields"])
if "obs_phase_state" in cfg: kw["phase_state"]=cfg["obs_phase_state"]
kw["priority_source"]=cfg.get("obs_priority_source", cfg.get("priority_source"))
for a,b in (("obs_downstream","include_downstream"),("obs_lane_occ","include_lane_occ"),("obs_awt_cap","awt_cap"),("obs_awt_basis","awt_basis"),("obs_since_green","include_since_green"),("obs_awt_log","awt_log")):
    if a in cfg: kw[b]=cfg[a]
if "obs_downstream_fields" in cfg: kw["downstream_fields"]=tuple(cfg["obs_downstream_fields"])
obs=functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
env=SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False, num_seconds=cfg.get("num_seconds",3600), min_green=cfg.get("min_green",5), max_green=cfg.get("max_green",50), use_max_green=cfg.get("use_max_green",False), single_agent=False, yellow_time=cfg.get("yellow_time",2), delta_time=cfg.get("delta_time",5), reward_fn=make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source"))), observation_class=obs, sumo_seed=cfg.get("seed",0), sumo_warnings=False, additional_sumo_cmd=f"--tripinfo-output {TMP} --tripinfo-output.write-unfinished true")
states=env.reset(seed); ts_ids=list(env.ts_ids); K={t: env.traffic_signals[t].num_green_phases for t in ts_ids}
ts_lane_map={t: env.traffic_signals[t].signal_controlled_lanes for t in ts_ids}; always_green=set().union(*(env.traffic_signals[t].always_green_lanes for t in ts_ids))
mc=EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green)
done={"__all__": False}; i=0; sp=max(int(round(G/env.delta_time)),1)
while not done["__all__"]:
    mc.collect_step(env.sumo); a={t: (CYC[(i//sp)%len(CYC)] if CYC else (i//sp)%K[t]) for t in ts_ids}; states,_,done,_=env.step(action=a); i+=1
mc.collect_step(env.sumo); mc.finalize(env.sumo); d=mc.to_flat_dict(prefix="eval"); env.close()
tr=[ti for ti in ET.parse(TMP).getroot().iter("tripinfo")]; os.remove(TMP)
n=len(tr); fin=sum(1 for ti in tr if float(ti.get("arrival",-1))>=0); dd=np.mean([float(ti.get("departDelay",0)) for ti in tr]); tl=np.mean([float(ti.get("timeLoss",0)) for ti in tr])
g=lambda c,m: d.get(f"eval/system/{c}/{m}", float("nan"))
# 另存为 eval_pervehicle 同构 JSON (收集器汇总+逐车 k_v/停车, 逐车 trip), 文件名 expfix{exp}g{G}_ep00000_seed{seed}_best_coll.json, 供出图脚本当一臂读
import json
recs=dict(getattr(mc,"_finalized",{})); recs.update(getattr(mc,"_active",{}))
coll={"summary":{k:(float(v) if isinstance(v,(int,float)) else v) for k,v in d.items() if isinstance(v,(int,float))},
      "vehicles":{vid:{"type":r["type"],"k_v":len(r.get("ts_visited",set())),"stopped_sec":dict(r.get("per_ts_stopped_sec",{})),"stop_events":dict(r.get("per_ts_stop_events",{}))} for vid,r in recs.items()}}
veh={}
for ti in tr:
    veh[ti.get("id")]={"type":ti.get("vType"),"trip":{k:float(ti.get(k)) for k in ("depart","departDelay","arrival","duration","routeLength","waitingTime","waitingCount","stopTime","timeLoss") if ti.get(k) is not None}}
out=dict(exp=f"fix{exp}g{G:.0f}", kind="fixed_cycle", green_s=G, cycle=CYC, collector=coll, ckpt=None, ckpt_episode=0, seed=seed, cfg=None, n_tripinfo=n, n_vehicles=len(veh), vehicles=veh)
os.makedirs("analysis/data/pervehicle", exist_ok=True); json.dump(out, open(f"analysis/data/pervehicle/expfix{exp}g{G:.0f}_ep00000_seed{seed}_best_coll.json","w"))
print(f"exp{exp} G={G:.0f} seed={seed}: 停车/visit all {g('all','avg_stopped_time_per_visit'):.2f} car {g('car','avg_stopped_time_per_visit'):.2f} bus {g('bus','avg_stopped_time_per_visit'):.2f} amb {g('ambulance','avg_stopped_time_per_visit'):.2f} | 次数 all {g('all','avg_stop_events_per_visit'):.3f} | 进入 {n} 完成 {fin} 出发延迟 {dd:.1f} timeLoss {tl:.1f}")
