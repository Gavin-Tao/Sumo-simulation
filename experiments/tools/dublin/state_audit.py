"""状态审计 (2026-10-10, 只读分析, 不训练): 按 ckpt_env/train.py 同一套 obs 键构造环境, 跑若干决策步 (每 4 步轮换相位), 核对观测布局与特征语义 —— 维度是否全路口一致、is_green 多热位、since_green 放行为 0/未放行递增/上限、awt_log 值域、无非有限值。
用法 (任意目录): python experiments/tools/dublin/state_audit.py <exp> [steps=40]。注意: 8STD legacy 编码的 8 维独热靠 train.py 塞入的 std_action_map, 本脚本里 legacy 配置维度会按各路口相位数变化, 不是 bug。"""
import os, sys, functools, yaml, numpy as np
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO="/home/xiaowen/sumo-rl"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments"); os.chdir(f"{REPO}/experiments")
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.environment.observations import PRIORITY_LEVELS
exp=sys.argv[1]; steps=int(sys.argv[2]) if len(sys.argv)>2 else 40
import glob; cfg_path=[c for c in sorted(glob.glob(f"configs/exp{exp}_*.yaml")) if "traindry" not in c and "smoke" not in c][0]
cfg=yaml.safe_load(open(cfg_path))
kw={}
if "obs_fields" in cfg: kw["fields"]=tuple(cfg["obs_fields"])
if "obs_phase_state" in cfg: kw["phase_state"]=cfg["obs_phase_state"]
kw["priority_source"]=cfg.get("obs_priority_source", cfg.get("priority_source"))
if "obs_downstream" in cfg: kw["include_downstream"]=bool(cfg["obs_downstream"])
if "obs_downstream_fields" in cfg: kw["downstream_fields"]=tuple(cfg["obs_downstream_fields"])
if "obs_lane_occ" in cfg: kw["include_lane_occ"]=bool(cfg["obs_lane_occ"])
if "obs_awt_cap" in cfg: kw["awt_cap"]=float(cfg["obs_awt_cap"])
if "obs_awt_basis" in cfg: kw["awt_basis"]=str(cfg["obs_awt_basis"])
if "obs_slot_stats" in cfg: kw["slot_stats"]=str(cfg["obs_slot_stats"])
if "obs_since_green" in cfg: kw["include_since_green"]=bool(cfg["obs_since_green"])
if "obs_awt_log" in cfg: kw["awt_log"]=bool(cfg["obs_awt_log"])
print(f"[{exp}] {os.path.basename(cfg_path)}\n  obs kwargs: {kw}")
obs_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
reward_fn=make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
env=SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
    num_seconds=cfg.get("num_seconds",1000), min_green=cfg.get("min_green",5), max_green=cfg.get("max_green",50), use_max_green=cfg.get("use_max_green",False),
    single_agent=False, yellow_time=cfg.get("yellow_time",2), delta_time=cfg.get("delta_time",5), reward_fn=reward_fn, observation_class=obs_class,
    sumo_seed=cfg.get("seed",0), sumo_warnings=False)
states=env.reset(123)
dims=sorted({len(v) for v in states.values()}); print(f"  18 路口观测维度集合: {dims}  (路口数 {len(states)})")
ts_ids=list(env.ts_ids); K={t: env.traffic_signals[t].num_green_phases for t in ts_ids}
legacy=(kw.get("phase_state")=="legacy"); fields=kw.get("fields",("count","queue","mean_awt","max_awt")); phi=len(fields)*len(PRIORITY_LEVELS)
psi=len(PRIORITY_LEVELS)*len(kw.get("downstream_fields",("count","queue"))) if kw.get("include_downstream") else 0
slot_dim=phi+psi+(1 if kw.get("include_lane_occ") else 0)+(1 if kw.get("include_since_green") else 0)
head=(8+1) if legacy else 2; per_slot=slot_dim+(0 if legacy else 1)
print(f"  布局: 头 {head} + 12 槽 × {per_slot} (is_green {0 if legacy else 1} + φ {phi} + ψ {psi} + lane_occ {1 if kw.get('include_lane_occ') else 0} + since_green {1 if kw.get('include_since_green') else 0}) = {head+12*per_slot}")
def slot_view(vec, slot):
    o=head+slot*per_slot; d={}
    if not legacy: d["is_green"]=vec[o]; o+=1
    d["phi"]=vec[o:o+phi]; o+=phi
    if psi: d["psi"]=vec[o:o+psi]; o+=psi
    if kw.get("include_lane_occ"): d["occ"]=vec[o]; o+=1
    if kw.get("include_since_green"): d["since_green"]=vec[o]; o+=1
    return d
watch=["389281","cluster_135109528_9101656"]; probs=[]; allvals=[]
prev_sg={t:None for t in watch}
for step in range(steps):
    act={t: (step//4)%K[t] for t in ts_ids}     # 每 4 决策换一个相位, 轮流
    states,_,done,_=env.step(action=act)
    for t in ts_ids:
        v=states[t]; allvals.append(v)
        if not np.all(np.isfinite(v)): probs.append(f"step{step} {t}: 非有限值")
        if len(v)!=head+12*per_slot: probs.append(f"step{step} {t}: 维度 {len(v)}")
    for t in watch:
        v=states[t]; ts=env.traffic_signals[t]
        ig=[slot_view(v,s).get("is_green",np.nan) for s in range(12)]; sg=[slot_view(v,s).get("since_green",np.nan) for s in range(12)]
        if kw.get("include_since_green"):
            for s in range(12):
                if ig[s]==1.0 and sg[s]!=0.0: probs.append(f"step{step} {t} 槽{s}: 放行中但 since_green={sg[s]}")
                if prev_sg[t] is not None and ig[s]==0.0 and sg[s]<prev_sg[t][s]-1e-6 and sg[s]!=0.0: probs.append(f"step{step} {t} 槽{s}: 未放行但 since_green 下降 {prev_sg[t][s]}→{sg[s]}")
                if sg[s]>6.0+1e-6: probs.append(f"step{step} {t} 槽{s}: since_green 超上限 {sg[s]}")
            prev_sg[t]=sg
        if step in (0,1,5,12,25,steps-1):
            print(f"  step{step:3d} t={env.sim_step:5.0f}s {t[:24]:24s} 绿相 {ts.green_phase}/{K[t]} 头={np.round(v[:head],2).tolist()} is_green={''.join(str(int(x)) if x==x else '-' for x in ig)} since_green={np.round(sg,2).tolist() if kw.get('include_since_green') else '无'}")
    if done["__all__"]: break
A=np.vstack(allvals)
print(f"  全体观测值域: min {A.min():.3f} max {A.max():.3f}; mean_awt 列 (对数刻度应在 [0,1]) 取样: ", end="")
mi=[]; 
for s in range(12):
    o=head+s*per_slot+(0 if legacy else 1)
    for pi,p in enumerate(PRIORITY_LEVELS):
        for fi,f in enumerate(fields):
            if f=="mean_awt": mi.append(o+pi*len(fields)+fi)
print(f"min {A[:,mi].min():.3f} max {A[:,mi].max():.3f}")
print("  问题数:", len(probs)); print("\n".join("   "+p for p in probs[:10]))
env.close()
