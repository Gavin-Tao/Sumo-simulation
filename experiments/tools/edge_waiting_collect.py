"""Collect per-EDGE per-class delta waiting (waiting_inc) for fixed + exp208,
3 seeds averaged. Dumps analysis/edge_waiting_data.json.
delta wait on an edge = sum over vehicles of (accumulated-wait gained while on
that edge), averaged per vehicle-of-class that traversed the edge."""
import os, sys, json, functools, glob
import numpy as np
os.environ['SUMO_RL_LIBSUMO']='1'; sys.path.insert(0,'/home/xiaowen/sumo-rl'); sys.path.insert(0,'/home/xiaowen/sumo-rl/experiments')
import yaml, torch
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.agents.dqn_agent_txw import DQN

CFG="configs/exp208_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095.yaml"
cfg=yaml.safe_load(open(CFG)); SEEDS=list(range(2000,2015))
CLASSES=["car","bus","ambulance"]
def vclass(t):
    t=t.lower()
    return "ambulance" if "amb" in t else ("bus" if "bus" in t else "car")

def collect_one(fixed, ckpt, seed):
    kw=dict(net_file=cfg['net_file'],route_file=cfg['route_file'],cfg_file=cfg['cfg_file'],
        out_csv_name=None,use_gui=False,num_seconds=3600,delta_time=5,yellow_time=2,min_green=5,
        max_green=50,use_max_green=True,single_agent=False,sumo_seed=seed,fixed_ts=fixed)
    if not fixed:
        kw["reward_fn"]=make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
        kw["observation_class"]=functools.partial(obsmod.PriorityMovementLegacyCQM,
            priority_source=cfg["priority_source"],include_downstream=True,downstream_fields=("count","queue"),
            awt_basis=cfg.get("obs_awt_basis","global"),awt_cap=cfg.get("obs_awt_cap"),
            include_lane_occ=cfg.get("obs_lane_occ",False))
    env=SumoEnvironment(**kw); sumo=None
    meta=json.load(open(cfg["action_meta_file"]))["tls"]
    ts_mask,std2green,green2std,ts_turn={},{},{},{}
    for tid,t in meta.items():
        ts_mask[tid]=np.array(t["mask"],bool); s2g=np.full(8,-1,int)
        for a,gi in t["std_to_green_index"].items(): s2g[int(a)]=gi
        std2green[tid]=s2g; g2s=np.full(int(s2g.max())+1,-1,int)
        for a in range(8):
            if s2g[a]>=0: g2s[s2g[a]]=a
        green2std[tid]=g2s; ts_turn[tid]={int(i):(c[0]["approach"],c[0]["turn"]) for i,c in t["links"].items()}
    env.reset(seed)
    agent=None
    if ckpt:
        for tid in env.ts_ids:
            ts=env.traffic_signals[tid]; ts.std_action_map=green2std[tid]
            ts.observation_fn.rebind_movements(ts_turn[tid]); ts.observation_space=ts.observation_fn.observation_space()
        states={tid:env.traffic_signals[tid].observation_fn() for tid in env.ts_ids}
        od=len(next(iter(states.values())))
        agent=DQN(starting_state=tuple([0.]*od),state_space=od,hidden_dim=128,action_space=8,learning_rate=1e-3,
            gamma=0.95,epsilon=0.0,target_update=10,capacity=1,mini_size=1,batch_size=1,eps_start=0,eps_end=0,eps_decay=1,device="cpu")
        agent.q_net.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=False)["policy_state_dict"]); agent.q_net.eval()
    sumo=env.traffic_signals[env.ts_ids[0]].sumo
    # 受控进车道对应的 edge 集合 (只统计这些; 出车道由下游路口的进车道代表)
    ctrl_edges=set()
    for tid in env.ts_ids:
        for ln in env.traffic_signals[tid].signal_controlled_lanes:
            ctrl_edges.add(ln.rsplit("_",1)[0])
    # per (edge,class): sum local delta-wait, and vehicle-set (denominator)
    edge_wait={}   # (edge,cls)->float
    edge_veh={}    # (edge,cls)->set(vid)
    edge_spd={}    # (edge,cls)->[sum_speed, n_steps]  时间加权平均速度
    edge_stop={}   # (edge,cls)->总停车事件数 (move->stop 转变)
    last_acc={}    # vid->last accumulated wait
    last_edge={}   # vid->last edge
    last_moving={} # vid->上一步是否在动 (speed>=0.1)
    done={"__all__":False}
    while not done["__all__"]:
        if fixed: acts={}
        else:
            acts={tid:int(std2green[tid][int(np.where(ts_mask[tid],
                  agent.q_net(torch.tensor(np.asarray(states[tid],np.float32)).unsqueeze(0)).squeeze(0).detach().numpy(),-np.inf).argmax())]) for tid in env.ts_ids}
        for vid in sumo.vehicle.getIDList():
            edge=sumo.vehicle.getRoadID(vid)
            if edge.startswith(":"):   # 跳过路口内部边; 其余所有边(进车道+出车道+非受控)都记
                last_acc[vid]=sumo.vehicle.getAccumulatedWaitingTime(vid); last_edge[vid]=edge
                continue
            acc=sumo.vehicle.getAccumulatedWaitingTime(vid)
            cls=vclass(sumo.vehicle.getTypeID(vid))
            dprev=last_acc.get(vid)
            if dprev is not None and last_edge.get(vid)==edge:
                inc=max(0.0, acc-dprev)
                edge_wait[(edge,cls)]=edge_wait.get((edge,cls),0.0)+inc   # 0 也累加 → 该边即便全程不等也被记录
            else:
                edge_wait.setdefault((edge,cls),0.0)    # 首次进入该边: 建条目(等待0), 保证被画
            edge_veh.setdefault((edge,cls),set()).add(vid)
            spd_now=sumo.vehicle.getSpeed(vid)
            sp=edge_spd.setdefault((edge,cls),[0.0,0]); sp[0]+=spd_now; sp[1]+=1
            edge_stop.setdefault((edge,cls),0)
            moving=spd_now>=0.1
            if last_moving.get(vid,True) and not moving and last_edge.get(vid)==edge:  # 在本边由动转停
                edge_stop[(edge,cls)]+=1
            last_moving[vid]=moving
            last_acc[vid]=acc; last_edge[vid]=edge
        out=env.step(acts); done=out[2]
        if not fixed: states=out[0]
    res={}
    for (edge,cls),tot in edge_wait.items():
        n=len(edge_veh.get((edge,cls),[]))
        if n>0: res[f"{edge}|{cls}"]=tot/n
    spd={f"{e}|{c}": sp[0]/sp[1] for (e,c),sp in edge_spd.items() if sp[1]>0}
    stp={f"{e}|{c}": edge_stop[(e,c)]/len(edge_veh[(e,c)]) for (e,c) in edge_stop if edge_veh.get((e,c))}  # 每车平均停车次数
    env.close(); return res, spd, stp

def collect(fixed, ckpt, tag):
    waccs,saccs,taccs=[],[],[]
    for sd in SEEDS:
        print(f"[{tag}] seed {sd}...",flush=True); w,s,t=collect_one(fixed,ckpt,sd); waccs.append(w); saccs.append(s); taccs.append(t)
    def avg(accs):
        keys=set().union(*[set(a) for a in accs])
        return {k:float(np.mean([a[k] for a in accs if k in a])) for k in keys}
    print(f"[{tag}] done",flush=True); return avg(waccs), avg(saccs), avg(taccs)

ck=sorted(glob.glob("models/exp208_*/*/best.pth"))[-1]
fw,fs,ft=collect(True,None,"fixed"); pw,ps,pt=collect(False,ck,"exp208")
A="/home/xiaowen/sumo-rl/experiments/analysis/"
json.dump({"fixed":fw,"prio":pw},open(A+"edge_waiting_data.json","w"))
json.dump({"fixed":fs,"prio":ps},open(A+"edge_speed_data.json","w"))
json.dump({"fixed":ft,"prio":pt},open(A+"edge_stopevents_data.json","w"))
print("SAVED waiting + speed + stopevents data")
