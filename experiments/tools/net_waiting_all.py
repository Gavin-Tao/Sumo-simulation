"""One-shot: collect per-intersection per-class avg_stopped_time (fixed + exp208)
AND plot the Dublin network map. No /tmp dependency — data kept in memory."""
import os, sys, json, functools, glob
import numpy as np
os.environ['SUMO_RL_LIBSUMO']='1'; sys.path.insert(0,'/home/xiaowen/sumo-rl'); sys.path.insert(0,'/home/xiaowen/sumo-rl/experiments')
import yaml, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sumolib
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.agents.dqn_agent_txw import DQN

CFG="configs/exp208_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095.yaml"
cfg=yaml.safe_load(open(CFG)); SEEDS=[2000,2001,2002]; METRIC="avg_stopped_time"
CLS=[("car","o"),("bus","s"),("ambulance","^")]

def collect_one(fixed, ckpt, seed):
    kw=dict(net_file=cfg['net_file'],route_file=cfg['route_file'],cfg_file=cfg['cfg_file'],
        out_csv_name=None,use_gui=False,num_seconds=3600,delta_time=5,yellow_time=2,min_green=5,
        max_green=50,use_max_green=True,single_agent=False,sumo_seed=seed,fixed_ts=fixed)
    if not fixed:   # 只有 RL 臂需要 priority obs/reward; fixed-time 用默认即可
        kw["reward_fn"]=make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
        kw["observation_class"]=functools.partial(obsmod.PriorityMovementLegacyCQM,
            priority_source=cfg["priority_source"],include_downstream=True,downstream_fields=("count","queue"),
            awt_basis=cfg.get("obs_awt_basis","global"),awt_cap=cfg.get("obs_awt_cap"),
            include_lane_occ=cfg.get("obs_lane_occ",False))
    env=SumoEnvironment(**kw)
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
    mc=EpisodeMetricsCollector({tid:env.traffic_signals[tid].signal_controlled_lanes for tid in env.ts_ids},
        delta_time=5,excluded_lanes=set().union(*(getattr(env.traffic_signals[t],'always_green_lanes',set()) for t in env.ts_ids)))
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
    done={"__all__":False}; st=0
    while not done["__all__"]:
        if st>=20: mc.collect_step(env.sumo)
        if fixed: acts={}
        else:
            acts={tid:int(std2green[tid][int(np.where(ts_mask[tid],
                  agent.q_net(torch.tensor(np.asarray(states[tid],np.float32)).unsqueeze(0)).squeeze(0).detach().numpy(),-np.inf).argmax())]) for tid in env.ts_ids}
        states,_,done,_=env.step(acts); st+=1
    mc.collect_step(env.sumo); mc.finalize(env.sumo); s=mc.summary()
    out={tid:{c:s.get(tid,{}).get(c,{}).get(METRIC) for c,_ in CLS} for tid in env.ts_ids}
    env.close(); return out

def collect(fixed, ckpt, tag):
    accs=[]
    for sd in SEEDS:
        print(f"[{tag}] seed {sd} 仿真中...", flush=True)
        accs.append(collect_one(fixed, ckpt, sd))
    tids=accs[0].keys()
    out={tid:{c:float(np.nanmean([a[tid][c] for a in accs if a[tid].get(c) is not None]))
              if any(a[tid].get(c) is not None for a in accs) else None for c,_ in CLS} for tid in tids}
    print(f"[{tag}] done {len(out)}路口 (3seed均值)",flush=True); return out

ck=sorted(glob.glob("models/exp208_*/*/best.pth"))[-1]
fixed=collect(True,None,"fixed"); prio=collect(False,ck,"exp208")
json.dump({"fixed":fixed,"prio":prio},open("/home/xiaowen/sumo-rl/experiments/analysis/perts_waiting_data.json","w"))

# ---- plot ----
print("[plot] 画图中...",flush=True)
net=sumolib.net.readNet("/home/xiaowen/sumo-rl/nets/dublin/dublin_8std.net.xml")
coord={}
for t in net.getTrafficLights():
    try: coord[t.getID()]=net.getNode(t.getID()).getCoord()
    except Exception: pass
allv=[v for d in (fixed,prio) for ts in d.values() for v in ts.values() if v is not None]
vmin,vmax=min(allv),float(np.percentile(allv,95)); cmap=plt.cm.RdYlGn_r
plt.rcParams.update({"font.size":13,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"pdf.fonttype":42})
fig,axes=plt.subplots(1,2,figsize=(17,8.2))
OFF={"car":(-95,0),"bus":(0,80),"ambulance":(95,0)}
for ax,(data,title) in zip(axes,[(fixed,"Fixed-time"),(prio,"Priority-RL (exp208, 5:3:1)")]):
    for e in net.getEdges():
        s=e.getShape(); ax.plot([p[0] for p in s],[p[1] for p in s],color="#DDDDDD",lw=0.6,zorder=1)
    for tid,(x,y) in coord.items():
        for c,mk in CLS:
            v=data.get(tid,{}).get(c)
            if v is None: continue
            dx,dy=OFF[c]
            ax.scatter(x+dx,y+dy,s=150,marker=mk,c=[cmap((min(v,vmax)-vmin)/(vmax-vmin+1e-9))],edgecolors="black",linewidths=0.8,zorder=3)
    ax.set_title(title,fontsize=15); ax.set_aspect("equal"); ax.axis("off")
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin,vmax)); sm.set_array([])
cb=fig.colorbar(sm,ax=axes,fraction=0.025,pad=0.02)
cb.set_label("Avg stopped time per intersection (s)  [green=low, red=high]",fontsize=12)
leg=[Line2D([0],[0],marker=m,color="w",markerfacecolor="gray",markeredgecolor="k",markersize=12,label=c.capitalize()) for c,m in CLS]
axes[0].legend(handles=leg,loc="upper left",frameon=True,fontsize=12,title="Vehicle class")
fig.suptitle("Dublin 11h network: per-intersection waiting by vehicle class",fontsize=16,y=0.98)
out="/home/xiaowen/sumo-rl/experiments/analysis/fig_network_waiting"
fig.savefig(out+".png",dpi=200,bbox_inches="tight",pad_inches=0.1); fig.savefig(out+".pdf",bbox_inches="tight",pad_inches=0.1)
plt.close(fig); print(f"SAVED {out}.png (色标 {vmin:.1f}~{vmax:.1f}s)",flush=True)
