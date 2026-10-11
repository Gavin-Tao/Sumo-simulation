"""(2026-10-11) 临时路网版本见 scratchpad; 本副本只为复现表格: 读取 analysis/data/pervehicle 下 *_ckpt_coll 与 *_demoted4_coll 文件。
"""301 ep800 vs 311 ep150: 原网 (18 信号) vs 4 路口降级网 (14 信号, 零样本) 完整对比表。口径: (a) 收集器官方汇总 (该网全部 RL 路口); (b) 限定共同 14 路口重算 (逐车记录);
(c) 整趟 timeLoss/车; (d) RL 路口延误/visit 与 全网 timeLoss/车 (rldelay 文件)。每格 mean ± sd (种子), 另给 311 对 301 的配对胜场。"""
import glob, json, numpy as np, collections
PV="analysis/data/pervehicle"; CL=("car","bus","ambulance","all")
ARMS={("301","orig"):("exp301_ep00800_seed*_ckpt_coll.json","_ckpt_coll","_rldelay"),("311","orig"):("exp311_ep00150_seed*_ckpt_coll.json","_ckpt_coll","_rldelay"),
      ("301","demoted4"):("exp301_ep00800_seed*_demoted4_coll.json","_demoted4_coll","_demoted4_rldelay"),("311","demoted4"):("exp311_ep00150_seed*_demoted4_coll.json","_demoted4_coll","_demoted4_rldelay")}
FOUR={"659802","133307283","32236377","135109534"}
def per_seed(f, rl_suffix, coll_suffix):
    d=json.load(open(f)); cv=d["collector"]["vehicles"]; sm=d["collector"]["summary"]; M={}
    for c in ("car","bus","ambulance"):
        M[f"off_t_{c}"]=sm.get(f"eval/system/{c}/avg_stopped_time_per_visit",np.nan); M[f"off_e_{c}"]=sm.get(f"eval/system/{c}/avg_stop_events_per_visit",np.nan)
    st=collections.Counter(); ev=collections.Counter(); vis=collections.Counter(); tl=collections.defaultdict(list)
    for vid,rec in d["vehicles"].items():
        c=rec["type"]; J=[t for t in rec["tls"] if t not in FOUR]; vis[c]+=len(J); vis["all"]+=len(J)
        s=cv.get(vid,{}).get("stopped_sec",{}); e=cv.get(vid,{}).get("stop_events",{})
        st[c]+=sum(s.get(t,0) for t in J); ev[c]+=sum(e.get(t,0) for t in J); st["all"]+=sum(s.get(t,0) for t in J); ev["all"]+=sum(e.get(t,0) for t in J)
        if rec.get("trip"): tl[c].append(float(rec["trip"]["timeLoss"])); tl["all"].append(float(rec["trip"]["timeLoss"]))
    for c in CL:
        M[f"r14_t_{c}"]=st[c]/vis[c] if vis[c] else np.nan; M[f"r14_e_{c}"]=ev[c]/vis[c] if vis[c] else np.nan; M[f"trip_{c}"]=float(np.mean(tl[c])) if tl[c] else np.nan
    rf=f.replace(coll_suffix+".json", rl_suffix+".json")
    try:
        r=json.load(open(rf))["classes"]
        for c in ("car","bus","ambulance"): M[f"rld_{c}"]=r[c]["rl_per_visit"]; M[f"net_{c}"]=r[c]["net_per_vehicle"]
    except Exception: pass
    return M
data={}
for k,(pat,cs,rs) in ARMS.items():
    data[k]={int(f.split("_seed")[1].split("_")[0]): per_seed(f,rs,cs) for f in sorted(glob.glob(f"{PV}/{pat}"))}
rows=[("官方汇总 停车时间/visit (该网全部 RL 路口)", "off_t", ("car","bus","ambulance")), ("官方汇总 停车次数/visit", "off_e", ("car","bus","ambulance")),
      ("限定共同 14 路口 停车时间/visit", "r14_t", CL), ("限定共同 14 路口 停车次数/visit", "r14_e", CL), ("整趟 timeLoss/车 (s)", "trip", CL),
      ("RL 路口延误/visit (s, 该网 RL 路口)", "rld", ("car","bus","ambulance")), ("全网 timeLoss/车 (s, rldelay 文件)", "net", ("car","bus","ambulance"))]
cols=[("301","orig"),("311","orig"),("301","demoted4"),("311","demoted4")]
print(f"n 种子: " + ", ".join(f"{a} {n}={len(data[(a,n)])}" for a,n in cols))
hdr=f"{'指标':44s} | " + " | ".join(f"{a} {('原网18' if n=='orig' else '降级14')}".ljust(16) for a,n in cols) + " | 311胜301 原网 / 降级网"
print(hdr); print("-"*len(hdr))
for title,key,classes in rows:
    for c in classes:
        cells=[]; vals={}
        for a,n in cols:
            v=np.array([m.get(f"{key}_{c}",np.nan) for m in data[(a,n)].values()],float); vals[(a,n)]=data[(a,n)]
            cells.append(f"{np.nanmean(v):6.2f} ± {np.nanstd(v):4.2f}" if len(v) and not np.all(np.isnan(v)) else "   -   ")
        wins=[]
        for n in ("orig","demoted4"):
            s=sorted(set(data[("301",n)])&set(data[("311",n)])); w=sum(1 for sd in s if data[("311",n)][sd].get(f"{key}_{c}",np.nan)<data[("301",n)][sd].get(f"{key}_{c}",np.nan)); wins.append(f"{w}/{len(s)}" if s else "-")
        lab=(title if c==classes[0] else ""); print(lab.ljust(44) + "| " + " | ".join(x.ljust(16) for x in cells) + f" | {wins[0]} / {wins[1]}   [{c}]")
