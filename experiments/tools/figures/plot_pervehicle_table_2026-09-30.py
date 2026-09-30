"""新表 (2026-09-30, 用户批准; 不改动旧表/旧代码): 逐车配对指标 8STD vs GS-ENUM。
数据: experiments/analysis/data/pervehicle/exp<exp>_ep<ep>_seed123.json (tools/dublin/eval_pervehicle.py, 各臂末 4 个检查点, 评估种子 123)
配对单位 = 同一辆车 (同种子同路由) 在两臂第 k 个检查点下 (k = 1..4)。
行: A 逐车时间损失 (SUMO timeLoss, 1 s 分辨率) 分车类: 两臂均值, 配对差 GS−8STD 及 95% CI (bootstrap), GS 更优比例 (剔平局), Wilcoxon p;
    J(531) 时间损失 (份额 × 权重 × 类均值, 按检查点), J_eq;
    B 相对时间损失 = timeLoss / (duration − timeLoss) (相对自由流行程, 跨路线可比);
    C 同口同 movement 分层的优先级核验 (每臂内): 救护车/小汽车、公交/小汽车 的进口道停车秒比 (按高优先级类访问量加权, 只用两类都出现的层);
    D 救护车事件指标: 到达绿灯率, 无停车通过率, 服务延迟中位 (红灯到达 → 首次绿), 单次最大停车;
    E 稳健性: 完成率, 锁死次数 (4 个检查点里任一路口进口道均停车 > 100 s 的次数)。
输出: experiments/analysis/figures/main_comparison_pervehicle_2026-09-30_en.png (+ 同名 .json 数值)
"""
import glob, json, os, re, sys
import numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"]=["DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
D="experiments/analysis/data/pervehicle"
SCEN=[("1x1",("274","263"),{"car":.965,"bus":.033,"ambulance":.0017}),("1x3",("275","265"),{"car":.950,"bus":.050,"ambulance":.0004}),
      ("Dublin 11:00",("208","211"),{"car":.9507,"bus":.0485,"ambulance":.0008}),("Dublin 02:00",("283","284"),{"car":.9879,"bus":.0109,"ambulance":.0012}),
      ("Dublin 02:00, ambulance probe route",("287","288"),{"car":.9879,"bus":.0109,"ambulance":.0012}),("Dublin 18:00",("285","286"),{"car":.9568,"bus":.0430,"ambulance":.0002}),
      ("Dublin 18:00, ambulance probe route",("289","290"),{"car":.9568,"bus":.0430,"ambulance":.0002})]
WT={"car":1,"bus":3,"ambulance":5}; CLS=("car","bus","ambulance","all")
def load(exp):
    fs=sorted(glob.glob(f"{D}/exp{exp}_ep*_seed123.json"))[-4:]
    return [json.load(open(f)) for f in fs]
def tl(v): return v["trip"]["timeLoss"] if v.get("trip") else None
def rel(v):
    t=v.get("trip"); 
    if not t or t["duration"]-t["timeLoss"]<=0: return None
    return t["timeLoss"]/(t["duration"]-t["timeLoss"])
def boot_ci(d, n=2000, seed=0):
    r=np.random.default_rng(seed); m=[r.choice(d, len(d)).mean() for _ in range(n)]; return np.percentile(m,2.5), np.percentile(m,97.5)
def analyse(name, pair, share):
    A,B=load(pair[0]),load(pair[1])
    if not A or not B: return None
    K=min(len(A),len(B)); A,B=A[-K:],B[-K:]
    out={"name":name,"exps":pair,"K":K,"eps":[(a["ckpt_episode"],b["ckpt_episode"]) for a,b in zip(A,B)],"rows":{}}
    # A: 逐车配对 timeLoss
    pairs={c:[] for c in CLS}; relv={c:([],[]) for c in CLS}
    for a,b in zip(A,B):
        for vid,va in a["vehicles"].items():
            vb=b["vehicles"].get(vid); 
            if not vb: continue
            x,y=tl(va),tl(vb)
            if x is None or y is None: continue
            for c in (va["type"],"all"):
                if c in pairs: pairs[c].append((x,y))
                ra,rb=rel(va),rel(vb)
                if ra is not None and rb is not None and c in relv: relv[c][0].append(ra); relv[c][1].append(rb)
    for c in CLS:
        P=np.array(pairs[c]); 
        if len(P)==0: continue
        d=P[:,1]-P[:,0]; nz=d[d!=0]; p=wilcoxon(nz).pvalue if len(nz)>=10 else float("nan")
        lo,hi=boot_ci(d) if len(d)>=2 else (np.nan,np.nan)
        out["rows"][f"tl_{c}"]=dict(n=len(P), std=P[:,0].mean(), gs=P[:,1].mean(), diff=d.mean(), ci=(lo,hi), win=(d<0).sum()/max((d!=0).sum(),1), ties=(d==0).sum(), p=p, med_std=np.median(P[:,0]), med_gs=np.median(P[:,1]))
        out["rows"][f"rel_{c}"]=dict(std=np.mean(relv[c][0]), gs=np.mean(relv[c][1]))
    # J(531) 按检查点
    J=[[sum(share[c]*WT[c]*np.mean([tl(v) for v in r["vehicles"].values() if v["type"]==c and tl(v) is not None] or [0]) for c in share) for r in R] for R in (A,B)]
    E=[[sum(WT[c]*np.mean([tl(v) for v in r["vehicles"].values() if v["type"]==c and tl(v) is not None] or [0]) for c in share) for r in R] for R in (A,B)]
    out["rows"]["J"]=dict(std=np.mean(J[0]), std_sd=np.std(J[0]), gs=np.mean(J[1]), gs_sd=np.std(J[1]), win=sum(1 for x,y in zip(*J) if y<x), K=K)
    out["rows"]["Jeq"]=dict(std=np.mean(E[0]), gs=np.mean(E[1]), win=sum(1 for x,y in zip(*E) if y<x), K=K)
    # C: 同口同 movement 分层 (进口道停车秒 / 次访问)
    for arm,R in (("std",A),("gs",B)):
        strata={}
        for r in R:
            for vid,v in r["vehicles"].items():
                for t,recs in v["tls"].items():
                    for x in recs:
                        if x["link"] is None: continue
                        strata.setdefault((t,x["link"]),{}).setdefault(v["type"],[]).append(x["stopped"])
        for hi_c in ("ambulance","bus"):
            num=den=0.0; nstr=0
            for s,d in strata.items():
                if hi_c in d and "car" in d and len(d["car"])>=5:
                    w=len(d[hi_c]); num+=w*np.mean(d[hi_c]); den+=w*np.mean(d["car"]); nstr+=1
            out["rows"][f"ratio_{hi_c}_{arm}"]=dict(ratio=(num/den if den>0 else np.nan), n_strata=nstr, hi_mean=(num/sum(len(d[hi_c]) for s,d in strata.items() if hi_c in d and "car" in d and len(d["car"])>=5) if nstr else np.nan))
    # D: 救护车事件
    for arm,R in (("std",A),("gs",B)):
        recs=[x for r in R for v in r["vehicles"].values() if v["type"]=="ambulance" for recs in v["tls"].values() for x in recs if x["link"] is not None]
        if recs:
            ga=np.mean([x["entry_state"] in ("G","g") for x in recs]); ns=np.mean([x["stopped"]==0 for x in recs])
            lat=[x["first_green"]-x["entry"] for x in recs if x["entry_state"] not in ("G","g") and x["first_green"] is not None]
            out["rows"][f"amb_{arm}"]=dict(n_visits=len(recs), green_on_arrival=ga, no_stop=ns, latency_med=(np.median(lat) if lat else np.nan), max_stop=max(x["stopped"] for x in recs), n_amb=sum(1 for r in R for v in r["vehicles"].values() if v["type"]=="ambulance"))
    # E: 完成率 与 锁死
    for arm,R in (("std",A),("gs",B)):
        comp=np.mean([np.mean([1.0 if (v.get("trip") and "arrival" in v["trip"] and v["trip"]["arrival"]>=0) else 0.0 for v in r["vehicles"].values()]) for r in R])
        grid=0
        for r in R:
            per={}
            for v in r["vehicles"].values():
                for t,recs in v["tls"].items():
                    for x in recs: per.setdefault(t,[]).append(x["stopped"])
            if per and max(np.mean(x) for x in per.values())>100: grid+=1
        out["rows"][f"robust_{arm}"]=dict(completion=comp, gridlock=grid, K=K)
    return out
def fmt(x,d=2): return "n/a" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.{d}f}"
def render(results, out_png):
    GREEN="#cfe9cf"; cells=[]; colors=[]; hdr=[]; titles=[]
    for R in results:
        hdr.append(len(cells)+1); titles.append(f"{R['name']}  (8STD: exp{R['exps'][0]}, GS-ENUM: exp{R['exps'][1]}; last {R['K']} checkpoints, seed 123)"); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)
        rows=R["rows"]
        for c in CLS:
            r=rows.get(f"tl_{c}"); 
            if not r: continue
            col=["#fafafa"]*5; col[2 if r["gs"]<r["std"] else 1]=GREEN
            cells.append([f"Time loss per vehicle (s) - {c}  [n = {r['n']}]", fmt(r["std"]), fmt(r["gs"]), f"{r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]", f"{100*r['win']:.0f}% ({r['ties']}), p = {r['p']:.2g}" if not np.isnan(r["p"]) else f"{100*r['win']:.0f}% ({r['ties']})"]); colors.append(col)
        r=rows["J"]; col=["#eef3ff"]*5; col[2 if r["gs"]<r["std"] else 1]=GREEN
        cells.append(["J(531) time loss (per checkpoint mean ± s.d.)", f"{r['std']:.2f} ± {r['std_sd']:.2f}", f"{r['gs']:.2f} ± {r['gs_sd']:.2f}", f"{100*(r['gs']/r['std']-1):+.0f}%", f"GS wins {r['win']} / {r['K']}"]); colors.append(col)
        for c in CLS:
            r=rows.get(f"rel_{c}"); 
            if not r: continue
            col=["#fafafa"]*5; col[2 if r["gs"]<r["std"] else 1]=GREEN
            cells.append([f"Relative time loss (time loss / free-flow travel time) - {c}", fmt(r["std"],3), fmt(r["gs"],3), f"{100*(r['gs']/r['std']-1):+.0f}%", ""]); colors.append(col)
        for hi_c,lab in (("ambulance","ambulance / car"),("bus","bus / car")):
            a,b=rows.get(f"ratio_{hi_c}_std"),rows.get(f"ratio_{hi_c}_gs")
            if a and b:
                col=["#fff8e6"]*5; 
                if not (np.isnan(a["ratio"]) or np.isnan(b["ratio"])): col[2 if b["ratio"]<a["ratio"] else 1]=GREEN
                note = "n/a" if (np.isnan(a["ratio"]) or np.isnan(b["ratio"])) else ("both < 1: priority honoured" if (a["ratio"]<1 and b["ratio"]<1) else ("8STD < 1 only" if a["ratio"]<1 else ("GS < 1 only" if b["ratio"]<1 else "both > 1: not honoured")))
                cells.append([f"Same-junction/movement stop ratio, {lab}  [strata {a['n_strata']}/{b['n_strata']}]", fmt(a["ratio"]), fmt(b["ratio"]), note, ""]); colors.append(col)
        a,b=rows.get("amb_std"),rows.get("amb_gs")
        if a and b:
            for key,lab,bigger in (("green_on_arrival","Ambulance green-on-arrival rate",True),("no_stop","Ambulance no-stop pass rate",True),("latency_med","Ambulance service latency, median (s)",False),("max_stop","Ambulance longest single stop (s)",False)):
                col=["#f3f3f3"]*5; x,y=a[key],b[key]
                if not (np.isnan(x) or np.isnan(y)) and x!=y: col[2 if ((y>x) if bigger else (y<x)) else 1]=GREEN
                cells.append([f"{lab}  [visits {a['n_visits']}/{b['n_visits']}]", fmt(x,2), fmt(y,2), "", ""]); colors.append(col)
        a,b=rows["robust_std"],rows["robust_gs"]
        cells.append(["Completion rate (arrived before episode end)", fmt(a["completion"],3), fmt(b["completion"],3), "", ""]); colors.append(["#ffffff"]*5)
        cells.append([f"Gridlock checkpoints (any junction mean stop > 100 s) / {a['K']}", str(a["gridlock"]), str(b["gridlock"]), "", ""]); colors.append(["#ffffff"]*5)
    fig,ax=plt.subplots(figsize=(16.5,0.3*len(cells)+2)); ax.axis("off")
    tbl=ax.table(cellText=cells,colLabels=["Metric","8STD","GS-ENUM","GS − 8STD [95% CI] / rel. diff","GS better (ties), Wilcoxon p"],cellColours=colors,colColours=["#c9d3df"]*5,loc="upper center",cellLoc="center",colWidths=[0.40,0.11,0.11,0.19,0.19])
    tbl.auto_set_font_size(False); tbl.scale(1,1.45)
    for (r,c),cell in tbl.get_celld().items():
        t=cell.get_text(); t.set_fontsize(10)
        if r==0: t.set_weight("bold"); t.set_fontsize(10.5)
        if c==0 and r>0: t.set_ha("left"); cell.PAD=0.012
        if r in hdr and c>0: cell.set_edgecolor("#dde3ea")
    ax.set_title("8STD vs GS-ENUM: per-vehicle paired metrics (same vehicles under both controllers; SUMO time loss, 1 s resolution)",fontsize=12.5,pad=6)
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); inv=fig.transFigure.inverted()
    for r,title in zip(hdr,titles):
        bb=tbl[r,0].get_window_extent(ren); x0,y0=inv.transform((bb.x0,bb.y0)); x1,y1=inv.transform((bb.x1,bb.y1))
        fig.text(x0+0.004,(y0+y1)/2,title,ha="left",va="center",fontsize=10,weight="bold",zorder=10)
    from matplotlib.transforms import Bbox
    tb=tbl.get_window_extent(ren); tt=ax.title.get_window_extent(ren); bb=Bbox.union([tb,tt]); pad=0.15*fig.dpi
    crop=Bbox.from_extents((bb.x0-pad)/fig.dpi,(bb.y0-pad)/fig.dpi,(bb.x1+pad)/fig.dpi,(bb.y1+pad)/fig.dpi)
    plt.savefig(out_png,dpi=200,bbox_inches=crop); print("saved",out_png)
def main():
    results=[r for r in (analyse(n,p,s) for n,p,s in SCEN) if r]
    json.dump(results, open("experiments/analysis/figures/main_comparison_pervehicle_2026-09-30_en.json","w"), default=float, indent=1)
    render(results, "experiments/analysis/figures/main_comparison_pervehicle_2026-09-30_en.png")
if __name__=="__main__": main()
