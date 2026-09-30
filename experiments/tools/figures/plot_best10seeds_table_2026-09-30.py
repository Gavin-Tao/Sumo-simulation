"""通用版 (2026-09-30): 任一时段探针对的 best.pth × 10 评估种子表 (02h 版脚本的参数化拷贝, 02h 脚本保留不动)。
用法: python plot_best10seeds_table_2026-09-30.py --window 18h --exps 289 290 [--anom 100]
网格场景 (2026-09-30 用户令追加): --window 1x1 --exps 274 263 --anom 50 --tukey / --window 1x3 --exps 275 265 --anom 50 --tukey
  网格无探针路由 → 不出换车型块; 停车次数门按 amb ≤ bus ≤ car 全判 (与主表 1x1/1x3 块一致); J 份额同主表 (1x1 .965/.033/.0017, 1x3 .950/.050/.0004);
  --tukey: 追加一块 "对称剔除统计离群种子" (任一臂全车类停车时间/visit 落在该臂 10 种子 Tukey 1.5×IQR 围栏外的种子, 两臂同剔); 02h/18h 默认行为与输出不变。
原说明: (用户令 2026-09-30), 口径与旧表 (倒数第二块) 相同:
收集器 5 s 采样的 per-visit 停车时间 / 停车次数, 分车类均值 ± sd (对 10 次评估), 逐指标 J(531) (份额 car .9879 / bus .0109 / amb .0012),
配对胜 (同种子), 保序门 (严格 / 容差 5 s)。块: (1) 全部 10 种子; (2) 对称剔除异常种子 (任一臂任一路口全车类停车 > 50 s);
(3) 同路线同时刻换车型: 救护车 vs 公交 (busprobe 路由), 10 种子配对。
数据: experiments/analysis/data/pervehicle/exp28{7,8}_ep*_seed*_best_coll.json / _best_busprobe_coll.json
输出: experiments/analysis/figures/main_comparison_02h_best10seeds_2026-09-30_en.{png,json} (新文件, 不覆盖旧图)
"""
import glob, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"]=["DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--window", default="18h"); ap.add_argument("--seeds", nargs="*", type=int, default=None, help="只用这些评估种子 (2026-09-30 加, 用户自用的筛选表; 默认全部)"); ap.add_argument("--label", default="", help="标题里的场景标签 (默认按 --window; 2026-09-30 加, 用于标注改道版场景)"); ap.add_argument("--out-suffix", default="", help="输出文件名附加后缀 (2026-09-30: 避免不同实验对覆盖同名图, 默认空 = 原文件名)"); ap.add_argument("--exps", nargs=2, default=["289","290"]); ap.add_argument("--anom", type=float, default=100.0, help="异常评估规则: 任一臂任一路口全车类停车 > 阈值 (s)"); ap.add_argument("--tag", default="best", help="数据文件标签前缀 (best → *_best_coll.json / *_best_busprobe_coll.json)"); ap.add_argument("--tukey", action="store_true", help="追加块: 对称剔除 Tukey 1.5×IQR 离群种子 (全车类停车时间/visit, 任一臂)")
ARGS=ap.parse_args()
D="experiments/analysis/data/pervehicle"; WT={"car":1,"bus":3,"ambulance":5}
SH={"02h":{"car":.9879,"bus":.0109,"ambulance":.0012},"11h":{"car":.9507,"bus":.0485,"ambulance":.0008},"18h":{"car":.9568,"bus":.0430,"ambulance":.0002},"1x1":{"car":.965,"bus":.033,"ambulance":.0017},"1x3":{"car":.950,"bus":.050,"ambulance":.0004}}[ARGS.window]
WLABEL={"02h":"Dublin 02:00","11h":"Dublin 11:00","18h":"Dublin 18:00","1x1":"1x1","1x3":"1x3"}[ARGS.window]; GRID=ARGS.window in ("1x1","1x3")
if ARGS.label: WLABEL=ARGS.label
AMB_IDS=("amb_1_early","amb_1") if ARGS.window=="02h" else ("amb_1",); BUS_IDS=("probe_bus_1_early","probe_bus_1") if ARGS.window=="02h" else ("probe_bus_1",)
def load(e, tag):
    out={}
    for f in glob.glob(f"{D}/exp{e}_ep*_seed*_{tag}.json"):
        seed=int(f.split("_seed")[1].split("_")[0]); out[seed]=json.load(open(f))
    return out
def gate(am,bu,ca,tol=5.0):
    if am<=bu<=ca: return "pass"
    return "pass (tol. 5 s)" if (am-bu<=tol and bu-ca<=tol) else "fail"
def metric_rows(A,B,seeds,m,label,jlabel):
    rows=[]; vals={}
    for c in ("car","bus","ambulance","all"):
        a=np.array([A[s]["collector"]["summary"].get(f"eval/system/{c}/{m}",np.nan) for s in seeds]); b=np.array([B[s]["collector"]["summary"].get(f"eval/system/{c}/{m}",np.nan) for s in seeds])
        vals[c]=(a,b); ok=~np.isnan(a)&~np.isnan(b)
        rows.append((f"{label} - {c}", np.nanmean(a), np.nanstd(a), np.nanmean(b), np.nanstd(b), f"{100*(np.nanmean(b)/np.nanmean(a)-1):+.0f}%", f"{int((b<a)[ok].sum())}/{int(ok.sum())} (ties {int((b==a)[ok].sum())})"))
    Ja=sum(SH[c]*WT[c]*np.nan_to_num(vals[c][0]) for c in SH); Jb=sum(SH[c]*WT[c]*np.nan_to_num(vals[c][1]) for c in SH)
    rows.append((jlabel, Ja.mean(), Ja.std(), Jb.mean(), Jb.std(), f"{100*(Jb.mean()/Ja.mean()-1):+.0f}%", f"{int((Jb<Ja).sum())}/{len(seeds)}"))
    g=[gate(np.nanmean(vals["ambulance"][i]),np.nanmean(vals["bus"][i]),np.nanmean(vals["car"][i])) for i in (0,1)]
    return rows, g
def anomalous(A,B,seeds,thr=None):
    thr=ARGS.anom if thr is None else thr
    bad=[]
    for s in seeds:
        for R in (A[s],B[s]):
            if any(v>thr for k,v in R["collector"]["summary"].items() if k.endswith("/all/avg_stopped_time") and not k.startswith("eval/system")):
                bad.append(s); break
    return sorted(set(bad))
def tukey_outliers(A,B,seeds,m="avg_stopped_time_per_visit"):
    bad=[]; fences={}
    for arm,X in (("8STD",A),("GS",B)):
        v=np.array([X[s]["collector"]["summary"][f"eval/system/all/{m}"] for s in seeds]); q1,q3=np.percentile(v,[25,75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        fences[arm]=(float(lo),float(hi)); bad+=[s for s,x in zip(seeds,v) if x<lo or x>hi]
    return sorted(set(bad)), fences
def typeswap(A,Ab,B,Bb,seeds):
    out={}
    for arm,(X,Xb) in (("8STD",(A,Ab)),("GS",(B,Bb))):
        def pv(v,field): 
            vals=[sum(r[field].values())/r["k_v"] for r in v if r["k_v"]>0]; return np.mean(vals) if vals else np.nan
        amb_pv=[]; bus_pv=[]; amb_ev=[]; bus_ev=[]
        for s in seeds:
            va=[X[s]["collector"]["vehicles"][k] for k in AMB_IDS if k in X[s]["collector"]["vehicles"]]
            vb=[Xb[s]["collector"]["vehicles"][k] for k in BUS_IDS if k in Xb[s]["collector"]["vehicles"]]
            amb_pv.append(pv(va,"stopped_sec")); bus_pv.append(pv(vb,"stopped_sec")); amb_ev.append(pv(va,"stop_events")); bus_ev.append(pv(vb,"stop_events"))
        out[arm]={"pv":(amb_pv,bus_pv),"ev":(amb_ev,bus_ev)}
    return out
def main():
    E1,E2=ARGS.exps; T=ARGS.tag; A,B=load(E1,f"{T}_coll"),load(E2,f"{T}_coll")
    Ab,Bb=(A,B) if GRID else (load(E1,f"{T}_busprobe_coll"),load(E2,f"{T}_busprobe_coll"))
    seeds=sorted(set(A)&set(B)&set(Ab)&set(Bb));
    if ARGS.seeds: seeds=[x for x in seeds if x in ARGS.seeds]   # 2026-09-30: 指定种子子集 (默认不生效)
    epA=next(iter(A.values()))["ckpt_episode"]; epB=next(iter(B.values()))["ckpt_episode"]
    bad=anomalous(A,B,seeds); keep=[s for s in seeds if s not in bad]
    GREEN="#cfe9cf"; cells=[]; colors=[]; hdr=[]; titles=[]; dump={"seeds":seeds,"anomalous":bad,"best_ep":(epA,epB)}
    def block(title, S):
        hdr.append(len(cells)+1); titles.append(title); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)
        for m,label,jlabel,gl in (("avg_stopped_time_per_visit","Stopped time / visit (s)","J(531) weighted cost (per visit)","Ordering gate (stopped time): amb ≤ bus ≤ car"),("avg_stop_events_per_visit","Stop events / visit","J(531) weighted stop events (per visit)","Ordering gate (stop events): amb ≤ bus ≤ car")):
            rows,g=metric_rows(A,B,S,m,label,jlabel); dump[title+"|"+m]=[(r[0],r[1],r[2],r[3],r[4],r[5],r[6]) for r in rows]
            for r in rows:
                col=["#eef3ff" if r[0].startswith("J(") else "#fafafa"]*5; col[2 if r[3]<r[1] else 1]=GREEN
                d=3 if "events" in label and not r[0].startswith("J(") else 2
                cells.append([r[0], f"{r[1]:.{d}f} ± {r[2]:.{d}f}", f"{r[3]:.{d}f} ± {r[4]:.{d}f}", r[5], r[6] if r[0].startswith("J(") else ""]); colors.append(col)
            amb_na = True   # 救护车 1-2 辆: 停车次数门只判 bus ≤ car
            gg=[]
            for i in (0,1):
                arm=(A,B)[i]; bu=np.nanmean([arm[s]["collector"]["summary"].get(f"eval/system/bus/{m}",np.nan) for s in S]); ca=np.nanmean([arm[s]["collector"]["summary"].get(f"eval/system/car/{m}",np.nan) for s in S])
                gg.append(("pass (amb n/a)" if bu<=ca else "fail (amb n/a)") if (m=="avg_stop_events_per_visit" and not GRID) else g[i])
            cells.append([gl, gg[0], gg[1], "", ""]); colors.append(["#ffffff"]*5)
    if GRID:
        jmax={arm:{s:max(v for k,v in X[s]["collector"]["summary"].items() if k.endswith("/all/avg_stopped_time") and not k.startswith("eval/system")) for s in seeds} for arm,X in (("8STD",A),("GS",B))}
        dump["max_junction_all_stopped_time"]=jmax; dump["anom_threshold"]=ARGS.anom
        dump["per_seed"]={arm:{c:{mm:[X[s]["collector"]["summary"].get(f"eval/system/{c}/{mm}",float("nan")) for s in seeds] for mm in ("avg_stopped_time_per_visit","avg_stop_events_per_visit")} for c in ("car","bus","ambulance","all")} for arm,X in (("8STD",A),("GS",B))}
        block(f"{WLABEL}, best checkpoints (8STD: exp{E1} ep{epA}, GS-ENUM: exp{E2} ep{epB}), {len(seeds)} seeds"+("" if bad else f"; anomaly rule (junction > {ARGS.anom:.0f} s): no hits"), seeds)
    else: block(f"{WLABEL}, ambulance probe route, best checkpoints (8STD: exp{E1} ep{epA}, GS-ENUM: exp{E2} ep{epB}), {len(seeds)} evaluation seeds", seeds)
    if bad and keep: block(f"{WLABEL}, best checkpoints, excl. {len(bad)} anomalous seed(s) (any junction > {ARGS.anom:.0f} s in either arm: {', '.join(map(str,bad))}), n = {len(keep)}", keep)
    if ARGS.tukey:
        tk,fences=tukey_outliers(A,B,seeds); dump["tukey_outliers"]=tk; dump["tukey_fences"]=fences
        if tk: block(f"{WLABEL}, best checkpoints, excl. Tukey-outlier seed(s) {', '.join(map(str,tk))} (1.5 IQR, all-class stopped time / visit, either arm), n = {len(seeds)-len(tk)}", [s for s in seeds if s not in tk])
    if not GRID: TS=typeswap(A,Ab,B,Bb,seeds); dump["typeswap"]={k:{kk:(list(map(float,v[0])),list(map(float,v[1]))) for kk,v in d.items()} for k,d in TS.items()}
    if not GRID: hdr.append(len(cells)+1); titles.append(f"{WLABEL}, same route & time, ambulance vs bus, best checkpoints, {len(seeds)} seeds"); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)
    for key,lab,d in (() if GRID else (("pv","Stopped time / visit (s)",2),("ev","Stop events / visit",3))):
        a,b=TS["8STD"][key],TS["GS"][key]; ma=(np.nanmean(a[0]),np.nanmean(a[1])); mb=(np.nanmean(b[0]),np.nanmean(b[1]))
        col=["#f3f3f3"]*5
        if ma[0]<ma[1]: col[1]=GREEN
        if mb[0]<mb[1]: col[2]=GREEN
        cells.append([f"{lab} - ambulance (probe route)", f"{ma[0]:.{d}f} ± {np.nanstd(a[0]):.{d}f}", f"{mb[0]:.{d}f} ± {np.nanstd(b[0]):.{d}f}", "", f"vs bus: {100*(ma[0]/ma[1]-1):+.0f}%, {100*(mb[0]/mb[1]-1):+.0f}%" if ma[1]>0 and mb[1]>0 else ""]); colors.append(col)
        cells.append([f"{lab} - bus (same route & time)", f"{ma[1]:.{d}f} ± {np.nanstd(a[1]):.{d}f}", f"{mb[1]:.{d}f} ± {np.nanstd(b[1]):.{d}f}", "", ""]); colors.append(["#f3f3f3"]*5)
        w=lambda x: f"{sum(1 for p,q in zip(*x) if p<q)}/{len(x[0])} (ties {sum(1 for p,q in zip(*x) if p==q)})"
        cells.append([f"{lab}: evals with ambulance < bus", w(a), w(b), "", ""]); colors.append(["#ffffff"]*5)
    fig,ax=plt.subplots(figsize=(13.5,0.32*len(cells)+2)); ax.axis("off")
    tbl=ax.table(cellText=cells,colLabels=["Metric","8STD","GS-ENUM","Rel. diff","GS wins / paired"],cellColours=colors,colColours=["#c9d3df"]*5,loc="upper center",cellLoc="center",colWidths=[0.40,0.17,0.17,0.11,0.15])
    tbl.auto_set_font_size(False); tbl.scale(1,1.45)
    for (r,c),cell in tbl.get_celld().items():
        t=cell.get_text(); t.set_fontsize(10.5)
        if r==0: t.set_weight("bold"); t.set_fontsize(11)
        if c==0 and r>0: t.set_ha("left"); cell.PAD=0.012
        if r in hdr and c>0: cell.set_edgecolor("#dde3ea")
    ax.set_title(f"8STD (template + DQN) vs GS-ENUM (enum + FRAP): {WLABEL}, best checkpoints re-evaluated on {len(seeds)} seeds, mean ± s.d. (training-collector metrics)",fontsize=12.5,pad=6)
    fig.canvas.draw(); ren=fig.canvas.get_renderer(); inv=fig.transFigure.inverted()
    for r,title in zip(hdr,titles):
        bb=tbl[r,0].get_window_extent(ren); x0,y0=inv.transform((bb.x0,bb.y0)); x1,y1=inv.transform((bb.x1,bb.y1))
        fig.text(x0+0.004,(y0+y1)/2,title,ha="left",va="center",fontsize=10.5,weight="bold",zorder=10)
    from matplotlib.transforms import Bbox
    tb=tbl.get_window_extent(ren); tt=ax.title.get_window_extent(ren); bb=Bbox.union([tb,tt]); pad=0.15*fig.dpi
    crop=Bbox.from_extents((bb.x0-pad)/fig.dpi,(bb.y0-pad)/fig.dpi,(bb.x1+pad)/fig.dpi,(bb.y1+pad)/fig.dpi)
    out=f"experiments/analysis/figures/main_comparison_{ARGS.window}_{ARGS.tag}10seeds{ARGS.out_suffix}_2026-09-30_en.png"; plt.savefig(out,dpi=200,bbox_inches=crop); print("saved",out)
    json.dump(dump, open(out.replace(".png",".json"),"w"), default=float, indent=1)
if __name__=="__main__": main()
