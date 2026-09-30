"""1x1 末 4 个检查点 × 10 评估种子汇总 (2026-09-30, 用户令 "best × 10 种子" 的补充检查): 判断 best 表与尾40 表方向相反
是 best 选点造成的还是评估种子 123 本身的偏差。收集器口径 (5 s 采样 per-visit), 同检查点序号同种子配对。
数据: experiments/analysis/data/pervehicle/exp{274,263}_ep0{1350,1400,1450,1500}_seed{123..132}_coll.json
  (生成: cd experiments && python tools/dublin/eval_pervehicle.py <exp> --ckpt <ckpt_epNNNNN.pth> --seed <s> --tag _coll --collector)
输出: 终端表 + experiments/analysis/data/1x1_tail4ckpt_10seeds_2026-09-30.json
运行 (仓库根目录): python experiments/tools/figures/summarize_tail_ckpts_10seeds_2026-09-30.py
"""
import glob, json, numpy as np
D="experiments/analysis/data/pervehicle"; SH={"car":.965,"bus":.033,"ambulance":.0017}; WT={"car":1,"bus":3,"ambulance":5}; M="avg_stopped_time_per_visit"
def load(e):
    out={}
    for f in glob.glob(f"{D}/exp{e}_ep*_seed*_coll.json"):
        if "best" in f or "busprobe" in f: continue
        out[(int(f.split("_ep")[1][:5]), int(f.split("_seed")[1].split("_")[0]))]=json.load(open(f))["collector"]["summary"]
    return out
A,B=load(274),load(263); keys=sorted(set(A)&set(B)); v=lambda X,k,c: X[k].get(f"eval/system/{c}/{M}",np.nan); dump={"n_paired":len(keys),"metric":M,"rows":{}}
print("n paired",len(keys))
for c in ("car","bus","ambulance","all"):
    a=np.array([v(A,k,c) for k in keys]); b=np.array([v(B,k,c) for k in keys]); ok=~np.isnan(a)&~np.isnan(b)
    dump["rows"][c]=dict(std=(np.nanmean(a),np.nanstd(a)),gs=(np.nanmean(b),np.nanstd(b)),gs_wins=int((b<a)[ok].sum()),n=int(ok.sum()))
    print(f"{c:10s} 8STD {np.nanmean(a):6.2f} ± {np.nanstd(a):.2f} | GS {np.nanmean(b):6.2f} ± {np.nanstd(b):.2f}  {100*(np.nanmean(b)/np.nanmean(a)-1):+.0f}%  GS wins {(b<a)[ok].sum()}/{ok.sum()}")
J=lambda X: np.array([sum(SH[c]*WT[c]*np.nan_to_num(v(X,k,c)) for c in SH) for k in keys]); Ja,Jb=J(A),J(B)
dump["rows"]["J"]=dict(std=(Ja.mean(),Ja.std()),gs=(Jb.mean(),Jb.std()),gs_wins=int((Jb<Ja).sum()),n=len(keys))
print(f"J(531)     8STD {Ja.mean():6.2f} ± {Ja.std():.2f} | GS {Jb.mean():6.2f} ± {Jb.std():.2f}  {100*(Jb.mean()/Ja.mean()-1):+.0f}%  GS wins {(Jb<Ja).sum()}/{len(keys)}")
for name,idx in (("by_ckpt",0),("by_seed",1)):
    dump[name]={}
    for g in sorted({k[idx] for k in keys}):
        ks=[k for k in keys if k[idx]==g]; a=np.mean([v(A,k,"all") for k in ks]); b=np.mean([v(B,k,"all") for k in ks]); w=sum(v(B,k,"all")<v(A,k,"all") for k in ks)
        dump[name][str(g)]=dict(std=a,gs=b,gs_wins=int(w),n=len(ks)); print(f"  {name} {g}: 8STD {a:.2f} GS {b:.2f} ({100*(b/a-1):+.0f}%)  GS wins {w}/{len(ks)}")
json.dump(dump,open("experiments/analysis/data/1x1_tail4ckpt_10seeds_2026-09-30.json","w"),default=float,indent=1)
