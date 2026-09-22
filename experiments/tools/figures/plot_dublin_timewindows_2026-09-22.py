"""Dublin 02h/18h 时段对比图 (8STD vs GS-ENUM), 复用 plot_main_comparison_2026-09-02 的风格函数。
数据: experiments/analysis/data/dublin_02h_18h_tail40_rows_2026-09-22.json (按评估步对齐的尾40逐评数据)。
输出 (新文件名, 不覆盖既有图, 按时段分文件夹): figures/dublin_02h/dublin02h_{pervisit,xts,jeq_pervisit,jeq_xts}, dublin18h_{同上},
                                figures/dublin_18h/dublin18h_{xts,pervisit}_excl4junctions (剔除 4 个大堵塞口: 路口等权 / 访问量加权视图)。\n运行 --only-excl 只重画剔除视图 (不重写其它图)。
"""
import importlib.util, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
spec=importlib.util.spec_from_file_location("pmc", os.path.join(HERE,"plot_main_comparison_2026-09-02.py")); pmc=importlib.util.module_from_spec(spec); spec.loader.exec_module(pmc)
S=json.load(open(os.path.join(REPO,"experiments/analysis/data/dublin_02h_18h_tail40_rows_2026-09-22.json")))
W={"car":1,"bus":3,"ambulance":5}; SH={"02h":{"car":0.9879,"bus":0.0109,"ambulance":0.0012},"18h":{"car":0.9568,"bus":0.0430,"ambulance":0.0002}}
CL=["car","bus","ambulance","all"]
def A(e,k): return np.array(S[e]["rows"][k],dtype=float)
def ms(x): x=x[~np.isnan(x)]; return round(float(x.mean()),2), round(float(x.std(ddof=1)),2)
def block(h,a,b,m,src=lambda e,c,m: A(e,f"eval_system/{c}/{m}")):
    std=[ms(src(a,c,m)) for c in CL]; gs=[ms(src(b,c,m)) for c in CL]
    Ja=sum(SH[h][c]*W[c]*src(a,c,m) for c in W); Jb=sum(SH[h][c]*W[c]*src(b,c,m) for c in W)
    Ea=sum(W[c]*src(a,c,m) for c in W); Eb=sum(W[c]*src(b,c,m) for c in W)
    return pmc.M([x[0] for x in std],[x[1] for x in std],[x[0] for x in gs],[x[1] for x in gs],[ms(Ja)[0],ms(Ea)[0]],[ms(Ja)[1],ms(Ea)[1]],[ms(Jb)[0],ms(Eb)[0]],[ms(Jb)[1],ms(Eb)[1]])
def scen(h,a,b,name):
    return dict(name=name, exps=(f"exp{a}",f"exp{b}"),
        pervisit={"stopped time / visit (s)":block(h,a,b,"avg_stopped_time_per_visit"),"stop events / visit":block(h,a,b,"avg_stop_events_per_visit")},
        xts={"xts stopped time (s)":block(h,a,b,"xts_avg_stopped_time"),"xts stop events":block(h,a,b,"xts_avg_stop_events"),"xts speed (m/s)":block(h,a,b,"xts_avg_speed")})
def _sub(p,h):
    """按时段分文件夹: figures/dublin_02h, figures/dublin_18h (11h 主对在 figures/dublin_11h)。"""
    base=os.path.dirname(str(p))
    if os.path.basename(base).startswith("dublin_"): base=os.path.dirname(base)
    d=os.path.join(base,f"dublin_{h}"); os.makedirs(d,exist_ok=True); return d
def main():
    pmc.apply_publication_style(pmc.FigureStyle(font_size=16, axes_linewidth=2))
    for h,a,b,name in [("02h","283","284","Dublin 02:00"),("18h","285","286","Dublin 18:00")]:
        sc=scen(h,a,b,name); tag=f"dublin{h}"
        # 复用主图/J_eq 图函数, 但输出名带时段
        sc["name"]=name
        for grp,suffix in (("pervisit","pervisit"),("xts","xts")):
            s2=dict(sc); s2["name"]=name
            fig_paths=pmc.panel_group({**s2,"name":name}, sc[grp], suffix)  # 会写 figures/dublin 02:00_xxx → 下面重命名
            for p in fig_paths:
                newp=os.path.join(_sub(p,h), f"{tag}_{suffix}{os.path.splitext(str(p))[1]}"); os.replace(str(p),newp); print("saved:",newp)
            fig_paths=pmc.jeq_group({**s2,"name":name}, sc[grp], suffix)
            for p in fig_paths:
                newp=os.path.join(_sub(p,h), f"{tag}_jeq_{suffix}{os.path.splitext(str(p))[1]}"); os.replace(str(p),newp); print("saved:",newp)
    excl_views()
def excl_views():
    """18h 剔除 4 个大堵塞口 (任一臂尾40均值 >50 s, 对两臂对称) 的两种视图:
    (1) 路口等权 (xts) — dublin18h_xts_excl4junctions;
    (2) 访问量加权 (近似 per-visit; 权重 = 路由文件中各车类经过该口的车辆数, 两臂完全相同,
        data/dublin_route_visits_by_junction_2026-09-22.json) — dublin18h_pervisit_excl4junctions。"""
    h,a,b="18h","285","286"; tls=S[a]["tls"]
    excl=[t for t in tls if A(a,f"eval_{t}/all/avg_stopped_time").mean()>50 or A(b,f"eval_{t}/all/avg_stopped_time").mean()>50]; keep=[t for t in tls if t not in excl]
    src=lambda e,c,m: np.nanmean(np.stack([A(e,f"eval_{t}/{c}/{m}") for t in keep]),axis=0)
    sc=dict(name=f"Dublin 18:00 (excl. {len(excl)} saturated junctions)", exps=("exp285","exp286"),
            xts={"xts stopped time (s)":block(h,a,b,"avg_stopped_time",src),"xts stop events":block(h,a,b,"avg_stop_events",src),"xts speed (m/s)":block(h,a,b,"avg_speed",src)})
    for p in pmc.panel_group(sc, sc["xts"], "xts"):
        newp=os.path.join(_sub(p,h), f"dublin18h_xts_excl4junctions{os.path.splitext(str(p))[1]}"); os.replace(str(p),newp); print("saved:",newp)
    # (2) 访问量加权视图 (近似 per-visit)
    W=json.load(open("experiments/analysis/data/dublin_route_visits_by_junction_2026-09-22.json"))[h]
    def wsrc(e,c,m):
        cls=("car","bus","ambulance") if c=="all" else (c,)
        num=sum(W[cc].get(t,0)*np.nan_to_num(A(e,f"eval_{t}/{cc}/{m}")) for cc in cls for t in keep)
        den=sum(W[cc].get(t,0) for cc in cls for t in keep); return num/den
    sc=dict(name=f"Dublin 18:00 (excl. {len(excl)} saturated junctions, visit-weighted)", exps=("exp285","exp286"),
            pervisit={"stopped time / visit (s)":block(h,a,b,"avg_stopped_time",wsrc),"stop events / visit":block(h,a,b,"avg_stop_events",wsrc)})
    for p in pmc.panel_group(sc, sc["pervisit"], "pervisit"):
        newp=os.path.join(_sub(p,h), f"dublin18h_pervisit_excl4junctions{os.path.splitext(str(p))[1]}"); os.replace(str(p),newp); print("saved:",newp)
if __name__=="__main__":
    import sys
    if "--only-excl" in sys.argv:
        pmc.apply_publication_style(pmc.FigureStyle(font_size=16, axes_linewidth=2)); excl_views()
    else: main()
