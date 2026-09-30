"""Dublin 02:00 救护车探针对 (exp287 8STD vs exp288 GS-ENUM, 救护车路线 = 11h amb_1) 的 per-visit / xts / J_eq 柱状图。
输入: experiments/analysis/data/dublin_02h_probe_tail40_rows_2026-09-30.json (尾40, 按评估步对齐)
输出 (新文件名, 不覆盖既有图): figures/dublin_02h/dublin02h_probe_{pervisit,xts,jeq_pervisit,jeq_xts}.{png,pdf}
运行: python experiments/tools/figures/plot_dublin_02h_probe_2026-09-30.py
"""
import importlib.util, json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
spec=importlib.util.spec_from_file_location("pmc","experiments/tools/figures/plot_main_comparison_2026-09-02.py"); pmc=importlib.util.module_from_spec(spec); spec.loader.exec_module(pmc)
S=json.load(open("experiments/analysis/data/dublin_02h_probe_tail40_rows_2026-09-30.json"))
A=lambda e,k: np.array(S[e]["rows"][k],dtype=float)
SH={"car":.9879,"bus":.0109,"ambulance":.0012}; WT={"car":1,"bus":3,"ambulance":5}   # 02h 路由份额 (car 1630 / bus 18 / amb 2)
def block(a,b,m):
    va={c:A(a,f"eval_system/{c}/{m}") for c in ("car","bus","ambulance","all")}; vb={c:A(b,f"eval_system/{c}/{m}") for c in ("car","bus","ambulance","all")}
    J=lambda v: sum(SH[c]*WT[c]*np.nan_to_num(v[c]) for c in SH); E=lambda v: sum(WT[c]*np.nan_to_num(v[c]) for c in SH)
    ms=lambda x: (float(np.nanmean(x)), float(np.nanstd(x)))
    return {"std":[ms(va[c])[0] for c in ("car","bus","ambulance","all")], "std_e":[ms(va[c])[1] for c in ("car","bus","ambulance","all")],
            "gs":[ms(vb[c])[0] for c in ("car","bus","ambulance","all")], "gs_e":[ms(vb[c])[1] for c in ("car","bus","ambulance","all")],
            "jstd":[ms(J(va))[0], ms(E(va))[0]], "jstd_e":[ms(J(va))[1], ms(E(va))[1]], "jgs":[ms(J(vb))[0], ms(E(vb))[0]], "jgs_e":[ms(J(vb))[1], ms(E(vb))[1]]}
def main():
    pmc.apply_publication_style(pmc.FigureStyle(font_size=16, axes_linewidth=2))
    a,b,name,tag="287","288","Dublin 02:00 (ambulance probe)","dublin02h_probe"
    sc=dict(name=name, exps=(f"exp{a}",f"exp{b}"), subdir="dublin_02h",
        pervisit={"stopped time / visit (s)":block(a,b,"avg_stopped_time_per_visit"),"stop events / visit":block(a,b,"avg_stop_events_per_visit")},
        xts={"xts stopped time (s)":block(a,b,"xts_avg_stopped_time"),"xts stop events":block(a,b,"xts_avg_stop_events"),"xts speed (m/s)":block(a,b,"xts_avg_speed")})
    for grp,suffix in (("pervisit","pervisit"),("xts","xts")):
        for fn,mid in ((pmc.panel_group,""),(pmc.jeq_group,"jeq_")):
            for p in fn(sc, sc[grp], suffix):
                newp=os.path.join(os.path.dirname(str(p)), f"{tag}_{mid}{suffix}{os.path.splitext(str(p))[1]}"); os.replace(str(p),newp); print("saved:",newp)
if __name__=="__main__": main()
