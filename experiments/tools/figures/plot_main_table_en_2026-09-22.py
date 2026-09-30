"""英文主对比表格 (2026-09-22 版, 新文件, 不覆盖 2026-09-02 版):
1x1 (274/263) → 1x3 (275/265) → Dublin 11:00 (208/211) → Dublin 02:00 (283/284) → Dublin 18:00 (285/286)
→ Dublin 18:00 剔除 4 个饱和路口 (任一臂尾40均值 >50 s, 两臂对称; 访问量加权, 权重 = 路由文件车辆数, 两臂相同)
→ Dublin 02:00 救护车探针对 (exp287/288, 救护车路线 = 11h amb_1, 两辆): 全量尾40 + 剔除 1 次异常评估 (任一臂任一路口 >50 s: 8STD ep124)
→ Dublin 02:00 best.pth × 10 评估种子 (287 ep770 / 288 ep760, 收集器口径, 停车时间 + 停车次数 + 门) 与其同路换车型块 (2026-09-30)
→ 1x1 / 1x3 best.pth × 10 评估种子 (274 ep1325 / 263 ep1370; 275 ep1785 / 265 ep580; 用户令 2026-09-30), 排在 02:00 best 块之前 (场景顺序 1x1 → 1x3 → Dublin):
   异常规则 "任一臂任一路口全车类停车 > 50 s" 两场景均未命中 (最大 37.0 s); 1x1 另附 "对称剔除 Tukey 1.5×IQR 离群种子 128" 块 (1x3 无离群种子, 不出块)。
   数据来自 plot_best10seeds_table_2026-09-30.py --window 1x1|1x3 的 JSON。
→ Dublin 18:00 两条规则叠加 (剔 4 口 + 剔锁死评估, n=28, 访问量加权)
→ Dublin 18:00 剔除合流口 cluster_21620852 锁死评估 (规则对称: 任一臂该口 all 停车时间 >100 s 的评估两臂同剔;
   尾40 里只有 GS 触发 12 次, 8STD 0 次, 标题如实标注; n=28 配对)。
每行更优者标绿 (停车时间/visit 与停车次数/visit 各 4 类 + J 行); 保序门: 严格 amb ≤ bus ≤ car 通过写 pass; 严格不过但相邻类差 ≤ 5 s (一个决策步 / 5 s 采样量子)
写 pass (tol. 5 s); 否则 fail。停车次数门: Dublin 02:00/18:00 救护车只有 2/1 辆 (一次停车事件即改变 0.08-0.5 次/visit), 救护车不可判, 只判 bus ≤ car 并标 (amb n/a)。数值来自 DUBLIN_02H_18H_VERDICT_2026-09-22.txt / 1X1 / 1X3 / DUBLIN_208_VS_211 文档。
运行: python experiments/tools/figures/plot_main_table_en_2026-09-22.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]=["DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
def gate(am,bu,ca,tol=5.0):
    if am<=bu<=ca: return "pass"
    if am-bu<=tol and bu-ca<=tol: return "pass (tol. 5 s)"
    return "fail"
S=[("1x1  (8STD: exp274, GS-ENUM: exp263)",{"car":((32.40,2.19),(31.30,2.19)),"bus":((7.46,1.32),(7.25,1.10)),"amb":((5.38,2.37),(4.50,2.84)),"all":((31.06,2.06),(30.00,2.11)),"stops":((1.11,0.05),(1.10,0.07)),"J":((32.06,2.10),(30.97,2.19))},"28 / 40"),
   ("1x3  (8STD: exp275, GS-ENUM: exp265)",{"car":((26.71,1.34),(28.60,1.01)),"bus":((5.52,0.74),(7.83,0.57)),"amb":((3.23,1.93),(5.90,2.12)),"all":((25.06,1.23),(26.99,0.95)),"stops":((1.00,0.03),(1.03,0.03)),"J":((26.20,1.28),(28.35,0.99))},"3 / 40"),
   ("Dublin 11:00  (8STD: exp208, GS-ENUM: exp211)",{"car":((4.95,1.24),(2.44,0.32)),"bus":((1.26,0.18),(1.06,0.14)),"amb":((0.90,0.69),(0.92,0.48)),"all":((4.75,1.17),(2.38,0.31)),"stops":((0.55,0.09),(0.27,0.03)),"J":((4.89,1.18),(2.48,0.31))},"40 / 40"),
   ("Dublin 02:00  (8STD: exp283, GS-ENUM: exp284)",{"car":((1.02,0.14),(0.85,0.13)),"bus":((0.57,0.28),(0.41,0.16)),"amb":((0.94,0.52),(1.28,0.95)),"all":((1.02,0.14),(0.84,0.13)),"stops":((0.15,0.02),(0.12,0.02)),"J":((1.04,0.14),(0.86,0.14))},"32 / 40"),
   ("Dublin 18:00  (8STD: exp285, GS-ENUM: exp286)",{"car":((16.33,2.49),(20.46,9.94)),"bus":((14.87,6.95),(17.96,19.41)),"amb":((2.94,2.65),(6.93,11.26)),"all":((16.25,2.66),(20.31,9.99)),"stops":((0.68,0.11),(0.51,0.21)),"J":((17.55,3.17),(19.94,9.69))},"16 / 35"),
   ("Dublin 18:00, excl. 4 saturated junctions  (8STD: exp285, GS-ENUM: exp286)",{"car":((9.84,2.73),(9.44,5.33)),"bus":((6.01,3.88),(10.05,16.66)),"amb":((4.50,4.58),(6.88,5.99)),"all":((9.53,2.76),(9.49,5.90)),"stops":((0.58,0.10),(0.45,0.23)),"J":((10.20,3.02),(10.34,6.73))},"27 / 40"),
   ("Dublin 18:00, excl. gridlock evals at cluster_21620852 (GS 12/40, 8STD 0/40)  (8STD: exp285, GS-ENUM: exp286)",{"car":((16.14,2.63),(15.58,4.04)),"bus":((14.92,7.80),(14.81,12.68)),"amb":((2.95,2.59),(4.55,3.54)),"all":((16.07,2.83),(15.53,4.24)),"stops":((0.67,0.12),(0.51,0.20)),"J":((17.37,3.41),(16.83,5.04))},"16 / 28"),
   ("Dublin 18:00, excl. 4 saturated junctions and gridlock evals (n = 28)  (8STD: exp285, GS-ENUM: exp286)",{"car":((9.38,2.58),(8.99,4.21)),"bus":((5.70,4.26),(5.99,9.67)),"amb":((4.29,4.57),(8.04,5.88)),"all":((9.07,2.67),(8.75,4.30)),"J":((9.71,2.95),(9.39,4.76))},"18 / 28"),
   ("Dublin 02:00, ambulance probe route (11h amb_1; 2 ambulances)  (8STD: exp287, GS-ENUM: exp288)",{"car":((1.02,0.78),(0.84,0.09)),"bus":((0.56,0.75),(0.36,0.15)),"amb":((0.69,0.88),(0.75,0.61)),"all":((1.02,0.78),(0.83,0.09)),"J":((1.030,0.797),(0.845,0.088))},"27 / 40"),
   ("Dublin 02:00, ambulance probe route, excl. 1 anomalous eval (8STD ep124 > 50 s)  (8STD: exp287, GS-ENUM: exp288)",{"car":((0.90,0.13),(0.84,0.09)),"bus":((0.44,0.18),(0.36,0.15)),"amb":((0.64,0.84),(0.74,0.61)),"all":((0.89,0.12),(0.84,0.08)),"J":((0.904,0.123),(0.849,0.085))},"26 / 39")]
EV={'1x1': {'ev_car': ((1.13, 0.05), (1.11, 0.07)), 'ev_bus': ((0.85, 0.08), (0.79, 0.07)), 'ev_amb': ((0.49, 0.14), (0.47, 0.16)), 'ev_all': ((1.11, 0.05), (1.1, 0.07)), 'ev_J': ((1.18, 0.05), (1.16, 0.07)), 'ev_wins': '25 / 40'}, '1x3': {'ev_car': ((1.02, 0.03), (1.05, 0.03)), 'ev_bus': ((0.68, 0.07), (0.84, 0.03)), 'ev_amb': ((0.4, 0.22), (0.72, 0.24)), 'ev_all': ((1.0, 0.03), (1.03, 0.03)), 'ev_J': ((1.07, 0.03), (1.12, 0.03)), 'ev_wins': '0 / 40'}, 'Dublin 11:00': {'ev_car': ((0.57, 0.09), (0.27, 0.03)), 'ev_bus': ((0.18, 0.02), (0.15, 0.02)), 'ev_amb': ((0.14, 0.1), (0.15, 0.07)), 'ev_all': ((0.55, 0.09), (0.27, 0.03)), 'ev_J': ((0.566, 0.091), (0.281, 0.027)), 'ev_wins': '40 / 40'}, 'Dublin 02:00': {'ev_car': ((0.15, 0.02), (0.12, 0.02)), 'ev_bus': ((0.09, 0.04), (0.07, 0.03)), 'ev_amb': ((0.16, 0.09), (0.2, 0.13)), 'ev_all': ((0.15, 0.02), (0.12, 0.02)), 'ev_J': ((0.156, 0.022), (0.125, 0.019)), 'ev_wins': '35 / 40'}, 'Dublin 18:00  ': {'ev_car': ((0.7, 0.12), (0.52, 0.22)), 'ev_bus': ((0.32, 0.05), (0.28, 0.05)), 'ev_amb': ((0.45, 0.37), (0.61, 0.42)), 'ev_all': ((0.68, 0.11), (0.51, 0.21)), 'ev_J': ((0.711, 0.117), (0.531, 0.212)), 'ev_wins': '26 / 35'}, 'Dublin 18:00, excl. 4': {'ev_car': ((0.61, 0.11), (0.48, 0.25)), 'ev_bus': ((0.24, 0.05), (0.19, 0.04)), 'ev_amb': ((0.72, 0.63), (0.78, 0.52)), 'ev_all': ((0.58, 0.1), (0.45, 0.23)), 'ev_J': ((0.612, 0.109), (0.481, 0.237)), 'ev_wins': '28 / 40'}, 'Dublin 18:00, excl. gridlock': {'ev_car': ((0.69, 0.13), (0.52, 0.21)), 'ev_bus': ((0.33, 0.05), (0.28, 0.05)), 'ev_amb': ((0.43, 0.32), (0.54, 0.3)), 'ev_all': ((0.67, 0.12), (0.51, 0.2)), 'ev_J': ((0.703, 0.12), (0.534, 0.2)), 'ev_wins': '21 / 28'}}
EV["Dublin 02:00, ambulance probe route (11h"]={"ev_car":((0.13,0.02),(0.12,0.01)),"ev_bus":((0.07,0.03),(0.06,0.03)),"ev_amb":((0.14,0.18),(0.15,0.12)),"ev_all":((0.13,0.02),(0.12,0.01)),"ev_J":((0.136,0.018),(0.124,0.014)),"ev_wins":"28 / 40"}
EV["Dublin 02:00, ambulance probe route, excl."]={"ev_car":((0.13,0.02),(0.12,0.01)),"ev_bus":((0.07,0.03),(0.06,0.03)),"ev_amb":((0.13,0.17),(0.15,0.12)),"ev_all":((0.13,0.02),(0.12,0.01)),"ev_J":((0.135,0.018),(0.125,0.014)),"ev_wins":"27 / 39"}
EV["Dublin 18:00, excl. 4 saturated junctions and gridlock"]={"ev_car":((0.60,0.12),(0.49,0.23)),"ev_bus":((0.25,0.04),(0.19,0.05)),"ev_amb":((0.68,0.60),(0.93,0.46)),"ev_all":((0.57,0.11),(0.46,0.22)),"ev_J":((0.61,0.12),(0.49,0.23)),"ev_wins":"19 / 28"}
ROWS=[("car","Stopped time / visit (s) - car"),("bus","Stopped time / visit (s) - bus"),("amb","Stopped time / visit (s) - ambulance"),("all","Stopped time / visit (s) - all"),("J","J(531) weighted cost (per visit)")]
fmt=lambda m,s: f"{m:.2f} ± {s:.2f}"; GREEN="#cfe9cf"; cells=[];colors=[];hdr=[]
for scen,d,wins in S:
    hdr.append(len(cells)+1); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)   # 块标题在下方以 fig.text 叠加 (跨列)
    for key,lab in ROWS:
        (am,asd),(bm,bsd)=d[key]; delta=(bm-am)/am*100
        c=["#eef3ff" if key=="J" else "#fafafa"]*5; c[2 if bm<am else 1]=GREEN
        cells.append([lab,fmt(am,asd),fmt(bm,bsd),f"{delta:+.0f}%",wins if key=="J" else ""]); colors.append(c)
    g=[gate(d["amb"][i][0],d["bus"][i][0],d["car"][i][0]) for i in (0,1)]
    cells.append(["Ordering gate (stopped time): amb ≤ bus ≤ car",g[0],g[1],"",""]); colors.append(["#ffffff"]*5)
    # 停车次数 / visit 分车类 + J + 门 (救护车 n ≤ 2 辆的时段, 一次停车事件即改变 0.08-0.5, 门只判 bus ≤ car, 救护车标 n/a)
    ev=EV[max((k for k in EV if scen.startswith(k)), key=len)]   # 取最长匹配键 (第 6/8 块前缀相同)
    for key,lab in (("ev_car","Stop events / visit - car"),("ev_bus","Stop events / visit - bus"),("ev_amb","Stop events / visit - ambulance"),("ev_all","Stop events / visit - all"),("ev_J","J(531) weighted stop events (per visit)")):
        (am,asd),(bm,bsd)=ev[key]; delta=(bm-am)/am*100
        c=["#eef3ff" if key=="ev_J" else "#fafafa"]*5; c[2 if bm<am else 1]=GREEN
        cells.append([lab,f"{am:.2f} ± {asd:.2f}",f"{bm:.2f} ± {bsd:.2f}",f"{delta:+.0f}%",ev["ev_wins"] if key=="ev_J" else ""]); colors.append(c)
    amb_na = scen.startswith("Dublin 02:00") or scen.startswith("Dublin 18:00")   # 02h/18h 救护车 1-2 辆
    ge=[]
    for i in (0,1):
        am,bu,ca=ev["ev_amb"][i][0],ev["ev_bus"][i][0],ev["ev_car"][i][0]
        ge.append(("pass (amb n/a)" if bu<=ca else "fail (amb n/a)") if amb_na else gate(am,bu,ca))
    cells.append(["Ordering gate (stop events): amb ≤ bus ≤ car",ge[0],ge[1],"",""]); colors.append(["#ffffff"]*5)

# ── 附加块 (2026-09-30 用户令): 同路线同时刻换车型 (评估专用, 现有检查点 ep650-800, 每臂 8 车次): 救护车 vs 公交, 只有车型不同 ──
import json as _json
TS=_json.load(open("experiments/analysis/data/typeswap_02h_pervisit_2026-09-30.json"))
def _tscol(TS,key):
    col=["#f3f3f3"]*5
    for i_,arm in ((1,"8STD"),(2,"GS")):
        if TS[arm][key][0] < TS[arm][key][1]: col[i_]=GREEN   # 救护车 < 同路公交 → 绿
    return col
def _tswins(TS,arm,key):
    pa=TS[arm]["per_eval"]["amb_"+key]; pb=TS[arm]["per_eval"]["bus_"+key]
    w=sum(1 for x,y in zip(pa,pb) if x<y); t=sum(1 for x,y in zip(pa,pb) if x==y); return f"{w}/{len(pa)} (ties {t})"

hdr.append(len(cells)+1); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)
TSWAP_TITLE=f"Dublin 02:00, same route & time, type swapped: ambulance vs bus  (8STD: exp287, GS-ENUM: exp288; {TS['8STD']['n']} evals/arm)"
for key,sdk,lab in (("pv","pv_sd","Stopped time / visit (s)"),("ev","ev_sd","Stop events / visit")):
    a,b=TS["8STD"][key],TS["GS"][key]; asd,bsd=TS["8STD"][sdk],TS["GS"][sdk]
    d=3 if key=="ev" else 2
    cells.append([f"{lab} - ambulance (probe route)", f"{a[0]:.{d}f} ± {asd[0]:.{d}f}", f"{b[0]:.{d}f} ± {bsd[0]:.{d}f}", "", f"vs bus: {100*(a[0]/a[1]-1):+.0f}%, {100*(b[0]/b[1]-1):+.0f}%"]); colors.append(_tscol(TS,key))
    cells.append([f"{lab} - bus (same route & time)", f"{a[1]:.{d}f} ± {asd[1]:.{d}f}", f"{b[1]:.{d}f} ± {bsd[1]:.{d}f}", "", ""]); colors.append(["#f3f3f3"]*5)
    cells.append([f"{lab}: evals with ambulance < bus", _tswins(TS,"8STD",key), _tswins(TS,"GS",key), "", ""]); colors.append(["#ffffff"]*5)

# ── 附加块 (2026-09-30 用户令): 1x1 / 1x3 best.pth × 10 评估种子 (收集器口径, 同种子配对); 全 10 种子块 + (若有) 对称剔除 Tukey 离群种子块 ──
GRID_TITLES=[]
def _grid_best(window, e1, e2):
    B_=_json.load(open(f"experiments/analysis/figures/main_comparison_{window}_best10seeds_2026-09-30_en.json")); epA,epB=B_["best_ep"]; ns=len(B_["seeds"]); tk=B_.get("tukey_outliers",[])
    for bt in dict.fromkeys(k.split("|")[0] for k in B_ if "|" in k):
        hdr.append(len(cells)+1); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)
        if "excl." in bt: GRID_TITLES.append(f"{window}, best checkpoints, excl. Tukey-outlier seed {', '.join(map(str,tk))} (1.5 IQR, either arm), n = {ns-len(tk)}  (8STD: exp{e1}, GS-ENUM: exp{e2})")
        else: GRID_TITLES.append(f"{window}, best checkpoints x {ns} seeds (anomaly rule > {B_['anom_threshold']:.0f} s: no hits)  (8STD: exp{e1} ep{epA}, GS-ENUM: exp{e2} ep{epB})")
        for m,glab in (("avg_stopped_time_per_visit","Ordering gate (stopped time): amb ≤ bus ≤ car"),("avg_stop_events_per_visit","Ordering gate (stop events): amb ≤ bus ≤ car")):
            rows=B_[bt+"|"+m]
            for lab,a,asd,b,bsd,rel,wins in rows:
                isJ=lab.startswith("J("); col=["#eef3ff" if isJ else "#fafafa"]*5; col[2 if b<a else 1]=GREEN
                d=3 if (m=="avg_stop_events_per_visit" and not isJ) else 2
                cells.append([lab, f"{a:.{d}f} ± {asd:.{d}f}", f"{b:.{d}f} ± {bsd:.{d}f}", rel, wins if isJ else ""]); colors.append(col)
            mean={r[0].split(" - ")[-1]: (r[1], r[3]) for r in rows if " - " in r[0]}
            g=[gate(mean["ambulance"][i],mean["bus"][i],mean["car"][i]) for i in (0,1)]   # 网格场景救护车每集 1-3 辆 × 10 种子, 两个门都全判 (与上面 1x1/1x3 尾40 块一致)
            cells.append([glab, g[0], g[1], "", ""]); colors.append(["#ffffff"]*5)
_grid_best("1x1","274","263"); _grid_best("1x3","275","265")

# ── 附加块 (2026-09-30 用户令): best.pth × 10 评估种子 (287 ep770 / 288 ep760), 收集器口径, 含停车次数; 数据来自 plot_02h_best10seeds_table 的 JSON ──
_B=_json.load(open("experiments/analysis/figures/main_comparison_02h_best10seeds_2026-09-30_en.json"))
_bk=[k for k in _B if "best checkpoints (" in k and "excl" not in k]
_epA,_epB=_B["best_ep"]; _ns=len(_B["seeds"])
hdr.append(len(cells)+1); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)
BEST_TITLE=f"Dublin 02:00, ambulance probe route, best checkpoints x {_ns} seeds  (8STD: exp287 ep{_epA}, GS-ENUM: exp288 ep{_epB})"
def _gate_txt(rows, m):
    mean={r[0].split(" - ")[-1]: (r[1], r[3]) for r in rows if " - " in r[0]}
    out=[]
    for i in (0,1):
        am,bu,ca=mean["ambulance"][i],mean["bus"][i],mean["car"][i]
        if m=="avg_stop_events_per_visit": out.append("pass (amb n/a)" if bu<=ca else "fail (amb n/a)")
        else: out.append(gate(am,bu,ca))
    return out
for m,lab_short,glab in (("avg_stopped_time_per_visit","Stopped time / visit (s)","Ordering gate (stopped time): amb ≤ bus ≤ car"),("avg_stop_events_per_visit","Stop events / visit","Ordering gate (stop events): amb ≤ bus ≤ car")):
    rows=_B[[k for k in _bk if k.endswith("|"+m)][0]]
    for r in rows:
        lab,a,asd,b,bsd,rel,wins=r; isJ=lab.startswith("J(")
        col=["#eef3ff" if isJ else "#fafafa"]*5; col[2 if b<a else 1]=GREEN
        d=3 if (m=="avg_stop_events_per_visit" and not isJ) else 2
        cells.append([lab, f"{a:.{d}f} ± {asd:.{d}f}", f"{b:.{d}f} ± {bsd:.{d}f}", rel, wins if isJ else ""]); colors.append(col)
    g=_gate_txt(rows, m); cells.append([glab, g[0], g[1], "", ""]); colors.append(["#ffffff"]*5)
# 同路线换车型 (best.pth × 10 seeds)
hdr.append(len(cells)+1); cells.append(["","","","",""]); colors.append(["#dde3ea"]*5)
BEST_TS_TITLE=f"Dublin 02:00, same route & time, ambulance vs bus, best checkpoints x {_ns} seeds  (8STD: exp287, GS-ENUM: exp288)"
import numpy as _np
for key,lab,d in (("pv","Stopped time / visit (s)",2),("ev","Stop events / visit",3)):
    a=_B["typeswap"]["8STD"][key]; b=_B["typeswap"]["GS"][key]
    ma=(_np.mean(a[0]),_np.mean(a[1])); mb=(_np.mean(b[0]),_np.mean(b[1])); col=["#f3f3f3"]*5
    if ma[0]<ma[1]: col[1]=GREEN
    if mb[0]<mb[1]: col[2]=GREEN
    cells.append([f"{lab} - ambulance (probe route)", f"{ma[0]:.{d}f} ± {_np.std(a[0]):.{d}f}", f"{mb[0]:.{d}f} ± {_np.std(b[0]):.{d}f}", "", f"vs bus: {100*(ma[0]/ma[1]-1):+.0f}%, {100*(mb[0]/mb[1]-1):+.0f}%"]); colors.append(col)
    cells.append([f"{lab} - bus (same route & time)", f"{ma[1]:.{d}f} ± {_np.std(a[1]):.{d}f}", f"{mb[1]:.{d}f} ± {_np.std(b[1]):.{d}f}", "", ""]); colors.append(["#f3f3f3"]*5)
    _w=lambda x: f"{sum(1 for p,q in zip(*x) if p<q)}/{len(x[0])} (ties {sum(1 for p,q in zip(*x) if p==q)})"
    cells.append([f"{lab}: evals with ambulance < bus", _w(a), _w(b), "", ""]); colors.append(["#ffffff"]*5)
fig,ax=plt.subplots(figsize=(13.5,max(52,0.3312*len(cells)))); ax.axis("off")   # 原 157 行 = 52 in; 加块后按行数等比加高
tbl=ax.table(cellText=cells,colLabels=["Metric","8STD","GS-ENUM","Rel. diff","GS wins / paired"],cellColours=colors,colColours=["#c9d3df"]*5,loc="upper center",cellLoc="center",colWidths=[0.40,0.17,0.17,0.11,0.15])
tbl.auto_set_font_size(False); tbl.scale(1,1.45)
for (r,c),cell in tbl.get_celld().items():
    t=cell.get_text(); t.set_fontsize(10.5)
    if r==0: t.set_weight("bold"); t.set_fontsize(11)
    if c==0 and r>0: t.set_ha("left"); cell.PAD=0.012
    if r in hdr:
        t.set_weight("bold")
        if c>0: cell.set_edgecolor("#dde3ea")
ax.set_title("8STD (template + DQN) vs GS-ENUM (enum + FRAP): tail-40 evaluations, mean ± s.d.",fontsize=13,pad=6)
fig.canvas.draw(); ren=fig.canvas.get_renderer(); inv=fig.transFigure.inverted()
for r,scen in zip(hdr,[x[0] for x in S]+[TSWAP_TITLE]+GRID_TITLES+[BEST_TITLE,BEST_TS_TITLE]):   # 块标题跨列叠加, 不受相邻单元格遮挡
    bb=tbl[r,0].get_window_extent(ren); x0,y0=inv.transform((bb.x0,bb.y0)); x1,y1=inv.transform((bb.x1,bb.y1))
    fig.text(x0+0.004, (y0+y1)/2, scen, ha="left", va="center", fontsize=10.5, weight="bold", zorder=10)
from matplotlib.transforms import Bbox
tb=tbl.get_window_extent(ren); tt=ax.title.get_window_extent(ren)
bb=Bbox.union([tb,tt]); pad=0.15*fig.dpi
crop=Bbox.from_extents((bb.x0-pad)/fig.dpi,(bb.y0-pad)/fig.dpi,(bb.x1+pad)/fig.dpi,(bb.y1+pad)/fig.dpi)
out="experiments/analysis/figures/main_comparison_8std_vs_frap_2026-09-22_en.png"
plt.savefig(out,dpi=200,bbox_inches=crop); print("saved",out)
