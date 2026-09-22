"""英文主对比表格 (2026-09-22 版, 新文件, 不覆盖 2026-09-02 版):
1x1 (274/263) → 1x3 (275/265) → Dublin 11:00 (208/211) → Dublin 02:00 (283/284) → Dublin 18:00 (285/286)
→ Dublin 18:00 剔除 4 个饱和路口 (任一臂尾40均值 >50 s, 两臂对称; 访问量加权, 权重 = 路由文件车辆数, 两臂相同)
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
   ("Dublin 18:00, excl. gridlock evals at cluster_21620852 (GS 12/40, 8STD 0/40)  (8STD: exp285, GS-ENUM: exp286)",{"car":((16.14,2.63),(15.58,4.04)),"bus":((14.92,7.80),(14.81,12.68)),"amb":((2.95,2.59),(4.55,3.54)),"all":((16.07,2.83),(15.53,4.24)),"stops":((0.67,0.12),(0.51,0.20)),"J":((17.37,3.41),(16.83,5.04))},"16 / 28")]
EV={'1x1': {'ev_car': ((1.13, 0.05), (1.11, 0.07)), 'ev_bus': ((0.85, 0.08), (0.79, 0.07)), 'ev_amb': ((0.49, 0.14), (0.47, 0.16)), 'ev_all': ((1.11, 0.05), (1.1, 0.07)), 'ev_J': ((1.18, 0.05), (1.16, 0.07)), 'ev_wins': '25 / 40'}, '1x3': {'ev_car': ((1.02, 0.03), (1.05, 0.03)), 'ev_bus': ((0.68, 0.07), (0.84, 0.03)), 'ev_amb': ((0.4, 0.22), (0.72, 0.24)), 'ev_all': ((1.0, 0.03), (1.03, 0.03)), 'ev_J': ((1.07, 0.03), (1.12, 0.03)), 'ev_wins': '0 / 40'}, 'Dublin 11:00': {'ev_car': ((0.57, 0.09), (0.27, 0.03)), 'ev_bus': ((0.18, 0.02), (0.15, 0.02)), 'ev_amb': ((0.14, 0.1), (0.15, 0.07)), 'ev_all': ((0.55, 0.09), (0.27, 0.03)), 'ev_J': ((0.566, 0.091), (0.281, 0.027)), 'ev_wins': '40 / 40'}, 'Dublin 02:00': {'ev_car': ((0.15, 0.02), (0.12, 0.02)), 'ev_bus': ((0.09, 0.04), (0.07, 0.03)), 'ev_amb': ((0.16, 0.09), (0.2, 0.13)), 'ev_all': ((0.15, 0.02), (0.12, 0.02)), 'ev_J': ((0.156, 0.022), (0.125, 0.019)), 'ev_wins': '35 / 40'}, 'Dublin 18:00  ': {'ev_car': ((0.7, 0.12), (0.52, 0.22)), 'ev_bus': ((0.32, 0.05), (0.28, 0.05)), 'ev_amb': ((0.45, 0.37), (0.61, 0.42)), 'ev_all': ((0.68, 0.11), (0.51, 0.21)), 'ev_J': ((0.711, 0.117), (0.531, 0.212)), 'ev_wins': '26 / 35'}, 'Dublin 18:00, excl. 4': {'ev_car': ((0.61, 0.11), (0.48, 0.25)), 'ev_bus': ((0.24, 0.05), (0.19, 0.04)), 'ev_amb': ((0.72, 0.63), (0.78, 0.52)), 'ev_all': ((0.58, 0.1), (0.45, 0.23)), 'ev_J': ((0.612, 0.109), (0.481, 0.237)), 'ev_wins': '28 / 40'}, 'Dublin 18:00, excl. gridlock': {'ev_car': ((0.69, 0.13), (0.52, 0.21)), 'ev_bus': ((0.33, 0.05), (0.28, 0.05)), 'ev_amb': ((0.43, 0.32), (0.54, 0.3)), 'ev_all': ((0.67, 0.12), (0.51, 0.2)), 'ev_J': ((0.703, 0.12), (0.534, 0.2)), 'ev_wins': '21 / 28'}}
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
    ev=[v for k,v in EV.items() if scen.startswith(k)][0]
    for key,lab in (("ev_car","Stop events / visit - car"),("ev_bus","Stop events / visit - bus"),("ev_amb","Stop events / visit - ambulance"),("ev_all","Stop events / visit - all"),("ev_J","J(531) weighted stop events (per visit)")):
        (am,asd),(bm,bsd)=ev[key]; delta=(bm-am)/am*100
        c=["#eef3ff" if key=="ev_J" else "#fafafa"]*5; c[2 if bm<am else 1]=GREEN
        cells.append([lab,f"{am:.2f} ± {asd:.2f}",f"{bm:.2f} ± {bsd:.2f}",f"{delta:+.0f}%",ev["ev_wins"] if key=="ev_J" else ""]); colors.append(c)
    amb_na = scen.startswith("Dublin 02:00") or scen.startswith("Dublin 18:00")
    ge=[]
    for i in (0,1):
        am,bu,ca=ev["ev_amb"][i][0],ev["ev_bus"][i][0],ev["ev_car"][i][0]
        ge.append(("pass (amb n/a)" if bu<=ca else "fail (amb n/a)") if amb_na else gate(am,bu,ca))
    cells.append(["Ordering gate (stop events): amb ≤ bus ≤ car",ge[0],ge[1],"",""]); colors.append(["#ffffff"]*5)
fig,ax=plt.subplots(figsize=(13.5,29)); ax.axis("off")
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
for r,(scen,_,_) in zip(hdr,S):   # 块标题跨列叠加, 不受相邻单元格遮挡
    bb=tbl[r,0].get_window_extent(ren); x0,y0=inv.transform((bb.x0,bb.y0)); x1,y1=inv.transform((bb.x1,bb.y1))
    fig.text(x0+0.004, (y0+y1)/2, scen, ha="left", va="center", fontsize=10.5, weight="bold", zorder=10)
from matplotlib.transforms import Bbox
tb=tbl.get_window_extent(ren); tt=ax.title.get_window_extent(ren)
bb=Bbox.union([tb,tt]); pad=0.15*fig.dpi
crop=Bbox.from_extents((bb.x0-pad)/fig.dpi,(bb.y0-pad)/fig.dpi,(bb.x1+pad)/fig.dpi,(bb.y1+pad)/fig.dpi)
out="experiments/analysis/figures/main_comparison_8std_vs_frap_2026-09-22_en.png"
plt.savefig(out,dpi=200,bbox_inches=crop); print("saved",out)
