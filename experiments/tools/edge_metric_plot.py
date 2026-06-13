"""Generic edge-metric plotter. Usage:
  python3 tools/edge_metric_plot.py <metric>
metric in {waiting, speed, stopevents}. Produces BOTH:
  - overlaid version (3 line styles)  -> fig_edge_<metric>.png
  - per-class 3x2 panels              -> fig_edge_<metric>_byclass.png
Reads analysis/edge_<metric>_data.json. Drops -532427444#2 (amb outlier edge)."""
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sumolib

METRIC=sys.argv[1] if len(sys.argv)>1 else "waiting"
FILE={"waiting":"edge_waiting_data.json","speed":"edge_speed_data.json","stopevents":"edge_stopevents_data.json"}[METRIC]
# (label, higher_is_better, cmap)
META={"waiting":("Avg local waiting per pass on edge (s)",False),
      "speed":("Avg speed on edge (m/s)",True),
      "stopevents":("Avg stop events per pass on edge",False)}[METRIC]
LABEL,HIGHER_BETTER=META
cmap=plt.cm.RdYlGn if HIGHER_BETTER else plt.cm.RdYlGn_r   # higher_better: 高=绿; 否则 高=红
A="/home/xiaowen/sumo-rl/experiments/analysis/"
net=sumolib.net.readNet("/home/xiaowen/sumo-rl/nets/dublin/dublin_8std.net.xml")
data=json.load(open(A+FILE))
# amb 少样本离群边 (St Stephen's Green 西侧瓶颈, amb 单次卡顿主导): 仅删这些边的 amb 序列
DROP_AMB={"-532427444#2","-532427444#3","532427444#7"}
for s in data:
    for k in [k for k in data[s] if k.split("|")[0] in DROP_AMB and k.endswith("|ambulance")]:
        del data[s][k]
# 保留真实街道边(含10m+短连接边,用自身值)→ 线段连续;
# 去掉: internal 路口内部边 + 长度<2m 的零长度几何残段(如gneE51,画成孤立点的artifact)。
edge_shape={e.getID():np.array(e.getShape(),float)
            for e in net.getEdges() if e.getFunction()!="internal" and e.getLength()>=2.0}
STYLE=[("car","solid",1.6),("bus",(0,(6,3)),1.8),("ambulance",(0,(1,3)),2.2)]
OFF={"car":0.0,"bus":10.0,"ambulance":-10.0}   # 偏移减小→接头更连续, 仍能区分车型
CLASSES=["car","bus","ambulance"]
allv=[v for s in data for v in data[s].values()]
lo,hi=float(np.percentile(allv,5)),float(np.percentile(allv,95))
if HIGHER_BETTER: vmin,vmax=lo,hi
else: vmin,vmax=0.0,hi
norm=plt.Normalize(vmin,vmax)
def off(shape,o):
    if len(shape)<2 or o==0: return shape
    out=[]
    for i in range(len(shape)):
        a=shape[max(0,i-1)]; b=shape[min(len(shape)-1,i+1)]; d=b-a; n=np.array([-d[1],d[0]]); ln=np.hypot(*n)
        out.append(shape[i]+(n/ln*o if ln>1e-6 else 0))
    return np.array(out)
plt.rcParams.update({"font.size":13,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"pdf.fonttype":42})

# ---- A) overlaid (3 line styles) ----
fig,axes=plt.subplots(1,2,figsize=(17,9.0))
for ax,(key,title) in zip(axes,[("fixed","Fixed-time"),("prio","Priority-RL (exp208, 5:3:1)")]):
    for sh in edge_shape.values(): ax.plot(sh[:,0],sh[:,1],color="#EAEAEA",lw=0.5,zorder=1)
    for cls,ls,lw in STYLE:
        for k,v in data[key].items():
            edge,c=k.split("|")
            if c!=cls or edge not in edge_shape: continue
            sh=off(edge_shape[edge],OFF[cls])
            ax.plot(sh[:,0],sh[:,1],color=cmap(norm(np.clip(v,vmin,vmax))),linestyle=ls,lw=lw,zorder=3,solid_capstyle="round",dash_capstyle="round")
    ax.set_title(title,fontsize=15); ax.set_aspect("equal"); ax.axis("off")
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,ax=axes,fraction=0.025,pad=0.02)
gd="green=high" if HIGHER_BETTER else "green=low"
cb.set_label(f"{LABEL}  [{gd}=better]",fontsize=12)
leg=[Line2D([0],[0],color="gray",lw=lw,linestyle=ls,label=c.capitalize()) for c,ls,lw in STYLE]
fig.legend(handles=leg,loc="lower center",ncol=3,frameon=False,fontsize=13,title="Vehicle class (line style)",bbox_to_anchor=(0.45,-0.01))
fig.suptitle(f"Dublin 11h: road-level {METRIC} by vehicle class",fontsize=15,y=0.99)
o1=A+f"fig_edge_{METRIC}"
fig.savefig(o1+".png",dpi=200,bbox_inches="tight",pad_inches=0.12); fig.savefig(o1+".pdf",bbox_inches="tight",pad_inches=0.12); plt.close(fig)

# ---- B) per-class 3x2 ----
fig,axes=plt.subplots(3,2,figsize=(12,15))
for r,cls in enumerate(CLASSES):
    for c,(key,coltitle) in enumerate([("fixed","Fixed-time"),("prio","Priority-RL (5:3:1)")]):
        ax=axes[r,c]
        for sh in edge_shape.values(): ax.plot(sh[:,0],sh[:,1],color="#ECECEC",lw=0.4,zorder=1)
        for k,v in data[key].items():
            edge,cc=k.split("|")
            if cc!=cls or edge not in edge_shape: continue
            sh=edge_shape[edge]
            ax.plot(sh[:,0],sh[:,1],color=cmap(norm(np.clip(v,vmin,vmax))),lw=2.6,zorder=3,solid_capstyle="round")
        ax.set_aspect("equal"); ax.axis("off")
        if r==0: ax.set_title(coltitle,fontsize=14,pad=8)
        if c==0: ax.text(-0.05,0.5,cls.capitalize(),transform=ax.transAxes,fontsize=15,fontweight="bold",rotation=90,va="center",ha="center")
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,ax=axes,fraction=0.025,pad=0.02); cb.set_label(f"{LABEL}  [{gd}=better]",fontsize=12)
fig.suptitle(f"Dublin 11h: road-level {METRIC} by vehicle class (per-class panels)",fontsize=15,y=0.995)
o2=A+f"fig_edge_{METRIC}_byclass"
fig.savefig(o2+".png",dpi=200,bbox_inches="tight",pad_inches=0.12); fig.savefig(o2+".pdf",bbox_inches="tight",pad_inches=0.12); plt.close(fig)
print(f"SAVED {o1}.png + {o2}.png  (metric={METRIC}, 色标 {vmin:.1f}~{vmax:.1f})")
