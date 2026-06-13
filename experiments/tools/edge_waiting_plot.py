"""Road-level local-waiting map (scheme A): each edge colored by the AVERAGE
local delta waiting of that vehicle class on it (per vehicle-pass). Line STYLE
= class (solid car / dashed bus / dotted amb). Two panels fixed | priority.
Legend OUTSIDE plot. Reads analysis/edge_waiting_data.json (no sim)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sumolib

net = sumolib.net.readNet("/home/xiaowen/sumo-rl/nets/dublin/dublin_8std.net.xml")
data = json.load(open("/home/xiaowen/sumo-rl/experiments/analysis/edge_waiting_data.json"))
# 线型差异 + 细线 + 大偏移, 三条不互相遮挡
STYLE=[("car","solid",1.6),("bus",(0,(6,3)),1.8),("ambulance",(0,(1,3)),2.2)]
OFF={"car":0.0,"bus":26.0,"ambulance":-26.0}
edge_shape={e.getID():np.array(e.getShape(),float) for e in net.getEdges()}

# 去掉 -532427444#2 这条边 (该瓶颈交叉口 amb 单点恶化, 按用户要求排除)
DROP_EDGE="-532427444#2"
for s in data:
    for k in [k for k in data[s] if k.split("|")[0]==DROP_EDGE]:
        del data[s][k]
allv=[v for s in data for v in data[s].values()]
vmin,vmax=0.0,float(np.percentile(allv,95)); cmap=plt.cm.RdYlGn_r; norm=plt.Normalize(vmin,vmax)

def off(shape,o):
    if len(shape)<2 or o==0: return shape
    out=[]
    for i in range(len(shape)):
        a=shape[max(0,i-1)]; b=shape[min(len(shape)-1,i+1)]; d=b-a; n=np.array([-d[1],d[0]]); ln=np.hypot(*n)
        out.append(shape[i]+(n/ln*o if ln>1e-6 else 0))
    return np.array(out)

plt.rcParams.update({"font.size":13,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"pdf.fonttype":42})
fig,axes=plt.subplots(1,2,figsize=(17,9.0))
for ax,(key,title) in zip(axes,[("fixed","Fixed-time"),("prio","Priority-RL (exp208, 5:3:1)")]):
    for sh in edge_shape.values():
        ax.plot(sh[:,0],sh[:,1],color="#EAEAEA",lw=0.5,zorder=1)   # 灰底全网
    for cls,ls,lw in STYLE:
        for k,v in data[key].items():
            edge,c=k.split("|")
            if c!=cls or edge not in edge_shape: continue
            sh=off(edge_shape[edge],OFF[cls])
            ax.plot(sh[:,0],sh[:,1],color=cmap(norm(min(v,vmax))),linestyle=ls,lw=lw,
                    zorder=3,solid_capstyle="round",dash_capstyle="round")
    ax.set_title(title,fontsize=15); ax.set_aspect("equal"); ax.axis("off")

sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,ax=axes,fraction=0.025,pad=0.02)
cb.set_label("Avg local waiting per pass on edge (s)  [green=low, red=high]",fontsize=12)
# 图例放图外底部, 不遮挡
leg=[Line2D([0],[0],color="gray",lw=lw,linestyle=ls,label=c.capitalize()) for c,ls,lw in STYLE]
fig.legend(handles=leg,loc="lower center",ncol=3,frameon=False,fontsize=13,
           title="Vehicle class (line style)",bbox_to_anchor=(0.45,-0.01))
fig.suptitle("Dublin 11h: road-level avg local waiting by vehicle class",fontsize=15,y=0.99)
out="/home/xiaowen/sumo-rl/experiments/analysis/fig_edge_waiting"
fig.savefig(out+".png",dpi=200,bbox_inches="tight",pad_inches=0.15); fig.savefig(out+".pdf",bbox_inches="tight",pad_inches=0.15)
plt.close(fig); print(f"SAVED {out}.png (色标 {vmin:.1f}~{vmax:.1f}s, 数据边数 fixed={len(data['fixed'])} prio={len(data['prio'])})")
