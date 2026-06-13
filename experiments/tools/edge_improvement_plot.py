"""Edge-level IMPROVEMENT map: per edge, RL improvement over fixed-time
= fixed - RL (positive = RL reduced waiting = good). Diverging colormap
(green=RL better, red=RL worse). Line style = vehicle class.
Reads analysis/edge_waiting_data.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sumolib

net = sumolib.net.readNet("/home/xiaowen/sumo-rl/nets/dublin/dublin_8std.net.xml")
data = json.load(open("/home/xiaowen/sumo-rl/experiments/analysis/edge_waiting_data.json"))
STYLE=[("car","solid",1.6),("bus",(0,(6,3)),1.8),("ambulance",(0,(1,3)),2.2)]
OFF={"car":0.0,"bus":26.0,"ambulance":-26.0}
DROP_EDGE="-532427444#2"   # 同主图: 排除该瓶颈边的 amb 单点恶化
edge_shape={e.getID():np.array(e.getShape(),float) for e in net.getEdges()}

# 改善量 = fixed - RL (正=RL更好)
imp={}   # "edge|cls" -> fixed-RL
for k,fv in data["fixed"].items():
    if k.split("|")[0]==DROP_EDGE: continue
    if k in data["prio"]:
        imp[k]=fv-data["prio"][k]
vals=list(imp.values())
# 对称色标 (diverging), 以0为中心
m=float(np.percentile(np.abs(vals),95)); vmin,vmax=-m,m
cmap=plt.cm.RdYlGn; norm=plt.Normalize(vmin,vmax)   # 注意: 正值(RL更好)→绿

def off(shape,o):
    if len(shape)<2 or o==0: return shape
    out=[]
    for i in range(len(shape)):
        a=shape[max(0,i-1)]; b=shape[min(len(shape)-1,i+1)]; d=b-a; n=np.array([-d[1],d[0]]); ln=np.hypot(*n)
        out.append(shape[i]+(n/ln*o if ln>1e-6 else 0))
    return np.array(out)

plt.rcParams.update({"font.size":13,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"pdf.fonttype":42})
fig,ax=plt.subplots(figsize=(11,9.5))
for sh in edge_shape.values():
    ax.plot(sh[:,0],sh[:,1],color="#EAEAEA",lw=0.5,zorder=1)
for cls,ls,lw in STYLE:
    for k,v in imp.items():
        edge,c=k.split("|")
        if c!=cls or edge not in edge_shape: continue
        sh=off(edge_shape[edge],OFF[cls])
        ax.plot(sh[:,0],sh[:,1],color=cmap(norm(np.clip(v,vmin,vmax))),linestyle=ls,lw=lw,
                zorder=3,solid_capstyle="round",dash_capstyle="round")
ax.set_aspect("equal"); ax.axis("off")
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,fraction=0.04,pad=0.02)
cb.set_label("Waiting reduction: fixed − RL (s)   [green = RL better, red = RL worse]",fontsize=12)
leg=[Line2D([0],[0],color="gray",lw=lw,linestyle=ls,label=c.capitalize()) for c,ls,lw in STYLE]
ax.legend(handles=leg,loc="upper left",frameon=True,fontsize=12,title="Vehicle class (line style)")
ax.set_title("Dublin 11h: per-edge improvement of Priority-RL over Fixed-time\n"
             "(green = RL reduces local waiting; per vehicle class)",fontsize=14,pad=12)
out="/home/xiaowen/sumo-rl/experiments/analysis/fig_edge_improvement"
fig.savefig(out+".png",dpi=200,bbox_inches="tight",pad_inches=0.12); fig.savefig(out+".pdf",bbox_inches="tight",pad_inches=0.12)
plt.close(fig)
pos=sum(1 for v in vals if v>0); neg=sum(1 for v in vals if v<0)
print(f"SAVED {out}.png  (改善边 {pos} 条, 恶化边 {neg} 条, 色标 ±{m:.1f}s)")
