"""Edge local-waiting map, 3 rows (car/bus/amb) x 2 cols (fixed/RL).
Each subplot: only that class's edges, colored by avg local waiting.
Shared colorbar. Reads analysis/edge_waiting_data.json."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sumolib

net = sumolib.net.readNet("/home/xiaowen/sumo-rl/nets/dublin/dublin_8std.net.xml")
data = json.load(open("/home/xiaowen/sumo-rl/experiments/analysis/edge_waiting_data.json"))
DROP_EDGE="-532427444#2"
for s in data:
    for k in [k for k in data[s] if k.split("|")[0]==DROP_EDGE]:
        del data[s][k]
edge_shape={e.getID():np.array(e.getShape(),float) for e in net.getEdges()}
CLASSES=["car","bus","ambulance"]
COLS=[("fixed","Fixed-time"),("prio","Priority-RL (5:3:1)")]

allv=[v for s in data for v in data[s].values()]
vmin,vmax=0.0,float(np.percentile(allv,95)); cmap=plt.cm.RdYlGn_r; norm=plt.Normalize(vmin,vmax)

plt.rcParams.update({"font.size":12,"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"pdf.fonttype":42})
fig,axes=plt.subplots(3,2,figsize=(12,15))
for r,cls in enumerate(CLASSES):
    for c,(key,coltitle) in enumerate(COLS):
        ax=axes[r,c]
        for sh in edge_shape.values():
            ax.plot(sh[:,0],sh[:,1],color="#ECECEC",lw=0.4,zorder=1)
        for k,v in data[key].items():
            edge,cc=k.split("|")
            if cc!=cls or edge not in edge_shape: continue
            sh=edge_shape[edge]
            ax.plot(sh[:,0],sh[:,1],color=cmap(norm(min(v,vmax))),lw=2.6,
                    zorder=3,solid_capstyle="round")
        ax.set_aspect("equal"); ax.axis("off")
        if r==0: ax.set_title(coltitle,fontsize=14,pad=8)
        if c==0: ax.text(-0.05,0.5,cls.capitalize(),transform=ax.transAxes,
                         fontsize=15,fontweight="bold",rotation=90,va="center",ha="center")
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cb=fig.colorbar(sm,ax=axes,fraction=0.025,pad=0.02)
cb.set_label("Avg local waiting per pass on edge (s)  [green=low, red=high]",fontsize=12)
fig.suptitle("Dublin 11h: road-level local waiting by vehicle class (per-class panels)",fontsize=15,y=0.995)
out="/home/xiaowen/sumo-rl/experiments/analysis/fig_edge_waiting_byclass"
fig.savefig(out+".png",dpi=200,bbox_inches="tight",pad_inches=0.12); fig.savefig(out+".pdf",bbox_inches="tight",pad_inches=0.12)
plt.close(fig); print(f"SAVED {out}.png (色标 {vmin:.1f}~{vmax:.1f}s)")
