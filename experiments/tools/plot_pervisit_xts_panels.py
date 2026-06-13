"""5-panel grouped bars: per_visit (2 metrics) + xts (3 metrics), one figure.
car/bus/amb x {fixed-time, priority-RL exp208 5:3:1}, error bars, value labels."""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {"fixed":"#B64342", "prio":"#0F4D92"}
CLS = ["car","bus","ambulance"]; XL = ["Car","Bus","Amb"]
fixed_rows = json.load(open("/tmp/fixed_rows.json"))
ers = list(csv.DictReader(open("/home/xiaowen/sumo-rl/experiments/analysis/dublin_2x2_extracts/exp208_eval.csv")))[-20:]
def fx(cls,m):
    a=np.array([r.get(f"eval/system/{cls}/{m}",np.nan) for r in fixed_rows],float); a=a[~np.isnan(a)]
    return (a.mean(),a.std()) if len(a) else (np.nan,np.nan)
def rl(cls,m):
    a=np.array([float(r[f"eval_system/{cls}/{m}"]) for r in ers if r.get(f"eval_system/{cls}/{m}","")])
    return (a.mean(),a.std()) if len(a) else (np.nan,np.nan)

PANELS = [
    ("avg_stopped_time_per_visit",  "Stopped time / intersection (s)",  True),
    ("avg_stop_events_per_visit",   "Stop events / intersection",       True),
    ("xts_avg_speed",               "xTS avg speed (m/s)",              False),
    ("xts_avg_stopped_time",        "xTS stopped time (s)",             True),
    ("xts_avg_stop_events",         "xTS stop events",                  True),
]
plt.rcParams.update({"font.size":14,"font.family":"sans-serif",
    "font.sans-serif":["DejaVu Sans","Helvetica","Arial"],"axes.linewidth":1.8,
    "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5)); axes = axes.flatten()
x = np.arange(len(CLS)); w = 0.38; hb1=hb2=None
for i,(m,title,lower) in enumerate(PANELS):
    ax = axes[i]
    fm=[fx(c,m)[0] for c in CLS]; fs=[fx(c,m)[1] for c in CLS]
    rm=[rl(c,m)[0] for c in CLS]; rs=[rl(c,m)[1] for c in CLS]
    b1=ax.bar(x-w/2,fm,w,yerr=fs,capsize=4,color=PAL["fixed"],edgecolor="black",
              linewidth=1.0,error_kw={"elinewidth":1.3,"ecolor":"#333"})
    b2=ax.bar(x+w/2,rm,w,yerr=rs,capsize=4,color=PAL["prio"],edgecolor="black",
              linewidth=1.0,error_kw={"elinewidth":1.3,"ecolor":"#333"})
    hb1,hb2=b1,b2
    for bars,means,sds in ((b1,fm,fs),(b2,rm,rs)):
        for rect,v,sd in zip(bars,means,sds):
            if np.isnan(v): continue
            ax.annotate(f"{v:.2f}\n±{sd:.2f}",(rect.get_x()+rect.get_width()/2,rect.get_height()+ (max([z for z in fm+rm if not np.isnan(z)])*0.01)),
                        xytext=(0,2),textcoords="offset points",ha="center",va="bottom",
                        fontsize=9.5,fontweight="bold",linespacing=0.9)
    ax.set_ylim(0, max([v for v in fm+rm if not np.isnan(v)])*1.40)
    ax.set_title(f"{title}\n({'lower better' if lower else 'higher better'})", fontsize=12.5, pad=6)
    ax.set_xticks(x); ax.set_xticklabels(XL)

axes[5].set_axis_off()
axes[5].legend(handles=[hb1,hb2],
               labels=["Fixed-time","Priority-RL (exp208)\nAmb:Bus:Car = 5:3:1"],
               loc="center", frameon=False, fontsize=15,
               title="Dublin 11h\nper-visit + xTS metrics", title_fontsize=15)
fig.suptitle("Dublin 11h: per-visit & cross-TS metrics — Fixed-time vs Priority-RL (5:3:1)",
             fontsize=15, y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
out="/home/xiaowen/sumo-rl/experiments/analysis/fig_pervisit_xts_5panel"
fig.savefig(out+".png",dpi=300,bbox_inches="tight",pad_inches=0.06)
fig.savefig(out+".pdf",bbox_inches="tight",pad_inches=0.06)
plt.close(fig); print("saved",out)
