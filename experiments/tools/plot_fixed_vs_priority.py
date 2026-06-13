"""Grouped bar: fixed-time vs priority(exp208), per class, stopped_time/visit.
Broken y-axis (fixed-time ~18-22s dwarfs exp208 ~1-5s), value labels on bars,
error bars (exp208 tail-20 std; fixed-time 3-seed std). Publication style."""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"blue_main":"#0F4D92","red_strong":"#B64342","neutral":"#CFCECE"}

# ---- data: avg_stopped_time_per_visit (fair, shows inversion) ----
CLS = ["car", "bus", "ambulance"]
fixed_rows = json.load(open("/tmp/fixed_rows.json"))
def fx(cls, m="avg_stopped_time_per_visit"):
    a = np.array([r.get(f"eval/system/{cls}/{m}", np.nan) for r in fixed_rows], float)
    a = a[~np.isnan(a)]; return a.mean(), a.std()
ers = list(csv.DictReader(open("/home/xiaowen/sumo-rl/experiments/analysis/dublin_2x2_extracts/exp208_eval.csv")))[-20:]
def rl(cls, m="avg_stopped_time_per_visit"):
    a = np.array([float(r[f"eval_system/{cls}/{m}"]) for r in ers if r.get(f"eval_system/{cls}/{m}","")])
    return a.mean(), a.std()

fixed_mean = [fx(c)[0] for c in CLS]; fixed_std = [fx(c)[1] for c in CLS]
prio_mean  = [rl(c)[0] for c in CLS]; prio_std  = [rl(c)[1] for c in CLS]

# ---- publication style ----
plt.rcParams.update({
    "font.size": 17, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans","Helvetica","Arial"],
    "axes.linewidth": 2.0, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

x = np.arange(len(CLS)); w = 0.38
# broken axis: low panel 0-7 (priority + lets bus/car fixed show), high panel 12-25 (fixed)
fig, (axtop, axbot) = plt.subplots(2, 1, sharex=True, figsize=(8.2, 6.4),
                                   gridspec_kw={"height_ratios":[1, 1.6], "hspace":0.08})
LOW_MAX, HIGH_MIN, HIGH_MAX = 7.0, 11.0, 25.0

def draw(ax):
    b1 = ax.bar(x-w/2, fixed_mean, w, yerr=fixed_std, capsize=5,
                color=PALETTE["red_strong"], edgecolor="black", linewidth=1.2,
                error_kw={"elinewidth":1.5,"ecolor":"#333"}, label="Fixed-time")
    b2 = ax.bar(x+w/2, prio_mean, w, yerr=prio_std, capsize=5,
                color=PALETTE["blue_main"], edgecolor="black", linewidth=1.2,
                error_kw={"elinewidth":1.5,"ecolor":"#333"}, label="Priority-RL (exp208, Amb:Bus:Car=5:3:1)")
    return b1, b2
b1, b2 = draw(axtop); draw(axbot)

axtop.set_ylim(HIGH_MIN, HIGH_MAX+3)    # 高：fixed-time (留顶部空间给图例/标注)
axbot.set_ylim(0, LOW_MAX)              # 低：priority + fixed 的 bus
axtop.spines["bottom"].set_visible(False)
axbot.spines["top"].set_visible(False)
axtop.tick_params(bottom=False)

# broken-axis 斜线标记
d = .008
for ax, y0 in [(axtop,0),(axbot,1)]:
    kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1.5)
    ax.plot((-d,+d),(y0-d*2.0,y0+d*2.0),**kw); ax.plot((1-d,1+d),(y0-d*2.0,y0+d*2.0),**kw)

# 柱顶数值标注 (每根柱在它所属面板里标)
def label(ax, bars, means, sds, ymin, ymax):
    for rect, m, sd in zip(bars, means, sds):
        h = rect.get_height()
        if ymin <= h <= ymax:
            ax.annotate(f"{m:.2f}±{sd:.2f}", (rect.get_x()+rect.get_width()/2, h),
                        xytext=(0,4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
for ax in (axtop, axbot):
    ymn, ymx = ax.get_ylim()
    label(ax, b1, fixed_mean, fixed_std, ymn, ymx); label(ax, b2, prio_mean, prio_std, ymn, ymx)

axbot.set_xticks(x); axbot.set_xticklabels(["Car","Bus","Ambulance"])
axbot.set_ylabel("Avg stopped time per intersection (s)", fontsize=15)
axbot.yaxis.set_label_coords(-0.085, 1.0)
axtop.legend(loc="upper left", frameon=False, fontsize=13, ncol=2,
             bbox_to_anchor=(0.0, 1.02))
axtop.set_title("Dublin 11h: avg stopped time per intersection (lower = better)",
                fontsize=14, pad=10)

out = "/home/xiaowen/sumo-rl/experiments/analysis/fig_fixed_vs_priority_pervisit"
fig.savefig(out+".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
fig.savefig(out+".pdf", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("saved", out+".png / .pdf")
print("fixed:", [f"{m:.2f}±{s:.1f}" for m,s in zip(fixed_mean,fixed_std)])
print("prio :", [f"{m:.2f}±{s:.1f}" for m,s in zip(prio_mean,prio_std)])
