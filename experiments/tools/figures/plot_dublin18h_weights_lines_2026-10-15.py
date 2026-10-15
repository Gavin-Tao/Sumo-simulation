"""都柏林 18:00 8STD (12 槽 + F1) 指标随优先级设定变化的折线图 (2026-10-15 用户令: "是不是得是线图, car bus amb all 随着优先级变化的曲线")。
数据: 表图 dublin18h_8std_slot12_f1_weights_best_1s_excl125all_noJ_nohl_mintitle_2026-10-15_en.json (逐秒口径, best 检查点, 9 个评估种子, 四列同剔种子 125)。
布局: 三个小图 (停车时间 / 停车次数 / 平均延误); 横轴 = 优先级设定 111 (无优先级) → 521 → 531 → 541 (公交权重由低到高);
四条线 = 小汽车 / 公交 / 救护车 / 全部, 点 = 均值, 竖线 = ± 标准差 (下端截在 0), 同一横轴位置四条线左右错开避免重叠;
111 → 521 用虚线 (111 的奖励是全体车辆平均等待, 不只是权重不同), 521 → 531 → 541 用实线 (只改公交权重)。
输出: experiments/analysis/figures/dublin18h_8std_slot12_f1_weights_best_1s_excl125all_lines_2026-10-15_en.{png,pdf}。用法 (仓库根目录): python experiments/tools/figures/plot_dublin18h_weights_lines_2026-10-15.py"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG = "experiments/analysis/figures"
SRC = f"{FIG}/dublin18h_8std_slot12_f1_weights_best_1s_excl125all_noJ_nohl_mintitle_2026-10-15_en.json"
OUT = f"{FIG}/dublin18h_8std_slot12_f1_weights_best_1s_excl125all_lines_2026-10-15_en"
ARMS = ["111", "521", "531", "541"]
XLAB = ["111\n(no priority)", "521", "531", "541"]
SERIES = [("car", "Car", "#B64342", "o"), ("bus", "Bus", "#0F4D92", "s"), ("ambulance", "Ambulance (1 vehicle / episode)", "#42949E", "^"), ("all", "All vehicles", "#555555", "D")]
PANELS = [("avg_stopped_time_per_visit", "Stopped time / visit (s)"), ("avg_stop_events_per_visit", "Stop events / visit"), ("rl_delay", "Average delay / visit (s)")]

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 13, "axes.linewidth": 1.4,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def main():
    D = json.load(open(SRC))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    x0 = np.arange(len(ARMS)); dodge = (np.arange(len(SERIES)) - (len(SERIES) - 1) / 2) * 0.07
    for ax, (m, ylab) in zip(axes, PANELS):
        for k, (c, lab, col, mk) in enumerate(SERIES):
            mu = np.array([np.nanmean(D[m][c][a]["per_seed"]) for a in ARMS]); sd = np.array([np.nanstd(D[m][c][a]["per_seed"]) for a in ARMS])
            x = x0 + dodge[k]
            ax.plot(x[:2], mu[:2], ls="--", color=col, lw=1.6, alpha=0.8, zorder=2)          # 111 → 521: 奖励形式不同, 虚线
            ax.plot(x[1:], mu[1:], ls="-", color=col, lw=2.2, zorder=2)                       # 521 → 531 → 541: 只改公交权重, 实线
            ax.errorbar(x, mu, yerr=[np.minimum(sd, mu), sd], fmt=mk, color=col, ms=7, mec="black", mew=0.6,
                        ecolor=col, elinewidth=1.3, capsize=3.5, zorder=3)
        ax.set_xticks(x0); ax.set_xticklabels(XLAB); ax.set_ylabel(ylab); ax.set_ylim(bottom=0)
        ax.set_xlim(-0.4, len(ARMS) - 0.6)
        ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    axes[1].set_xlabel("Priority weights (ambulance/bus/car)")
    handles = [Line2D([0], [0], color=col, marker=mk, ms=7, mec="black", mew=0.6, lw=2.2, label=lab) for _, lab, col, mk in SERIES]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Dublin 18:00, 8STD (12-slot state + F1)", y=1.10, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print("saved", OUT + ".{png,pdf}")


if __name__ == "__main__":
    main()
