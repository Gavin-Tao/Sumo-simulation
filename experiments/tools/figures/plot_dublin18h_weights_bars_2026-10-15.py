"""都柏林 18:00 8STD (12 槽 + F1) 四种优先级权重的分组柱状图 (2026-10-15 用户令: "把这个表格画成带方差的对比图, 一目了然")。
数据: 表图 dublin18h_8std_slot12_f1_weights_best_1s_excl125all_noJ_nohl_mintitle_2026-10-15_en.json (逐秒口径, best 检查点, 9 个评估种子, 四列同剔种子 125)。
布局: 三个指标 (停车时间 / 停车次数 / 平均延误), 每个指标一个主坐标轴 (小汽车 / 公交 / 全部) 加一个窄坐标轴 (救护车, 每集 1 辆, 单独刻度), 每组 4 根柱 = 111 / 521 / 531 / 541;
柱高 = 均值, 误差线 = ± 标准差 (下端截在 0, 这些量非负), 小点 = 各评估种子的值。配色: 111 (无优先级) 灰, 521→531→541 由浅到深的蓝 (公交权重由低到高)。
输出: experiments/analysis/figures/dublin18h_8std_slot12_f1_weights_best_1s_excl125all_bars_2026-10-15_en.{png,pdf}。用法 (仓库根目录): python experiments/tools/figures/plot_dublin18h_weights_bars_2026-10-15.py"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG = "experiments/analysis/figures"
SRC = f"{FIG}/dublin18h_8std_slot12_f1_weights_best_1s_excl125all_noJ_nohl_mintitle_2026-10-15_en.json"
OUT = f"{FIG}/dublin18h_8std_slot12_f1_weights_best_1s_excl125all_bars_2026-10-15_en"
ARMS = ["111", "521", "531", "541"]
COLORS = {"111": "#BDBDBD", "521": "#A9C8E8", "531": "#3775BA", "541": "#0F4D92"}   # 灰 = 无优先级; 浅→深蓝 = 公交权重由低到高
CLASSES = [("car", "Car"), ("bus", "Bus"), ("all", "All")]          # 主坐标轴
AMB = [("ambulance", "Ambulance")]                                    # 单独的窄坐标轴 (每集只有 1 辆救护车, 尺度与噪声都不同)
PANELS = [("avg_stopped_time_per_visit", "Stopped time / visit (s)"), ("avg_stop_events_per_visit", "Stop events / visit"), ("rl_delay", "Average delay / visit (s)")]

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 13, "axes.linewidth": 1.4,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def draw(ax, m, classes, rng, w, offs):
    top = 0.0
    for i, arm in enumerate(ARMS):
        for j, (c, _) in enumerate(classes):
            v = np.array(D[m][c][arm]["per_seed"], float); v = v[np.isfinite(v)]
            mu, sd = float(v.mean()), float(v.std()); x = j + offs[i]
            ax.bar(x, mu, w * 0.92, color=COLORS[arm], edgecolor="black", linewidth=0.8, zorder=2)
            ax.errorbar(x, mu, yerr=[[min(sd, mu)], [sd]], fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=3)
            ax.scatter(x + rng.uniform(-w * 0.22, w * 0.22, len(v)), v, s=9, color="black", alpha=0.55, linewidths=0, zorder=4)
            top = max(top, mu + sd, v.max())
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([lab for _, lab in classes])
    ax.set_ylim(0, top * 1.08 if top > 0 else 1)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=0); ax.set_axisbelow(True)


def main():
    global D
    D = json.load(open(SRC)); rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(19, 5))
    gs = fig.add_gridspec(1, 8, width_ratios=[3, 1.05, 0.45, 3, 1.05, 0.45, 3, 1.05], wspace=0.28)
    w = 0.19; offs = (np.arange(len(ARMS)) - (len(ARMS) - 1) / 2) * w
    for k, (m, ylab) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, 3 * k]); axa = fig.add_subplot(gs[0, 3 * k + 1])
        draw(ax, m, CLASSES, rng, w, offs); draw(axa, m, AMB, rng, w, offs)
        ax.set_ylabel(ylab); axa.set_xlabel("1 vehicle / episode", fontsize=10, color="#555555")
    handles = [Patch(facecolor=COLORS[a], edgecolor="black", label=("111 (no priority)" if a == "111" else a)) for a in ARMS]
    fig.legend(handles=handles, title="Priority weights (ambulance/bus/car)", loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Dublin 18:00, 8STD (12-slot state + F1)", y=1.10, fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print("saved", OUT + ".{png,pdf}")


if __name__ == "__main__":
    main()
