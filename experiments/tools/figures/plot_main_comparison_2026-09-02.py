"""三场景主方法对比图 (8std template + DQN vs enum + FRAP g/s), 论文风格 (scientific-figure-making 规范).
数据 = wandb 尾 40 次 eval 均值 ± 标准差 (per-visit 停车时间, J(531) per-visit, 配对胜率, 保序门)。
运行: python experiments/tools/figures/plot_main_comparison_2026-09-02.py  → experiments/analysis/figures/main_comparison_8std_vs_frap.{png,pdf}
"""
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B",
           "red_strong": "#B64342", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E"}

@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")

def apply_publication_style(style=FigureStyle()):
    plt.rcParams.update({
        "font.family": list(style.font_family), "font.size": style.font_size,
        "axes.linewidth": style.axes_linewidth, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "xtick.major.width": style.axes_linewidth * 0.6,
        "ytick.major.width": style.axes_linewidth * 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False})

def make_grouped_bar_err(ax, categories, series, errors, labels, colors, ylabel=None, annotate=True, fmt="{:.2f}"):
    """分组柱 + 误差棒 + 数值标注 (标注放在误差棒顶端之上, 不遮挡)。"""
    n = len(series); x = np.arange(len(categories)); width = 0.8 / n
    tops = []
    for i, (s, e, lab, col) in enumerate(zip(series, errors, labels, colors)):
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, s, width * 0.92, yerr=e, capsize=4, label=lab, color=col,
                      edgecolor="black", linewidth=1.5, error_kw=dict(elinewidth=1.5, ecolor="black"))
        if annotate:
            for b, v, err in zip(bars, s, e):
                ax.text(b.get_x() + b.get_width() / 2, v + err, fmt.format(v), ha="center", va="bottom",
                        fontsize=plt.rcParams["font.size"] * 0.62, rotation=0, clip_on=False)
        tops.extend(np.asarray(s) + np.asarray(e))
    ax.set_xticks(x); ax.set_xticklabels(categories)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(tops) * 1.32)   # 顶部余量: 放数值标注 + 面板内注记, 互不遮挡
    return bars

def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300, pad=0.05):
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True); saved = []
    for f in formats:
        p = out.with_suffix("." + f); fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=pad); saved.append(p)
    plt.close(fig); return saved

# ---------------- 数据 (尾 40 次 eval, 均值 ± sd; 每个指标附各自的 J(531)) ----------------
CATS = ["car", "bus", "ambulance", "all", "J(531)"]
def M(std, std_e, gs, gs_e): return dict(std=std, std_e=std_e, gs=gs, gs_e=gs_e)
SCEN = [
  dict(name="Dublin", exps=("exp208", "exp211"),
    pervisit={"stopped time / visit (s)": M([4.95,1.26,0.90,4.75,4.89],[1.24,0.18,0.69,1.17,1.18],[2.44,1.06,0.92,2.38,2.48],[0.32,0.14,0.48,0.31,0.31]),
              "stop events / visit":       M([0.57,0.18,0.14,0.55,0.57],[0.09,0.02,0.10,0.09,0.09],[0.27,0.15,0.15,0.27,0.28],[0.03,0.02,0.07,0.03,0.03])},
    xts={"xts stopped time (s)": M([3.10,1.71,1.00,3.01,3.20],[0.42,0.30,0.78,0.40,0.40],[2.43,1.85,1.27,2.36,2.58],[0.35,0.29,0.64,0.33,0.33]),
         "xts stop events":      M([0.38,0.20,0.15,0.37,0.39],[0.03,0.02,0.11,0.03,0.03],[0.28,0.21,0.19,0.27,0.30],[0.04,0.02,0.09,0.04,0.04]),
         "xts speed (m/s)":      M([5.38,6.34,6.38,5.46,6.07],[0.10,0.10,0.41,0.09,0.10],[5.69,6.40,6.39,5.77,6.37],[0.18,0.09,0.39,0.17,0.17])}),
  dict(name="1x1", exps=("exp274", "exp263"),
    pervisit={"stopped time / visit (s)": M([32.40,7.46,5.38,31.06,32.06],[2.19,1.32,2.37,2.06,2.10],[31.30,7.25,4.50,30.00,30.97],[2.19,1.10,2.84,2.11,2.19]),
              "stop events / visit":       M([1.13,0.85,0.49,1.11,1.18],[0.05,0.08,0.14,0.05,0.05],[1.11,0.79,0.47,1.10,1.16],[0.07,0.07,0.16,0.07,0.07])},
    xts={"xts stopped time (s)": M([32.40,7.46,5.38,31.06,32.06],[2.19,1.32,2.37,2.06,2.10],[31.30,7.25,4.50,30.00,30.97],[2.19,1.10,2.84,2.11,2.19]),
         "xts stop events":      M([1.13,0.85,0.49,1.11,1.18],[0.05,0.08,0.14,0.05,0.05],[1.11,0.79,0.47,1.10,1.16],[0.07,0.07,0.16,0.07,0.07]),
         "xts speed (m/s)":      M([3.49,6.99,8.54,3.59,4.14],[0.17,0.39,1.08,0.17,0.16],[3.57,7.11,8.32,3.67,4.23],[0.18,0.31,1.32,0.18,0.19])}),
  dict(name="1x3", exps=("exp244", "exp242"),
    pervisit={"stopped time / visit (s)": M([29.45,3.34,2.77,27.42,28.48],[1.32,0.54,1.79,1.22,1.26],[30.77,5.92,4.02,28.83,30.11],[1.28,0.34,2.10,1.19,1.24]),
              "stop events / visit":       M([1.10,0.49,0.37,1.05,1.12],[0.04,0.07,0.23,0.04,0.04],[1.20,0.76,0.51,1.17,1.26],[0.04,0.04,0.21,0.04,0.04])},
    xts={"xts stopped time (s)": M([31.94,3.34,2.77,30.36,30.84],[1.49,0.54,1.79,1.40,1.42],[34.22,5.92,4.02,32.64,33.40],[1.44,0.34,2.10,1.37,1.39]),
         "xts stop events":      M([1.14,0.49,0.37,1.11,1.16],[0.04,0.07,0.23,0.04,0.04],[1.25,0.76,0.51,1.22,1.31],[0.04,0.04,0.21,0.04,0.04]),
         "xts speed (m/s)":      M([3.40,8.68,8.78,3.53,4.55],[0.11,0.33,1.16,0.11,0.12],[3.22,7.48,7.79,3.33,4.19],[0.10,0.13,0.95,0.10,0.10])}),
]

def panel_group(sc, metrics, suffix):
    colors = [PALETTE["red_strong"], PALETTE["blue_main"]]
    labels = [f"8STD ({sc['exps'][0]})", f"GS-ENUM ({sc['exps'][1]})"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 5.8))
    axes = np.atleast_1d(axes)
    for ax, (mname, d) in zip(axes, metrics.items()):
        make_grouped_bar_err(ax, CATS, [d["std"], d["gs"]], [d["std_e"], d["gs_e"]], labels, colors, ylabel=mname)
        ax.axvline(3.5, ymax=0.85, color=PALETTE["neutral"], linewidth=1.2, linestyle="--", zorder=0)
    axes[0].set_title(sc["name"], loc="left")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return finalize_figure(fig, f"experiments/analysis/figures/{sc['name'].lower()}_{suffix}", formats=("png", "pdf"))

def main():
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2))
    for sc in SCEN:
        print("saved:", *panel_group(sc, sc["pervisit"], "pervisit"))
        print("saved:", *panel_group(sc, sc["xts"], "xts"))

if __name__ == "__main__":
    main()
