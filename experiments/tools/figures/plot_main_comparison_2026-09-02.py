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

# ---------------- 数据 (尾 40 次 eval, 均值 ± sd). 每个指标: 四类车 + 两个 J (J(531) 带份额, J_eq = 5a+3b+c) ----------------
CLS = ["car", "bus", "amb", "all"]; JC = ["J(531)", "J_eq"]
def M(std, std_e, gs, gs_e, jstd, jstd_e, jgs, jgs_e):
    return dict(std=std, std_e=std_e, gs=gs, gs_e=gs_e, jstd=jstd, jstd_e=jstd_e, jgs=jgs, jgs_e=jgs_e)
SCEN = [
  dict(name="Dublin", exps=("exp208", "exp211"),
    pervisit={"stopped time / visit (s)": M([4.95,1.26,0.90,4.75],[1.24,0.18,0.69,1.17],[2.44,1.06,0.92,2.38],[0.32,0.14,0.48,0.31],[4.89,13.21],[1.18,4.05],[2.48,10.22],[0.31,2.45]),
              "stop events / visit":       M([0.57,0.18,0.14,0.55],[0.09,0.02,0.10,0.09],[0.27,0.15,0.15,0.27],[0.03,0.02,0.07,0.03],[0.57,1.81],[0.09,0.59],[0.28,1.44],[0.03,0.38])},
    xts={"xts stopped time (s)": M([3.10,1.71,1.00,3.01],[0.42,0.30,0.78,0.40],[2.43,1.85,1.27,2.36],[0.35,0.29,0.64,0.33],[3.20,13.21],[0.40,4.17],[2.58,14.32],[0.33,3.19]),
         "xts stop events":      M([0.38,0.20,0.15,0.37],[0.03,0.02,0.11,0.03],[0.28,0.21,0.19,0.27],[0.04,0.02,0.09,0.04],[0.39,1.74],[0.03,0.58],[0.30,1.90],[0.04,0.45]),
         "xts speed (m/s)":      M([5.38,6.34,6.38,5.46],[0.10,0.10,0.41,0.09],[5.69,6.40,6.39,5.77],[0.18,0.09,0.39,0.17],[6.07,56.30],[0.10,2.19],[6.37,56.86],[0.17,2.06])}),
  dict(name="1x1", exps=("exp274", "exp263"),
    pervisit={"stopped time / visit (s)": M([32.40,7.46,5.38,31.06],[2.19,1.32,2.37,2.06],[31.30,7.25,4.50,30.00],[2.19,1.10,2.84,2.11],[32.06,81.67],[2.10,13.73],[30.97,75.54],[2.19,15.59]),
              "stop events / visit":       M([1.13,0.85,0.49,1.11],[0.05,0.08,0.14,0.05],[1.11,0.79,0.47,1.10],[0.07,0.07,0.16,0.07],[1.18,6.12],[0.05,0.76],[1.16,5.86],[0.07,0.80])},
    xts={"xts stopped time (s)": M([32.40,7.46,5.38,31.06],[2.19,1.32,2.37,2.06],[31.30,7.25,4.50,30.00],[2.19,1.10,2.84,2.11],[32.06,81.67],[2.10,13.73],[30.97,75.54],[2.19,15.59]),
         "xts stop events":      M([1.13,0.85,0.49,1.11],[0.05,0.08,0.14,0.05],[1.11,0.79,0.47,1.10],[0.07,0.07,0.16,0.07],[1.18,6.12],[0.05,0.76],[1.16,5.86],[0.07,0.80]),
         "xts speed (m/s)":      M([3.49,6.99,8.54,3.59],[0.17,0.39,1.08,0.17],[3.57,7.11,8.32,3.67],[0.18,0.31,1.32,0.18],[4.14,67.15],[0.16,5.87],[4.23,66.51],[0.19,6.56])}),
  dict(name="1x3", exps=("exp244", "exp242"),
    pervisit={"stopped time / visit (s)": M([29.45,3.34,2.77,27.42],[1.32,0.54,1.79,1.22],[30.77,5.92,4.02,28.83],[1.28,0.34,2.10,1.19],[28.48,53.32],[1.26,10.02],[30.11,68.62],[1.24,10.87]),
              "stop events / visit":       M([1.10,0.49,0.37,1.05],[0.04,0.07,0.23,0.04],[1.20,0.76,0.51,1.17],[0.04,0.04,0.21,0.04],[1.12,4.39],[0.04,1.25],[1.26,6.02],[0.04,1.05])},
    xts={"xts stopped time (s)": M([31.94,3.34,2.77,30.36],[1.49,0.54,1.79,1.40],[34.22,5.92,4.02,32.64],[1.44,0.34,2.10,1.37],[30.84,55.81],[1.42,10.01],[33.40,72.08],[1.39,10.86]),
         "xts stop events":      M([1.14,0.49,0.37,1.11],[0.04,0.07,0.23,0.04],[1.25,0.76,0.51,1.22],[0.04,0.04,0.21,0.04],[1.16,4.44],[0.04,1.25],[1.31,6.07],[0.04,1.05]),
         "xts speed (m/s)":      M([3.40,8.68,8.78,3.53],[0.11,0.33,1.16,0.11],[3.22,7.48,7.79,3.33],[0.10,0.13,0.95,0.10],[4.55,73.36],[0.12,6.24],[4.19,64.61],[0.10,4.80])}),
]

def panel_group(sc, metrics, suffix):
    """主图: 每个指标一个面板, 类别 = car/bus/ambulance/all + J(531) (与之前版本一致)."""
    colors = [PALETTE["red_strong"], PALETTE["blue_main"]]
    labels = [f"8STD ({sc['exps'][0]})", f"GS-ENUM ({sc['exps'][1]})"]
    cats = ["car", "bus", "ambulance", "all", "J(531)"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 5.8)); axes = np.atleast_1d(axes)
    for ax, (mname, d) in zip(axes, metrics.items()):
        make_grouped_bar_err(ax, cats, [d["std"] + [d["jstd"][0]], d["gs"] + [d["jgs"][0]]],
                             [d["std_e"] + [d["jstd_e"][0]], d["gs_e"] + [d["jgs_e"][0]]], labels, colors, ylabel=mname)
        ax.axvline(3.5, ymax=0.85, color=PALETTE["neutral"], linewidth=1.2, linestyle="--", zorder=0)
    axes[0].set_title(sc["name"], loc="left")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return finalize_figure(fig, f"experiments/analysis/figures/{sc['name'].lower()}_{suffix}", formats=("png", "pdf"))

def jeq_group(sc, metrics, suffix):
    """J_eq 单独成图: 每个指标一个面板, 两根柱 (8STD / GS-ENUM), J_eq = 5·amb + 3·bus + 1·car."""
    colors = [PALETTE["red_strong"], PALETTE["blue_main"]]
    labels = [f"8STD ({sc['exps'][0]})", f"GS-ENUM ({sc['exps'][1]})"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.4)); axes = np.atleast_1d(axes)
    for ax, (mname, d) in zip(axes, metrics.items()):
        make_grouped_bar_err(ax, ["J_eq"], [[d["jstd"][1]], [d["jgs"][1]]], [[d["jstd_e"][1]], [d["jgs_e"][1]]],
                             labels, colors, ylabel="J_eq of " + mname.split(" (")[0])
        ax.set_xlim(-0.7, 0.7); ax.set_xticks([])
    axes[0].set_title(sc["name"], loc="left")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return finalize_figure(fig, f"experiments/analysis/figures/{sc['name'].lower()}_jeq_{suffix}", formats=("png", "pdf"))

def main():
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2))
    for sc in SCEN:
        print("saved:", *panel_group(sc, sc["pervisit"], "pervisit"))
        print("saved:", *panel_group(sc, sc["xts"], "xts"))
        print("saved:", *jeq_group(sc, sc["pervisit"], "pervisit"))
        print("saved:", *jeq_group(sc, sc["xts"], "xts"))

if __name__ == "__main__":
    main()
