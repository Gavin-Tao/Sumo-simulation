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

# ---------------- 数据 (尾 40 次 eval; per-visit 停车时间 s; J(531) per-visit) ----------------
CATS = ["car", "bus", "ambulance", "all", "J(531)"]
SCEN = [
    dict(name="Dublin", file="main_comparison_dublin", exps=("exp208", "exp211"),
         std=[4.95, 1.26, 0.90, 4.75, 4.89], std_e=[1.24, 0.18, 0.69, 1.17, 1.18],
         gs=[2.44, 1.06, 0.92, 2.38, 2.48],  gs_e=[0.32, 0.14, 0.48, 0.31, 0.31]),
    dict(name="1x1", file="main_comparison_1x1", exps=("exp274", "exp263"),
         std=[32.40, 7.46, 5.38, 31.06, 32.06], std_e=[2.19, 1.32, 2.37, 2.06, 2.10],
         gs=[31.30, 7.25, 4.50, 30.00, 30.97],  gs_e=[2.19, 1.10, 2.84, 2.11, 2.19]),
    dict(name="1x3", file="main_comparison_1x3", exps=("exp244", "exp242"),
         std=[29.45, 3.34, 2.77, 27.42, 28.48], std_e=[1.32, 0.54, 1.79, 1.22, 1.26],
         gs=[30.77, 5.92, 4.02, 28.83, 30.11],  gs_e=[1.28, 0.34, 2.10, 1.19, 1.24]),
]

def main():
    apply_publication_style(FigureStyle(font_size=16, axes_linewidth=2))
    colors = [PALETTE["red_strong"], PALETTE["blue_main"]]
    for sc in SCEN:
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        make_grouped_bar_err(ax, CATS, [sc["std"], sc["gs"]], [sc["std_e"], sc["gs_e"]], list(sc["exps"]), colors,
                             ylabel="stopped time per visit (s) / J")
        ax.set_title(sc["name"], loc="left")
        ax.axvline(3.5, ymax=0.85, color=PALETTE["neutral"], linewidth=1.2, linestyle="--", zorder=0)
        ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        out = finalize_figure(fig, f"experiments/analysis/figures/{sc['file']}", formats=("png", "pdf"))
        print("saved:", *out)

if __name__ == "__main__":
    main()
