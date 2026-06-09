"""1x1 PER ablation: vanilla DQN / +PER / +PER+Double on 5-2-1 NSamb.

3-agent bar comparison (single weight, single intersection).
Output: figures/1x1_per_ablation/01_per_ablation_<scope>.png

Window=20 evals (matched to user's wandb view), MAD k=3.5 outlier filter.
"""
from __future__ import annotations
import argparse
import os
import struct
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from wandb.proto import wandb_internal_pb2 as pb

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    "red_strong": "#B64342",   # vanilla DQN
    "orange":     "#E07B00",   # +Double (alone)
    "purple":     "#7B3F99",   # +PER (alone)
    "green_teal": "#2A8C7C",   # +PER+Double
    "grid":       "#CFCECE",
}
COLOR_V  = PALETTE["red_strong"]
COLOR_D  = PALETTE["orange"]
COLOR_P  = PALETTE["purple"]
COLOR_PD = PALETTE["green_teal"]

RC = {
    "font.family":       ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size":         14,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.linewidth":    1.6,
    "axes.labelsize":    13,
    "axes.titlesize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.frameon":    False,
    "legend.fontsize":   13,
    "svg.fonttype":      "none",
}

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)

RUN_BASE = "logs/wandb/wandb"
RUNS = {
    "vanilla":     f"{RUN_BASE}/run-20260427_214845-6eahys1g",  # exp113
    "+Double":     f"{RUN_BASE}/run-20260608_184109-oyp4neal",  # exp142
    "+PER":        f"{RUN_BASE}/run-20260608_184124-ai2ikpww",  # exp143
    "+PER+Double": f"{RUN_BASE}/run-20260608_184139-lb0eur84",  # exp144
}

AGENTS = list(RUNS.keys())
AGENT_COLOR = {"vanilla": COLOR_V, "+Double": COLOR_D, "+PER": COLOR_P, "+PER+Double": COLOR_PD}
AGENT_LABEL = {
    "vanilla":     "vanilla DQN (baseline)",
    "+Double":     "+Double DQN",
    "+PER":        "+PER (prioritized replay)",
    "+PER+Double": "+PER+Double DQN",
}

VTYPES = ["car", "bus", "ambulance"]
# Metrics common to ALL 3 runs (exp113 vanilla is the old run, doesn't have xts_/per_visit)
SYSTEM_METRICS = [
    "avg_stopped_time",
    "avg_stop_events",
    "avg_speed",
    "avg_wait",
]

# ── Parser ──────────────────────────────────────────────────────────────────
BLOCK_SIZE = 32 * 1024
FILE_HEADER = 7
REC_HEADER = 7
FULL, FIRST, MID, LAST = 1, 2, 3, 4


def iter_records(path):
    with open(path, 'rb') as f:
        f.read(FILE_HEADER); chunk = b""
        while True:
            pos = f.tell(); offset = pos % BLOCK_SIZE; sl = BLOCK_SIZE - offset
            if sl < REC_HEADER:
                f.read(sl); continue
            hdr = f.read(REC_HEADER)
            if len(hdr) < REC_HEADER: return
            _, length, dtype = struct.unpack("<IHB", hdr)
            data = f.read(length)
            if len(data) < length: return
            if dtype == FULL: yield data
            elif dtype == FIRST: chunk = data
            elif dtype == MID:   chunk += data
            elif dtype == LAST:  chunk += data; yield chunk; chunk = b""


def load_history(run_dir):
    wf = [f for f in os.listdir(run_dir) if f.endswith('.wandb')]
    if not wf: return {}
    path = os.path.join(run_dir, wf[0])
    hist = {}
    for payload in iter_records(path):
        try:
            r = pb.Record(); r.ParseFromString(payload)
        except Exception: continue
        if r.WhichOneof("record_type") != "history": continue
        for item in r.history.item:
            key = item.nested_key[0] if item.nested_key else item.key
            if not key: continue
            try: v = json.loads(item.value_json)
            except Exception: continue
            if isinstance(v, (int, float)):
                hist.setdefault(key, []).append(v)
    return hist


def mad_filter(arr, k=3.5):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 4: return arr, 0
    median = np.median(arr); mad = np.median(np.abs(arr - median))
    if mad < 1e-9: return arr, 0
    threshold = k * mad * 1.4826
    mask = np.abs(arr - median) <= threshold
    return arr[mask], int(len(arr) - mask.sum())


def collect_stats(hist, scope, vtype, metric, window, mad_k=3.5):
    col = f"eval_{scope}/{vtype}/{metric}"
    series = hist.get(col, [])
    if not series: return None, None, 0
    if vtype == "ambulance":
        series = [v for v in series if v != 0.0]
    tail = series[-window:]
    if len(tail) < 2: return None, None, len(tail)
    cleaned, _ = mad_filter(tail, k=mad_k)
    if len(cleaned) < 2: return None, None, len(cleaned)
    return float(cleaned.mean()), float(cleaned.std(ddof=1)), len(cleaned)


# ── Plot ────────────────────────────────────────────────────────────────────
def plot_scope(histories, scope, metrics, out_dir, window, mad_k):
    with mpl.rc_context(RC):
        n_rows = len(metrics)
        n_cols = len(VTYPES)
        fig = plt.figure(figsize=(6.5 * n_cols, 3.8 * n_rows + 1.0))
        gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                      height_ratios=[0.45] + [3.0] * n_rows,
                      hspace=0.60, wspace=0.30)

        # Legend
        lax = fig.add_subplot(gs[0, :])
        lax.set_axis_off()
        handles = [
            Line2D([0], [0], color=AGENT_COLOR[a], marker="s",
                   lw=0, markersize=12, label=AGENT_LABEL[a])
            for a in AGENTS
        ]
        lax.legend(handles=handles, loc="center", ncol=4, fontsize=12,
                   handletextpad=0.5, columnspacing=1.5, frameon=False)

        x = np.arange(len(AGENTS))
        bar_w = 0.60

        for i, m in enumerate(metrics):
            for j, v in enumerate(VTYPES):
                ax = fig.add_subplot(gs[i + 1, j])

                means, stds, ns = [], [], []
                for a in AGENTS:
                    h = histories.get(a, {})
                    mu, sd, n = collect_stats(h, scope, v, m, window, mad_k)
                    means.append(np.nan if mu is None else mu)
                    stds.append(0.0 if sd is None else sd)
                    ns.append(n)

                colors = [AGENT_COLOR[a] for a in AGENTS]
                bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.80,
                              edgecolor="black", linewidth=0.8, width=bar_w,
                              capsize=5, error_kw={"linewidth": 1.5})

                # Annotate value on top of each bar
                ymin, ymax = ax.get_ylim()
                yr = max(ymax - ymin, 1e-6)
                ax.set_ylim(ymin - 0.05 * yr, ymax + 0.25 * yr)
                for k, (mu, sd) in enumerate(zip(means, stds)):
                    if np.isnan(mu): continue
                    ax.text(x[k], mu + sd + (ymax - ymin) * 0.04,
                            f"{mu:.2f}\n±{sd:.2f}",
                            ha="center", va="bottom",
                            fontsize=10.5, color=colors[k], fontweight="bold",
                            linespacing=1.05)

                ax.set_xticks(x)
                if v == "ambulance":
                    tick_lbls = [f"{a}\n(n={ns[k]})" for k, a in enumerate(AGENTS)]
                    ax.set_xticklabels(tick_lbls, fontsize=9.5)
                else:
                    ax.set_xticklabels(AGENTS, fontsize=11)
                ax.set_title(f"{m}  |  {v}", fontsize=11, pad=8)
                ax.grid(alpha=0.25, color=PALETTE["grid"], axis="y", zorder=0)

        scope_label = "system" if scope == "system" else f'intersection "{scope}"'
        fig.suptitle(
            f"1x1 PER Ablation @ {scope_label}   "
            f"(exp113/143/144, 5-2-1 NSamb, wandb last {window} evals, MAD k={mad_k})",
            fontsize=15, fontweight="bold", y=0.995,
        )

        os.makedirs(out_dir, exist_ok=True)
        save_path = f"{out_dir}/01_per_ablation_{scope}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  ✓ Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="figures/1x1_per_ablation")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--mad-k", type=float, default=3.5)
    args = p.parse_args()

    print("Loading 1x1 histories (3 runs)...")
    histories = {}
    for a, d in RUNS.items():
        if not os.path.isdir(d):
            print(f"  ✗ {a}: not found"); histories[a] = {}; continue
        h = load_history(d)
        histories[a] = h
        n_all = len(h.get("eval_system/all/avg_stopped_time", []))
        print(f"  ✓ {a:14s} : {len(h):3d} metrics, {n_all:4d} eval pts")

    # 1x1 only has system + 1 intersection — auto-detect intersection name
    scopes = ["system"]
    for k in histories.get("vanilla", {}):
        if k.startswith("eval_") and "/" in k:
            scope = k.split("/")[0].replace("eval_", "")
            if scope not in scopes and scope != "system":
                scopes.append(scope)
                break  # only need one

    print(f"\nGenerating plots for scopes: {scopes}")
    for scope in scopes:
        # For non-system scope, use simpler metrics (only avg_*)
        metrics = SYSTEM_METRICS if scope == "system" else [
            "avg_stopped_time", "avg_stop_events", "avg_speed"
        ]
        plot_scope(histories, scope, metrics, args.out_dir, args.window, args.mad_k)

    print(f"\nDone — figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
