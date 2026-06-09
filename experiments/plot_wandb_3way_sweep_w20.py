"""3-way weight sensitivity (vanilla / CoLight / CoeffDQN) — window=20, outlier-filtered.

Differences from plot_wandb_3way_sweep.py:
  • window=20 evals (= last ~100 training episodes, matches user's wandb view)
  • MAD-based outlier filter on each cell's eval window (drops |x - median| > 3.5 × MAD,
    the standard Iglewicz & Hoaglin robust threshold; flagged values are excluded from
    mean ± std)
  • Output: figures/wandb_3way_sweep_w20/
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
    "blue_main":  "#0F4D92",
    "red_strong": "#B64342",
    "green_teal": "#2A8C7C",
    "neutral":    "#4D4D4D",
    "grid":       "#CFCECE",
}
COLOR_DQ = PALETTE["red_strong"]
COLOR_CO = PALETTE["blue_main"]
COLOR_CF = PALETTE["green_teal"]

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
    "5-2-1": {
        "vanilla":  f"{RUN_BASE}/run-20260513_214320-perw7og9",
        "CoLight":  f"{RUN_BASE}/run-20260514_030613-0i7dt5kz",
        "CoeffDQN": f"{RUN_BASE}/run-20260609_112728-j5n1msir",
    },
    "5-3-1": {
        "vanilla":  f"{RUN_BASE}/run-20260513_230448-jtpfs94u",
        "CoLight":  f"{RUN_BASE}/run-20260514_054904-en3rmt0q",
        "CoeffDQN": f"{RUN_BASE}/run-20260609_112743-uk09mham",
    },
    "5-4-1": {
        "vanilla":  f"{RUN_BASE}/run-20260513_230451-ecayx65r",
        "CoLight":  f"{RUN_BASE}/run-20260514_054907-972s5ure",
        "CoeffDQN": f"{RUN_BASE}/run-20260609_112758-3rz36v73",
    },
}

WEIGHTS = list(RUNS.keys())
AGENTS = ["vanilla", "CoLight", "CoeffDQN"]
AGENT_COLOR = {"vanilla": COLOR_DQ, "CoLight": COLOR_CO, "CoeffDQN": COLOR_CF}
AGENT_MARKER = {"vanilla": "s", "CoLight": "o", "CoeffDQN": "D"}
AGENT_LABEL = {
    "vanilla":  "vanilla DQN (no coordination)",
    "CoLight":  "CoLight (GAT state-layer coordination)",
    "CoeffDQN": "CoeffDQN (β reward-layer coordination)",
}
VTYPES = ["car", "bus", "ambulance"]

SYSTEM_METRICS = [
    "avg_stopped_time_per_visit",
    "avg_stop_events_per_visit",
    "xts_avg_stopped_time",
    "xts_avg_stop_events",
    "xts_avg_speed",
]
INTERSECTION_METRICS = [
    "avg_stopped_time",
    "avg_stop_events",
    "avg_speed",
]

# ── Local .wandb parser ─────────────────────────────────────────────────────
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
    if not wf:
        return {}
    path = os.path.join(run_dir, wf[0])
    hist = {}
    for payload in iter_records(path):
        try:
            r = pb.Record(); r.ParseFromString(payload)
        except Exception:
            continue
        if r.WhichOneof("record_type") != "history":
            continue
        for item in r.history.item:
            key = item.nested_key[0] if item.nested_key else item.key
            if not key:
                continue
            try:
                v = json.loads(item.value_json)
            except Exception:
                continue
            if isinstance(v, (int, float)):
                hist.setdefault(key, []).append(v)
    return hist


# ── Outlier filter (modified Z-score using MAD) ─────────────────────────────
def filter_outliers_mad(values, k=3.5):
    """Drop values with |x - median| > k * MAD * 1.4826  (Iglewicz & Hoaglin 1993).

    Returns (filtered_values, n_dropped). With k=3.5 only EXTREME outliers are removed
    (in a normal distribution this drops <0.05% of points; in practice it catches
    genuine spikes/glitches without aggressive trimming).
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4:
        return arr, 0
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad < 1e-9:
        return arr, 0  # all identical, no outliers
    threshold = k * mad * 1.4826
    mask = np.abs(arr - median) <= threshold
    return arr[mask], int(len(arr) - mask.sum())


def collect_stats(hist, scope, vtype, metric, window, mad_k=3.5):
    """Return (mean, std, n_used, n_dropped) for last `window` eval rows, with outliers."""
    col = f"eval_{scope}/{vtype}/{metric}"
    series = hist.get(col, [])
    if not series:
        return None, None, 0, 0
    # For ambulance, drop n_amb=0 evals first (the 0's aren't outliers — they're "no amb present")
    if vtype == "ambulance":
        series = [v for v in series if v != 0.0]
    tail = series[-window:]
    if len(tail) < 2:
        return None, None, len(tail), 0
    # MAD-based outlier filter
    cleaned, dropped = filter_outliers_mad(tail, k=mad_k)
    if len(cleaned) < 2:
        return None, None, len(cleaned), dropped
    return float(cleaned.mean()), float(cleaned.std(ddof=1)), len(cleaned), dropped


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_scope(histories, scope, metrics, out_dir, window, mad_k):
    with mpl.rc_context(RC):
        n_rows = len(metrics)
        n_cols = len(VTYPES)
        fig = plt.figure(figsize=(6.0 * n_cols, 3.6 * n_rows + 1.2))
        gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                      height_ratios=[0.5] + [3.0] * n_rows,
                      hspace=0.50, wspace=0.30)

        # Legend
        lax = fig.add_subplot(gs[0, :])
        lax.set_axis_off()
        legend_handles = [
            Line2D([0], [0], color=AGENT_COLOR[a], marker=AGENT_MARKER[a],
                   lw=2.2, markersize=8, label=AGENT_LABEL[a])
            for a in AGENTS
        ]
        lax.legend(handles=legend_handles, loc="center", ncol=3,
                   fontsize=13, frameon=False,
                   handletextpad=0.5, columnspacing=1.8)

        x_plot = list(range(len(WEIGHTS)))
        total_dropped = {a: 0 for a in AGENTS}

        for i, m in enumerate(metrics):
            for j, v in enumerate(VTYPES):
                ax = fig.add_subplot(gs[i + 1, j])
                by_agent = {}
                for a in AGENTS:
                    means, stds, ns, drops = [], [], [], []
                    for w in WEIGHTS:
                        h = histories.get((w, a), {})
                        mu, sd, n, d = collect_stats(h, scope, v, m, window, mad_k)
                        means.append(mu); stds.append(sd); ns.append(n); drops.append(d)
                        total_dropped[a] += d
                    by_agent[a] = {
                        "m": np.array([np.nan if x is None else x for x in means]),
                        "s": np.array([0.0   if x is None else x for x in stds]),
                        "n": ns, "d": drops,
                    }

                for a in AGENTS:
                    ax.errorbar(
                        x_plot, by_agent[a]["m"], yerr=by_agent[a]["s"],
                        marker=AGENT_MARKER[a], linewidth=2.0, markersize=7,
                        capsize=4, color=AGENT_COLOR[a], zorder=3,
                    )

                ax.set_xlim(-0.6, len(WEIGHTS) - 1 + 0.6)
                ymin, ymax = ax.get_ylim()
                yr = max(ymax - ymin, 1e-6)
                ax.set_ylim(ymin - 0.30 * yr, ymax + 0.30 * yr)
                yL, yH = ax.get_ylim()
                yR = yH - yL

                label_y = {
                    "CoLight":  yH - 0.07 * yR,
                    "CoeffDQN": (yL + yH) / 2 + 0.18 * yR,
                    "vanilla":  yL + 0.07 * yR,
                }
                for k in range(len(x_plot)):
                    for a in AGENTS:
                        mu = by_agent[a]["m"][k]; sd = by_agent[a]["s"][k]
                        if np.isnan(mu): continue
                        s = sd if not np.isnan(sd) else 0.0
                        ax.text(x_plot[k], label_y[a],
                                f"{mu:.2f}\n±{s:.2f}",
                                ha="center", va="center",
                                fontsize=9, color=AGENT_COLOR[a],
                                fontweight="bold", linespacing=1.0)

                ax.set_xticks(x_plot)
                if v == "ambulance":
                    # Show n (used) for ambulance — dropped count is small in extreme-only filter
                    ns_co = by_agent["CoLight"]["n"]
                    ns_cf = by_agent["CoeffDQN"]["n"]
                    ns_dq = by_agent["vanilla"]["n"]
                    tick_lbls = [
                        f"{w}\n(n={ns_dq[k]}/{ns_co[k]}/{ns_cf[k]})"
                        for k, w in enumerate(WEIGHTS)
                    ]
                    ax.set_xticklabels(tick_lbls, fontsize=9)
                else:
                    ax.set_xticklabels(WEIGHTS, fontsize=11)

                ax.set_title(f"{m}  |  {v}", fontsize=11, pad=8)
                ax.grid(alpha=0.25, color=PALETTE["grid"], zorder=0)
                if i == n_rows - 1:
                    ax.set_xlabel("Priority weights (amb / bus / car)",
                                  fontsize=11)

        scope_label = ("system" if scope == "system"
                       else f'intersection "{scope}"')
        fig.suptitle(
            f"3-way Weight Sensitivity @ {scope_label}   "
            f"(wandb last {window} evals, MAD-filtered k={mad_k}, NS 20% bus, 1% amb U-route)",
            fontsize=15, fontweight="bold", y=0.995,
        )

        os.makedirs(out_dir, exist_ok=True)
        save_path = f"{out_dir}/02_weight_3way_w{window}_{scope}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        td_sum = sum(total_dropped.values())
        print(f"  ✓ Saved: {save_path}  (outliers dropped across all panels: "
              f"vanilla={total_dropped['vanilla']}, "
              f"CoLight={total_dropped['CoLight']}, "
              f"CoeffDQN={total_dropped['CoeffDQN']}, total={td_sum})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="figures/wandb_3way_sweep_w20")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--mad-k", type=float, default=3.5,
                   help="MAD threshold for outlier filter (default: 3.5 = Iglewicz&Hoaglin std)")
    args = p.parse_args()

    print(f"Loading histories for 9 runs...")
    histories = {}
    for w, agents in RUNS.items():
        for a, d in agents.items():
            if not os.path.isdir(d):
                print(f"  ✗ {w} {a}: dir not found")
                histories[(w, a)] = {}
                continue
            h = load_history(d)
            histories[(w, a)] = h
            n_all = len(h.get("eval_system/all/avg_stopped_time", []))
            print(f"  ✓ {w} {a:10s}: {len(h):3d} metrics, {n_all:4d} eval pts")

    print(f"\nGenerating plots (window={args.window} evals = last ~{args.window*5} eps, "
          f"MAD k={args.mad_k} extreme-only filter)...")
    plot_scope(histories, "system", SYSTEM_METRICS, args.out_dir, args.window, args.mad_k)
    for inter in ["t", "e", "J0"]:
        plot_scope(histories, inter, INTERSECTION_METRICS, args.out_dir, args.window, args.mad_k)

    print(f"\nDone — 4 figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
