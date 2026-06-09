"""3-way weight sensitivity plot (vanilla DQN / CoLightOrig / CoeffDQN) on NS20bus_1amb_U.

Mirrors plot_wandb_weight_sweep.py but:
  • Adds CoeffDQN (exp145/146/147) as a 3rd line
  • Loads from local .wandb files (works for offline runs — CoeffDQN runs were offline)
  • Per-intersection (t/e/J0) plot uses only avg_* metrics (xts_* don't exist per-intersection)
  • System plot keeps the original 4 metrics (per_visit + 3 xts)

Output: figures/wandb_3way_sweep/02_weight_3way_<scope>.png
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

# ── Publication style ───────────────────────────────────────────────────────
PALETTE = {
    "blue_main":  "#0F4D92",   # CoLight (existing — state-layer coordination)
    "red_strong": "#B64342",   # vanilla DQN (baseline)
    "green_teal": "#2A8C7C",   # CoeffDQN (proposed — reward-layer coordination)
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

# ── Run identifiers (one per cell of the 3 × 3 ablation matrix) ─────────────
# weight → (vanilla, CoLight, CoeffDQN) run dir (full path to wandb run-* folder)
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
AGENTS  = ["vanilla", "CoLight", "CoeffDQN"]
AGENT_COLOR = {"vanilla": COLOR_DQ, "CoLight": COLOR_CO, "CoeffDQN": COLOR_CF}
AGENT_MARKER = {"vanilla": "s", "CoLight": "o", "CoeffDQN": "D"}
AGENT_LABEL = {
    "vanilla":  "vanilla DQN (no coordination)",
    "CoLight":  "CoLight (GAT state-layer coordination)",
    "CoeffDQN": "CoeffDQN (β reward-layer coordination)",
}
VTYPES = ["car", "bus", "ambulance"]

# System: same 4 metrics as the 2-way script (per_visit + 3 xts_*)
SYSTEM_METRICS = [
    "avg_stopped_time_per_visit",
    "xts_avg_stopped_time",
    "xts_avg_stop_events",
    "xts_avg_speed",
]
# Per-intersection: only avg_* exist (no xts_*, no _per_visit)
INTERSECTION_METRICS = [
    "avg_stopped_time",
    "avg_stop_events",
    "avg_speed",
]

# ── Local .wandb file parser ─────────────────────────────────────────────────
BLOCK_SIZE  = 32 * 1024
FILE_HEADER = 7
REC_HEADER  = 7
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
    """Return dict[metric_key] = list of values (time-ordered)."""
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


def collect_stats(hist, scope, vtype, metric, window):
    """Return (mean, std, n) of last `window` eval rows for the given metric.

    For ambulance, filter out 0-value rows (likely n_amb=0 evaluation episodes).
    """
    col = f"eval_{scope}/{vtype}/{metric}"
    series = hist.get(col, [])
    if not series:
        return None, None, 0
    if vtype == "ambulance":
        series = [v for v in series if v != 0.0]
    tail = series[-window:]
    if len(tail) < 2:
        return None, None, len(tail)
    arr = np.array(tail, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)), len(arr)


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_scope(histories, scope, metrics, out_dir, window):
    """Publication-style 3-way figure."""
    with mpl.rc_context(RC):
        n_rows = len(metrics)
        n_cols = len(VTYPES)
        fig = plt.figure(figsize=(6.0 * n_cols, 3.6 * n_rows + 1.2))
        gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                      height_ratios=[0.5] + [3.0] * n_rows,
                      hspace=0.50, wspace=0.30)

        # Top legend row (spans all columns)
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

        for i, m in enumerate(metrics):
            for j, v in enumerate(VTYPES):
                ax = fig.add_subplot(gs[i + 1, j])

                # Collect stats per (agent, weight)
                by_agent = {}
                for a in AGENTS:
                    means, stds, ns = [], [], []
                    for w in WEIGHTS:
                        h = histories.get((w, a), {})
                        mu, sd, n = collect_stats(h, scope, v, m, window)
                        means.append(mu); stds.append(sd); ns.append(n)
                    by_agent[a] = {
                        "m": np.array([np.nan if x is None else x for x in means]),
                        "s": np.array([0.0   if x is None else x for x in stds]),
                        "n": ns,
                    }

                # Plot 3 lines
                for a in AGENTS:
                    ax.errorbar(
                        x_plot, by_agent[a]["m"], yerr=by_agent[a]["s"],
                        marker=AGENT_MARKER[a], linewidth=2.0, markersize=7,
                        capsize=4, color=AGENT_COLOR[a], zorder=3,
                    )

                ax.set_xlim(-0.6, len(WEIGHTS) - 1 + 0.6)
                # Expand y so labels don't collide
                ymin, ymax = ax.get_ylim()
                yr = max(ymax - ymin, 1e-6)
                ax.set_ylim(ymin - 0.30 * yr, ymax + 0.30 * yr)
                yL, yH = ax.get_ylim()
                yR = yH - yL

                # 3 label bands stacked vertically: CoLight top, CoeffDQN mid, vanilla bottom
                label_y = {
                    "CoLight":  yH - 0.07 * yR,
                    "CoeffDQN": (yL + yH) / 2 + 0.18 * yR,
                    "vanilla":  yL + 0.07 * yR,
                }
                for k in range(len(x_plot)):
                    for a in AGENTS:
                        mu = by_agent[a]["m"][k]; sd = by_agent[a]["s"][k]
                        if np.isnan(mu):
                            continue
                        s = sd if not np.isnan(sd) else 0.0
                        ax.text(
                            x_plot[k], label_y[a],
                            f"{mu:.2f}\n±{s:.2f}",
                            ha="center", va="center",
                            fontsize=9, color=AGENT_COLOR[a],
                            fontweight="bold", linespacing=1.0,
                        )

                ax.set_xticks(x_plot)
                if v == "ambulance":
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
            f"(wandb last {window} evals, NS 20% bus, 1% amb U-route)",
            fontsize=15, fontweight="bold", y=0.995,
        )

        os.makedirs(out_dir, exist_ok=True)
        save_path = f"{out_dir}/02_weight_3way_{scope}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  ✓ Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="figures/wandb_3way_sweep")
    p.add_argument("--window", type=int, default=50,
                   help="last N wandb eval rows to aggregate (default: 50)")
    args = p.parse_args()

    print(f"Loading histories for 9 runs (3 weights × 3 agents)...")
    histories = {}
    for w, agents in RUNS.items():
        for a, d in agents.items():
            if not os.path.isdir(d):
                print(f"  ✗ {w} {a}: dir not found: {d}")
                histories[(w, a)] = {}
                continue
            h = load_history(d)
            histories[(w, a)] = h
            n_all = len(h.get("eval_system/all/avg_stopped_time", []))
            print(f"  ✓ {w} {a:10s}: {len(h):3d} metrics, {n_all:4d} eval pts")

    print(f"\nGenerating plots (window={args.window} evals = "
          f"~ last {args.window * 5} episodes)...")
    # System: use the 4 metrics with xts/per_visit
    plot_scope(histories, "system", SYSTEM_METRICS, args.out_dir, args.window)
    # Per-intersection: only avg_* exist
    for inter in ["t", "e", "J0"]:
        plot_scope(histories, inter, INTERSECTION_METRICS, args.out_dir, args.window)

    print(f"\nDone — 4 figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
