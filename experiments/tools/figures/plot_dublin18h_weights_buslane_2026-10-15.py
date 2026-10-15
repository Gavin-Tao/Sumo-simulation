"""都柏林 18:00 8STD (12 槽 + F1) 四种优先级权重: 按车道是否有公交经过拆分的折线图 (2026-10-15 用户令: "做个 bus 和 car 同时存在同一车道的对比, car 自己单独没有 bus 路过的一张图")。
数据: 逐秒口径 best 评估文件 (analysis/data/pervehicle/exp{e}_ep{best}_seed{s}_best_coll1s.json), 9 个评估种子 (四列同剔种子 125), 默认剔除 4 个虚构信号路口 (与 xjfake4 表图一致)。
分组 (按车道, 不按时刻): 公交车道集合 = 四臂 × 9 种子的全部评估中, 公交在保留路口的受控进口车道上出现过的 (路口, 车道);
  每次过路口按逐车记录里的车道 (停止线前所在车道) 归组: 落在公交车道集合 → "与公交同车道", 否则 → "无公交车道" (同一路口多条记录时任一条在集合内即算前者, 全网仅约 6 次)。
  不用 "同一时刻同一车道上有公交" 分组: 在路口停留越久越容易与公交重叠, 该组天然偏向等得久的车 (停留时长偏差)。
指标: 停车时间 / visit 与停车次数 / visit, 组内沿用收集器口径 (每车在该组路口的 Σ / 该组访问路口数, 跳过该组访问数为 0 的车, 再对车平均);
  所有车道合并时与 xjfake4 表图逐项相同 (脚本内核对)。延误只存到路口一级 (edgeData), 拆不到车道, 故本图不画延误。
输出: 图1 与公交同车道 (公交 + 小汽车), 图2 无公交车道 (只小汽车), 两图同一纵轴范围便于对比; 数据 json 记录分组定义、公交车道集合与各组样本量。
用法 (仓库根目录): python experiments/tools/figures/plot_dublin18h_weights_buslane_2026-10-15.py [--all-junctions]"""
import glob, json, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG = "experiments/analysis/figures"; PV = "experiments/analysis/data/pervehicle"
ARMS = [("111", 458, "exp458_dublin18h_reroute_v2_amb11h_noprio_avgwait_8std_slot12_f1"), ("521", 456, "exp456_dublin18h_reroute_v2_amb11h_521_8std_slot12_f1"),
        ("531", 449, "exp449_dublin18h_reroute_v2_amb11h_531_8std_slot12_f1"), ("541", 457, "exp457_dublin18h_reroute_v2_amb11h_541_8std_slot12_f1")]
SEEDS = [s for s in range(123, 133) if s != 125]          # 四列同剔种子 125 (同表图)
FAKE4 = ["659802", "133307283", "32236377", "135109534"]  # 现实中无信号的 4 个路口
XJ = [] if "--all-junctions" in sys.argv else FAKE4
TAG = "" if not XJ else "_xjfake4"
STEM = f"{FIG}/dublin18h_8std_slot12_f1_weights_best_1s_excl125all{TAG}"
XLAB = ["111\n(no priority)", "521", "531", "541"]
PANELS = [("stopped", "Stopped time / visit (s)"), ("events", "Stop events / visit")]
CAR, BUS = "#B64342", "#0F4D92"

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 13, "axes.linewidth": 1.4,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def best_ep(name):
    return int(json.load(open(f"experiments/models/{name}/frozen_best_2026-10-14/best_metrics.json"))["_meta"]["episode"])


def ctrl_lanes():
    """各 RL 路口的受控进口车道 (不含恒绿车道), 与收集器 ts_lane_map 相同 (同表图脚本 ctrl_lanes)。"""
    import yaml; os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
    sys.path.insert(0, "/home/xiaowen/sumo-rl")
    from sumo_rl.environment.env import SumoEnvironment
    cfg = yaml.safe_load(open(glob.glob("experiments/configs/exp449_*.yaml")[0]))
    cwd = os.getcwd(); os.chdir("experiments")
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False, num_seconds=10,
                          min_green=5, max_green=50, single_agent=False, yellow_time=2, delta_time=5, sumo_seed=0, sumo_warnings=False)
    env.reset(123); cl = {t: set(env.traffic_signals[t].signal_controlled_lanes) for t in env.ts_ids}; env.close(); os.chdir(cwd)
    return cl


def visits(V, vid, CL):
    """{路口: [逐车记录]}: 只留受控进口车道上的记录 (与收集器的访问定义一致)。"""
    out = {}
    for t, rs in V.get(vid, {}).get("tls", {}).items():
        rs = [r for r in rs if t in CL and r["lane"] in CL[t]]
        if rs: out[t] = rs
    return out


def main():
    CL = ctrl_lanes(); runs = {}
    for a, e, n in ARMS:
        ep = best_ep(n); runs[a] = {s: json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_best_coll1s.json")) for s in SEEDS}
        assert all(d["collector_dt"] == 1.0 for d in runs[a].values()), a
    # 公交车道集合: 四臂 × 9 种子并集 (同一集合用于四臂, 分组对四臂相同)
    buscnt = collections.Counter()
    for a in runs:
        for d in runs[a].values():
            V = d["vehicles"]
            for vid, v in V.items():
                if v.get("type") != "bus": continue
                for t, rs in visits(V, vid, CL).items():
                    if t not in XJ:
                        for r in rs: buscnt[(t, r["lane"])] += 1
    BL = set(buscnt)
    res = {g: {a: {"stopped": [], "events": [], "n_vehicles": [], "n_visits": []} for a, _, _ in ARMS} for g in ("bus", "car_buslane", "car_nobuslane", "car_all")}
    kv_mismatch = 0
    for a in runs:
        for s in SEEDS:
            d = runs[a][s]; V = d["vehicles"]; acc = {g: [] for g in res}; nvis = collections.Counter()
            for vid, c in d["collector"]["vehicles"].items():
                vis = visits(V, vid, CL)
                kv_mismatch += len(vis) != c.get("k_v", 0)
                vis = {t: rs for t, rs in vis.items() if t not in XJ}
                if not vis or c["type"] not in ("car", "bus"): continue
                ss, se = c.get("stopped_sec", {}), c.get("stop_events", {})
                grp = collections.defaultdict(lambda: [0.0, 0.0, 0])
                for t, rs in vis.items():
                    keys = ["all"] if c["type"] == "bus" else ["all", "buslane" if any((t, r["lane"]) in BL for r in rs) else "nobuslane"]
                    for k in keys:
                        g = grp[k]; g[0] += ss.get(t, 0); g[1] += se.get(t, 0); g[2] += 1
                for k, (x, y, m) in grp.items():
                    gname = "bus" if c["type"] == "bus" else f"car_{k}"
                    acc[gname].append((x / m, y / m)); nvis[gname] += m
            for g, L in acc.items():
                r = res[g][a]; r["stopped"].append(float(np.mean([x for x, _ in L]))); r["events"].append(float(np.mean([y for _, y in L])))
                r["n_vehicles"].append(len(L)); r["n_visits"].append(nvis[g])
    assert kv_mismatch == 0, kv_mismatch   # 逐车访问数与收集器 k_v 全部一致
    # 核对: 小汽车所有车道合并 / 公交 与表图 json 逐项相同
    tab = f"{FIG}/dublin18h_8std_slot12_f1_weights_best_1s_excl125all_noJ_nohl_mintitle{TAG}_2026-10-15_en.json"
    if os.path.exists(tab):
        T = json.load(open(tab)); diff = 0.0
        for a, _, _ in ARMS:
            for m, k in (("avg_stopped_time_per_visit", "stopped"), ("avg_stop_events_per_visit", "events")):
                diff = max(diff, float(np.max(np.abs(np.array(T[m]["car"][a]["per_seed"]) - res["car_all"][a][k]))),
                           float(np.max(np.abs(np.array(T[m]["bus"][a]["per_seed"]) - res["bus"][a][k]))))
        print("与表图 json 最大差:", diff); assert diff < 1e-9
    dump = {"seeds": SEEDS, "excluded_junctions": XJ, "best_episode": {a: best_ep(n) for a, _, n in ARMS}, "collector_dt": 1.0,
            "grouping": "per visit, by the approach lane in the per-vehicle record; bus lanes = (junction, lane) pairs used by any bus in any of the 4 arms x 9 seeds",
            "bus_lanes": sorted(f"{t}|{l}" for t, l in BL), "bus_visits_per_lane_all_runs": {f"{t}|{l}": n for (t, l), n in sorted(buscnt.items())},
            "n_controlled_lanes_kept": sum(len(CL[t]) for t in CL if t not in XJ), "groups": res}
    json.dump(dump, open(f"{STEM}_buslane_split_2026-10-15_en.json", "w"), indent=1)

    def mu_sd(g, k):
        return (np.array([np.mean(res[g][a][k]) for a, _, _ in ARMS]), np.array([np.std(res[g][a][k]) for a, _, _ in ARMS]))
    ylim = {k: 1.08 * max(float((mu_sd(g, k)[0] + mu_sd(g, k)[1]).max()) for g in ("bus", "car_buslane", "car_nobuslane")) for k, _ in PANELS}
    figs = [("buslanes", "Approach lanes shared by buses and cars", [("bus", "Bus", BUS, "s"), ("car_buslane", "Car (same lanes as buses)", CAR, "o")]),
            ("nobuslanes", "Approach lanes with no bus traffic", [("car_nobuslane", "Car (lanes with no buses)", CAR, "o")])]
    for tag, sub, series in figs:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9)); x0 = np.arange(len(ARMS))
        dodge = (np.arange(len(series)) - (len(series) - 1) / 2) * 0.07
        for ax, (k, ylab) in zip(axes, PANELS):
            for j, (g, lab, col, mk) in enumerate(series):
                mu, sd = mu_sd(g, k); x = x0 + dodge[j]
                ax.plot(x[:2], mu[:2], ls="--", color=col, lw=1.6, alpha=0.8, zorder=2)   # 111 → 521: 奖励形式不同, 虚线
                ax.plot(x[1:], mu[1:], ls="-", color=col, lw=2.2, zorder=2)                # 521 → 531 → 541: 只改公交权重, 实线
                ax.errorbar(x, mu, yerr=[np.minimum(sd, mu), sd], fmt=mk, color=col, ms=7, mec="black", mew=0.6, ecolor=col, elinewidth=1.3, capsize=3.5, zorder=3)
            ax.set_xticks(x0); ax.set_xticklabels(XLAB); ax.set_ylabel(ylab); ax.set_ylim(0, ylim[k]); ax.set_xlim(-0.4, len(ARMS) - 0.6)
            ax.set_xlabel("Priority weights (ambulance/bus/car)")
            ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
        handles = [Line2D([0], [0], color=col, marker=mk, ms=7, mec="black", mew=0.6, lw=2.2, label=lab) for _, lab, col, mk in series]
        fig.legend(handles=handles, loc="upper center", ncol=len(series), bbox_to_anchor=(0.5, 1.03))
        fig.suptitle(f"Dublin 18:00, 8STD (12-slot state + F1): {sub}", y=1.10, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(f"{STEM}_{tag}_lines_2026-10-15_en.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig); print("saved", f"{STEM}_{tag}_lines_2026-10-15_en.{{png,pdf}}")
    for g in ("bus", "car_buslane", "car_nobuslane", "car_all"):
        print(g, " ".join(f"{a}: {np.mean(res[g][a]['stopped']):.2f}/{np.mean(res[g][a]['events']):.3f} (n_veh {np.mean(res[g][a]['n_vehicles']):.0f})" for a, _, _ in ARMS))


if __name__ == "__main__":
    main()
