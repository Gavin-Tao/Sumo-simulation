"""1x1 全部方法一张表 (2026-10-11 用户令): 14 臂 (274/263 主图臂 + 316-327) 作行, 指标作列, 按 all 停车时间/visit 升序排 (最好在最上),
每格 mean ± s.d. (10 评估种子 123-132), 每列最小值标绿, 另给 "对最好一臂的配对胜场" 与 "相对最好的 %"。口径与 plot_1x1_gs_redesign 相同。
用法 (仓库根目录): python experiments/tools/figures/plot_1x1_all_methods_ranked_2026-10-11.py
输出: experiments/analysis/figures/grid1x1_all_methods_ranked_2026-10-11_en.{png,json}
"""
import glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FIG = "experiments/analysis/figures"; PV = "experiments/analysis/data/pervehicle"
SH = {"car": .965, "bus": .033, "ambulance": .0017}; WT = {"car": 1, "bus": 3, "ambulance": 5}
ARMS = [("8STD perphase (main fig.)", "exp274", 274, 1325), ("old GS FRAP (main fig., seed 1)", "exp263", 263, 1370),
        ("8STD + log-wait", "exp327", 327, 1300), ("8STD + since-green", "exp326", 326, 1485), ("8STD + both obs", "exp316", 316, 1475),
        ("8STD + both + green-elapsed", "exp320", 320, 805),
        ("FRAP pressure + both obs", "exp317", 317, 960), ("FRAP pressure + header", "exp318", 318, 335), ("FRAP pressure + green-elapsed", "exp319", 319, 1275),
        ("FRAP learned + both obs", "exp323", 323, 1330), ("FRAP learned + header", "exp324", 324, 1365), ("FRAP learned + green-elapsed", "exp325", 325, 500),
        ("menu-Q, enumerated 12", "exp321", 321, 1470), ("menu-Q, standard 8", "exp322", 322, 1375)]
GREEN = "#cfe9cf"; GREY = "#f3f3f3"; GOLD = "#ffe9a8"

def load(e, ep):
    return {int(f.split("_seed")[1].split("_")[0]): json.load(open(f)) for f in glob.glob(f"{PV}/exp{e}_ep{ep:05d}_seed*_best_coll.json")}

def delay(d):
    cv = d["collector"]["vehicles"]; acc = {c: [] for c in ("car", "bus", "ambulance", "all")}
    for vid, rec in d["vehicles"].items():
        t = rec.get("trip"); k = cv.get(vid, {}).get("k_v", 0)
        if not t or k <= 0 or rec.get("type") not in acc: continue
        x = float(t["timeLoss"]) / k; acc[rec["type"]].append(x); acc["all"].append(x)
    return {c: (float(np.mean(v)) if v else np.nan) for c, v in acc.items()}

def main():
    data = {a[2]: load(a[2], a[3]) for a in ARMS}; seeds = sorted(set.intersection(*[set(v) for v in data.values()]))
    per = {}   # exp -> metric -> array over seeds
    for _, _, e, _ in ARMS:
        M = {}
        for c in ("car", "bus", "ambulance", "all"):
            M[f"t_{c}"] = np.array([data[e][s]["collector"]["summary"].get(f"eval/system/{c}/avg_stopped_time_per_visit", np.nan) for s in seeds])
            M[f"e_{c}"] = np.array([data[e][s]["collector"]["summary"].get(f"eval/system/{c}/avg_stop_events_per_visit", np.nan) for s in seeds])
        dl = [delay(data[e][s]) for s in seeds]
        for c in ("car", "bus", "ambulance", "all"): M[f"d_{c}"] = np.array([x[c] for x in dl])
        for k in ("t", "e", "d"): M[f"J_{k}"] = sum(SH[c] * WT[c] * np.nan_to_num(M[f"{k}_{c}"]) for c in SH)
        per[e] = M
    order = sorted(ARMS, key=lambda a: np.nanmean(per[a[2]]["t_all"])); best = order[0][2]
    cols = [("t_all", "Stopped\nall (s)", 2), ("t_car", "Stopped\ncar", 2), ("t_bus", "Stopped\nbus", 2), ("t_ambulance", "Stopped\namb", 2), ("J_t", "J(531)\nstopped", 2),
            ("e_all", "Stops\nall", 3), ("e_car", "Stops\ncar", 3), ("e_bus", "Stops\nbus", 3), ("J_e", "J(531)\nstops", 3),
            ("d_all", "Delay\nall (s)", 2), ("d_car", "Delay\ncar", 2), ("d_bus", "Delay\nbus", 2), ("d_ambulance", "Delay\namb", 2), ("J_d", "J(531)\ndelay", 2)]
    cells, colors, dump = [], [], {"seeds": seeds, "best_by_all_stopped_time": order[0][1], "rows": []}
    mins = {k: min(np.nanmean(per[a[2]][k]) for a in ARMS) for k, _, _ in cols}
    for rank, (name, tag, e, ep) in enumerate(order, 1):
        M = per[e]; rel = 100 * (np.nanmean(M["t_all"]) / np.nanmean(per[best]["t_all"]) - 1)
        wins = int((per[best]["t_all"] < M["t_all"]).sum()) if e != best else None
        row = [f"{rank}", f"{name}\n{tag}, best ep{ep}", f"{rel:+.0f}%" if e != best else "best", f"{wins}/{len(seeds)}" if wins is not None else "-"]
        col = [GOLD if e == best else GREY] * 4
        for k, _, d in cols:
            m, sd = float(np.nanmean(M[k])), float(np.nanstd(M[k])); row.append(f"{m:.{d}f} ± {sd:.{d}f}"); col.append(GREEN if abs(m - mins[k]) < 1e-12 else GREY)
        cells.append(row); colors.append(col)
        dump["rows"].append({"rank": rank, "name": name, "exp": e, "best_ep": ep, "rel_vs_best_all_stopped_time_pct": rel, "best_wins_over_this": wins,
                             **{k: [float(np.nanmean(M[k])), float(np.nanstd(M[k]))] for k, _, _ in cols}})
    print(f"seeds n={len(seeds)}; 最好 (all 停车时间): {order[0][0]} ({order[0][1]})")
    for row in cells: print("  " + " | ".join(x.replace("\n", " ") for x in row[:5]), "| ...")
    fig, ax = plt.subplots(figsize=(24, 0.42 * len(cells) + 1.9)); ax.axis("off")
    hdr = ["Rank", "Method (1x1, best checkpoint)", "vs best\n(all stopped)", "best wins\n(seeds)"] + [c[1] for c in cols]
    tbl = ax.table(cellText=cells, colLabels=hdr, cellColours=colors, colColours=["#c9d3df"] * len(hdr), loc="upper center", cellLoc="center",
                   colWidths=[0.028, 0.165, 0.048, 0.045] + [0.0515] * len(cols))
    tbl.auto_set_font_size(False); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        t = cell.get_text(); t.set_fontsize(9.2)
        if r == 0: t.set_weight("bold"); cell.set_height(cell.get_height() * 1.3)
        if c == 1 and r > 0: t.set_ha("left"); cell.PAD = 0.01
    ax.set_title("1x1 grid: all 14 methods ranked by stopped time per visit (all vehicles), mean ± s.d. over evaluation seeds 123-132; green = best in column; gold = overall best",
                 fontsize=12.5, pad=8)
    note = ("Each row = best checkpoint (sliding-3) of one run, evaluated with the merged evaluator on 10 seeds. 'both obs' = since-green + log-scale wait. "
            "Delay = whole-trip time loss / junction visits. 'best wins' = number of seeds on which the top method beats this row (all stopped time). "
            "J(531) = 0.965·car + 3·0.033·bus + 5·0.0017·ambulance (1x1 visit shares). Ambulance = 1 veh/h flows (underpowered, s.d. ≈ mean). Main five-scenario figure's 1x1 block excluded seed 128; here all 10 seeds.")
    fig.text(0.02, 0.005, note, fontsize=8.5, color="#333333", va="bottom")
    out = f"{FIG}/grid1x1_all_methods_ranked_2026-10-11_en.png"; plt.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)
    json.dump(dump, open(out.replace(".png", ".json"), "w"), default=float, indent=1)

if __name__ == "__main__": main()
