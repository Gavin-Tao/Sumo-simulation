"""换车型探针 (02h) 的 per-visit 指标, 按训练评估收集器的口径 (EpisodeMetricsCollector: 5 s 采样, k_v = 采样时刻所在受控口数):
救护车版 = 原探针路由 (*_coll.json), 公交版 = busprobe 路由 (*_busprobe_coll.json), 各臂末 4 个检查点 × 评估种子 123-132 (= 40 次评估/臂; 检查点每 50 集一存, 无法复现训练时每 5 集的 40 个策略快照, 用 10 个评估种子补齐点数)。
每次评估: 两辆探针车的 (Σ停车秒 / k_v) 的均值 (与收集器 _per_visit_mean 同式), 再对 4 次评估取均值 ± sd —— 与旧表 "尾40 均值 ± sd" 同构。
输出: experiments/analysis/data/typeswap_02h_pervisit_2026-09-30.json
"""
import glob, json, os, numpy as np
D="experiments/analysis/data/pervehicle"
def pv(vrec, field):
    vals=[sum(r[field].values())/r["k_v"] for r in vrec if r["k_v"]>0]
    return float(np.mean(vals)) if vals else float("nan")
res={}
for e,arm in (("287","8STD"),("288","GS")):
    A=sorted(glob.glob(f"{D}/exp{e}_ep*_seed*_coll.json"))   # 4 检查点 × 全部评估种子 (用户令: 40 个评估点)
    per={"amb_pv":[],"bus_pv":[],"amb_ev":[],"bus_ev":[],"amb_log":[]}
    for f in A:
        g=f.replace("_coll.json","_busprobe_coll.json")
        if not os.path.exists(g): continue
        ca=json.load(open(f))["collector"]; cb=json.load(open(g))["collector"]
        va=[ca["vehicles"][k] for k in ("amb_1_early","amb_1") if k in ca["vehicles"]]
        vb=[cb["vehicles"][k] for k in ("probe_bus_1_early","probe_bus_1") if k in cb["vehicles"]]
        per["amb_pv"].append(pv(va,"stopped_sec")); per["bus_pv"].append(pv(vb,"stopped_sec"))
        per["amb_ev"].append(pv(va,"stop_events")); per["bus_ev"].append(pv(vb,"stop_events"))
        per["amb_log"].append(ca["summary"].get("eval/system/ambulance/avg_stopped_time_per_visit"))
    n=len(per["amb_pv"])
    res[arm]=dict(n=n, protocol="collector 5 s sampling; per-eval mean over the 2 probe vehicles; mean ± sd over 4 checkpoints x 10 evaluation seeds",
                  pv=(float(np.mean(per["amb_pv"])), float(np.mean(per["bus_pv"]))), pv_sd=(float(np.std(per["amb_pv"])), float(np.std(per["bus_pv"]))),
                  ev=(float(np.mean(per["amb_ev"])), float(np.mean(per["bus_ev"]))), ev_sd=(float(np.std(per["amb_ev"])), float(np.std(per["bus_ev"]))),
                  per_eval=per)
    print(f"{arm}: n={n} | 停车时间/visit 救护车 {res[arm]['pv'][0]:.2f} ± {res[arm]['pv_sd'][0]:.2f} (逐评估 {np.round(per['amb_pv'],2).tolist()}, 收集器汇总 {np.round(per['amb_log'],2).tolist()}) | 公交 {res[arm]['pv'][1]:.2f} ± {res[arm]['pv_sd'][1]:.2f} (逐评估 {np.round(per['bus_pv'],2).tolist()})")
    print(f"      停车次数/visit 救护车 {res[arm]['ev'][0]:.3f} ± {res[arm]['ev_sd'][0]:.3f} | 公交 {res[arm]['ev'][1]:.3f} ± {res[arm]['ev_sd'][1]:.3f}")
json.dump(res, open("experiments/analysis/data/typeswap_02h_pervisit_2026-09-30.json","w"), indent=1)
