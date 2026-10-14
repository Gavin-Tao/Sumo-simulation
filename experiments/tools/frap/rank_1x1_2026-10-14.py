# 仓库副本 (2026-10-14): 中间文件在 analysis/data/synthesis_1x1_2026-10-14/ (会话中原在临时目录); 逐集行为记录合并为 behav_out.jsonl。
"""1x1 开 F1 各臂总排名 (只读): 末段均值 (5 检查点 × 10 种子) 分车类停车、回报 (best+末检查点 20 集)、训练曲线学习速度; 加长臂与基线。"""
import json, sys, numpy as np
sys.argv = ["x"]; exec(open("/home/xiaowen/sumo-rl/experiments/tools/frap/factor_pairs_1x1_2026-10-14.py").read().split("F = {}")[0])
C = json.load(open(f"{S}/curves_1x1.json"))
rows = []
for e, r in M.items():
    if not r["f"]["F1"]: continue
    L, eps = late(int(e))
    rl_ = beh(e, "late"); rb = beh(e, "best")
    ret = float(np.nanmean(rl_)) if rl_ is not None else np.nan          # 主度量: 末段 5 个检查点的训练奖励回报
    ret_best = float(np.nanmean(rb)) if rb is not None else np.nan
    c = C.get(e) or C.get(int(e)); t30 = None; early = np.nan
    if c:
        E = np.array(c["E"], float); V = np.array(c["stop"], float); ma = np.convolve(V, np.ones(10) / 10, mode="valid")
        t30 = int(E[9:][np.argmax(ma <= 30)]) if (ma <= 30).any() else None; early = float(np.mean(V[(E > 0) & (E <= 300)]))
    f = r["f"]; arch = f["arch"] + ("2层" if f["arch"] == "枚举条件化" and f["layers"] == 2 else ("1层" if f["arch"] == "枚举条件化" else ""))
    rows.append(dict(e=e, arch=arch, act=("8" if "8标准" in f["act"] else "12"), enc=("legacy" if f["enc"] == "legacy" else "12槽"), wait=f["wait"], ge=(f["green_el"] != "无"), ep=f["episodes"],
                     late=L[:, 0].mean() if L is not None else np.nan, car=L[:, 1].mean() if L is not None else np.nan, bus=L[:, 2].mean() if L is not None else np.nan,
                     amb=np.nanmean(L[:, 3]) if L is not None else np.nan, best=np.mean(r["best"]["stop_all"]), fin=np.mean(r["final"]["stop_all"]) if r.get("final") else np.nan,
                     ret=ret, ret_best=ret_best, t30=t30, early=early, sd_ck=np.nan))
json.dump(rows, open(f"{S}/rank_1x1.json", "w"), ensure_ascii=False, default=float)
rows.sort(key=lambda x: -x["ret"])
print(f"{'exp':4s} {'结构':10s}{'相位':3s} {'编码':6s} {'等待':3s} {'时长':3s} | {'回报':>7s} | 末段 全部 小汽车 公交 救护车 | best  末检查点 | 首300集 均线≤30")
for x in rows:
    print(f"{x['e']:4s} {x['arch']:10s}{x['act']:3s} {x['enc']:6s} {x['wait']:3s} {'有' if x['ge'] else '-':3s} | {x['ret']:7.2f} | {x['late']:5.2f} {x['car']:5.2f} {x['bus']:5.2f} {x['amb']:5.2f} | {x['best']:5.2f} {x['fin']:5.2f} | {x['early']:5.1f} {x['t30']}")
