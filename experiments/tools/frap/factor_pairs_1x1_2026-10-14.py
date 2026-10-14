# 仓库副本 (2026-10-14): 中间文件在 analysis/data/synthesis_1x1_2026-10-14/ (会话中原在临时目录); 逐集行为记录合并为 behav_out.jsonl。
"""1x1 逐因素配对分析 (只读, 2026-10-14)。每对 (A, B) 只差一个因素; 同一评估种子 123-132 配对。
度量 (都换算成 "A 相对 B 的改善 %", 正 = A 更好):
  停车 best / 停车 末检查点 / 停车 末段均值 (最后 5 个检查点 × 10 种子, 每种子先对 5 个检查点取平均) / 训练奖励回报 末段均值 (同样 5 × 10; 对照臂缺末段时用 best) /
  末段小汽车停车 / 末段公交停车。
单项判读 (同 §4): |改善| ≤ 5% 等价; > 5% 且同向种子 ≥ 7/10 → 更好/更差; 否则 不确定。
综合判读: 以 "末段均值" 与 "回报" 为主 (前者去掉单个检查点的波动, 后者是方法实际优化的目标);
  两者同为更好 (或同为更差) → 强; 一个更好另一个等价 → 中; 方向相反 → 不确定 (标出); 都等价 → 等价。
用法: python factor_pairs_1x1.py [输出 json]"""
import json, glob, os, sys, numpy as np
S = "/home/xiaowen/sumo-rl/experiments/analysis/data/synthesis_1x1_2026-10-14"
EXP = "/home/xiaowen/sumo-rl/experiments"
PV, PL = f"{EXP}/analysis/data/pervehicle", f"{EXP}/analysis/data/pervehicle_late_2026-10-14"
SEEDS = list(range(123, 133)); M = json.load(open(f"{S}/master_1x1.json"))

def summ(path):
    s = json.load(open(path))["collector"]["summary"]
    return [float(s.get(f"eval/system/{c}/avg_stopped_time_per_visit", np.nan)) for c in ("all", "car", "bus", "ambulance")]

_late = {}
def late(e):
    """(10 种子, 4 车类) 末段均值; 用到的检查点集合。"""
    if e in _late: return _late[e]
    eps = {}
    for f in glob.glob(f"{PL}/exp{e}_ep*_seed*_late_coll.json"):
        ep = int(f.split("_ep")[1].split("_")[0]); sd = int(f.split("_seed")[1].split("_")[0]); eps.setdefault(ep, {})[sd] = f
    fin = M[str(e)]["final"]["ep"] if M[str(e)].get("final") else None
    if fin:
        for f in glob.glob(f"{PV}/exp{e}_ep{fin:05d}_seed*_ckpt_coll.json"):
            sd = int(f.split("_seed")[1].split("_")[0]); eps.setdefault(fin, {})[sd] = f
    full = sorted(ep for ep, d in eps.items() if all(s in d for s in SEEDS))
    if len(full) < 4: _late[e] = (None, full); return _late[e]
    arr = np.array([[summ(eps[ep][s]) for ep in full] for s in SEEDS])   # (10, n_ck, 4)
    _late[e] = (np.nanmean(arr, axis=1), full); return _late[e]

_beh = None
def _load_beh():
    global _beh
    _beh = {}
    for line in open(f"{S}/behav_out.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        if d["pol"] != "rl": continue
        b = os.path.basename(d["ckpt"]); ck = "best" if "best" in b else int(b.split("ckpt_ep")[1].split(".")[0])
        _beh.setdefault((str(d["exp"]), d["interv"]), {}).setdefault(ck, {})[d["seed"]] = d
def beh(e, which, key="ret", interv="none"):
    """which: "best" | "final" (最后检查点) | "late" (末段 5 个检查点逐种子平均)。"""
    if _beh is None: _load_beh()
    d = _beh.get((str(e), interv), {})
    ok = lambda ck: ck in d and all(s in d[ck] for s in SEEDS)
    vec = lambda ck: np.array([d[ck][s][key] if d[ck][s][key] is not None else np.nan for s in SEEDS], float)
    fin = M[str(e)]["final"]["ep"] if M[str(e)].get("final") else (1500 if ok(1500) else None)
    if which == "best": return vec("best") if ok("best") else None
    if which == "final": return vec(fin) if fin and ok(fin) else None
    eps = [fin - 200, fin - 150, fin - 100, fin - 50, fin] if fin else []
    if not eps or not all(ok(ep) for ep in eps): return None
    return np.nanmean(np.array([vec(ep) for ep in eps]), axis=0)

def stopvec(e, which):
    r = M[str(e)].get(which)
    return None if not r else np.array(r["stop_all"], float)

def judge(imp, wins, n=10):
    if imp is None: return "缺"
    if abs(imp) <= 5: return "等价"
    if imp > 0 and wins >= 7: return "更好"
    if imp < 0 and (n - wins) >= 7: return "更差"
    return "不确定"

def cmp(a, b, higher_better=False):
    if a is None or b is None: return None, None, "缺"
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if higher_better: imp = 100 * (a.mean() - b.mean()) / abs(b.mean()); wins = int((a > b).sum())
    else: imp = 100 * (b.mean() - a.mean()) / b.mean(); wins = int((a < b).sum())
    return float(imp), wins, judge(imp, wins, len(a))

def pair(A, B):
    out = {"A": A, "B": B}
    out["best"] = cmp(stopvec(A, "best"), stopvec(B, "best"))
    out["final"] = cmp(stopvec(A, "final"), stopvec(B, "final"))
    la, lb = late(A)[0], late(B)[0]
    out["late"] = cmp(None if la is None else la[:, 0], None if lb is None else lb[:, 0])
    out["late_car"] = cmp(None if la is None else la[:, 1], None if lb is None else lb[:, 1])
    out["late_bus"] = cmp(None if la is None else la[:, 2], None if lb is None else lb[:, 2])
    out["ret_best"] = cmp(beh(A, "best"), beh(B, "best"), higher_better=True)
    out["ret_final"] = cmp(beh(A, "final"), beh(B, "final"), higher_better=True)
    # 回报 (主度量, 2026-10-14 审查后改): 末段 5 个检查点逐种子平均, 与 "末段停车" 同口径; 不开 F1 的对照臂没有末段回报时退回 best 对 best (标注)
    la_r, lb_r = beh(A, "late"), beh(B, "late")
    if la_r is not None and lb_r is not None:
        out["ret"] = cmp(la_r, lb_r, higher_better=True); out["ret_src"] = "末段"
    else:
        out["ret"] = cmp(beh(A, "best"), beh(B, "best"), higher_better=True); out["ret_src"] = "best"
    v1, v2 = out["late"][2], out["ret"][2]
    if v1 == "缺" or v2 == "缺": out["verdict"] = f"仅{v1 if v2 == '缺' else v2}" if not (v1 == "缺" and v2 == "缺") else "缺"
    elif v1 == v2 and v1 in ("更好", "更差"): out["verdict"] = v1 + "(强)"
    elif {v1, v2} <= {"更好", "等价"} and "更好" in (v1, v2): out["verdict"] = "更好(中)"
    elif {v1, v2} <= {"更差", "等价"} and "更差" in (v1, v2): out["verdict"] = "更差(中)"
    elif v1 == v2 == "等价": out["verdict"] = "等价"
    elif {v1, v2} == {"更好", "更差"}: out["verdict"] = "矛盾"
    else: out["verdict"] = "不确定"
    return out

F = {}
def fam(name, pairs): F[name] = pairs
# ---- 家族定义 (A = 加了该因素 / 新选项, B = 对照) ----
fam("F1 (开 vs 不开)", [(371, 353), (372, 360), (373, 327), (374, 256), (375, 335), (377, 366), (378, 363), (380, 367), (381, 364), (382, 370)])
fam("编码 legacy vs 12槽 · MLP", [(374, 377), (375, 378), (376, 379), (446, 437), (434, 425)])
fam("编码 legacy vs 12槽 · 枚举条件化1层", [(399, 393), (400, 394), (401, 395), (396, 390), (397, 391), (398, 392), (447, 438), (435, 426), (448, 442), (436, 430)])
fam("编码 legacy vs 12槽 · 枚举条件化2层", [(414, 408), (415, 409), (416, 410), (411, 406), (412, 389), (413, 407)])
LIN_LOG = [(374, 375), (377, 378), (380, 381), (383, 384), (402, 386), (404, 387), (390, 391), (393, 394), (396, 397), (399, 400), (406, 389), (408, 409),
           (411, 412), (414, 415), (419, 420), (422, 423), (437, 425), (438, 426), (439, 427), (440, 428), (441, 429), (442, 430), (443, 431), (444, 432), (445, 433), (446, 434), (447, 435), (448, 436)]
fam("等待刻度 对数 vs 线性", [(b, a) for a, b in LIN_LOG])
fam("等待刻度 两种 vs 对数", [(376, 375), (379, 378), (382, 381), (385, 384), (403, 386), (405, 387), (392, 391), (395, 394), (398, 397), (401, 400), (407, 389), (410, 409), (413, 412), (416, 415), (421, 420), (424, 423)])
fam("等待刻度 两种 vs 线性", [(376, 374), (379, 377), (382, 380), (385, 383), (403, 402), (405, 404), (392, 390), (395, 393), (398, 396), (401, 399), (407, 406), (410, 408), (413, 411), (416, 414), (421, 419), (424, 422)])
fam("绿灯时长 (加 vs 不加) · 对数", [(425, 378), (426, 394), (427, 386), (428, 387), (429, 423), (430, 391), (431, 381), (432, 384), (433, 420), (434, 375), (435, 400), (436, 397)])
fam("绿灯时长 (加 vs 不加) · 线性", [(437, 377), (438, 393), (439, 402), (440, 404), (441, 422), (442, 390), (443, 380), (444, 383), (445, 419), (446, 374), (447, 399), (448, 396)])
fam("结构 集合式相位头 vs MLP (8 标准相位, 同观测)", [(422, 377), (423, 378), (424, 379), (441, 437), (429, 425)])
fam("结构 集合式相位头 vs FRAP", [(422, 402), (423, 386), (424, 403), (419, 380), (420, 381), (421, 382), (441, 439), (429, 427), (445, 443), (433, 431)])
fam("结构 集合式相位头 vs 枚举条件化2层", [(422, 408), (423, 409), (424, 410), (419, 406), (420, 389), (421, 407)])
fam("结构 枚举条件化2层 vs MLP (8 标准相位)", [(408, 377), (409, 378), (410, 379), (414, 374), (415, 375), (416, 376)])
fam("结构 枚举条件化1层 vs MLP (8 标准相位)", [(393, 377), (394, 378), (395, 379), (399, 374), (400, 375), (401, 376), (426, 425), (438, 437), (435, 434), (447, 446)])
fam("结构 FRAP vs MLP (8 标准相位)", [(402, 377), (386, 378), (403, 379), (439, 437), (427, 425)])
fam("结构 GS-G vs FRAP (切换代价项)", [(383, 380), (384, 381), (385, 382), (404, 402), (387, 386), (405, 403), (440, 439), (428, 427), (444, 443), (432, 431)])
fam("枚举条件化 2层 vs 1层", [(408, 393), (409, 394), (410, 395), (406, 390), (389, 391), (407, 392), (414, 399), (415, 400), (416, 401), (411, 396), (412, 397), (413, 398)])
fam("动作集合 12 枚举 vs 8 标准", [(390, 393), (391, 394), (392, 395), (396, 399), (397, 400), (398, 401), (406, 408), (389, 409), (407, 410), (411, 414), (412, 415), (413, 416),
                                (380, 402), (381, 386), (382, 403), (383, 404), (384, 387), (385, 405), (419, 422), (420, 423), (421, 424),
                                (430, 426), (431, 427), (432, 428), (433, 429), (442, 438), (443, 439), (444, 440), (445, 441), (436, 435), (448, 447)])
fam("FRAP 加大 10 倍 vs 原", [(388, 381)])

res = {k: [pair(a, b) for a, b in v] for k, v in F.items()}
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/factor_pairs_1x1_rerun.json", "w"), ensure_ascii=False)
fmt = lambda t: "   缺   " if t[0] is None else f"{t[0]:+5.1f}% {t[1]:2d}"
for k, rows in res.items():
    vs = [r["verdict"] for r in rows]; lv = [r["late"][0] for r in rows if r["late"][0] is not None]; rv = [r["ret"][0] for r in rows if r["ret"][0] is not None]
    print(f"\n### {k}  ({len(rows)} 对)  综合: " + ", ".join(f"{v} {vs.count(v)}" for v in dict.fromkeys(vs)) +
          f" | 末段改善中位 {np.median(lv):+.1f}%" * bool(lv) + f" | 回报改善中位 {np.median(rv):+.1f}%" * bool(rv))
    for r in rows:
        print(f"   {r['A']}对{r['B']}: best {fmt(r['best'])} | 末检查点 {fmt(r['final'])} | 末段 {fmt(r['late'])} | 回报({r.get('ret_src','')}) {fmt(r['ret'])} | 末段 小汽车 {fmt(r['late_car'])} 公交 {fmt(r['late_bus'])} → {r['verdict']}")
