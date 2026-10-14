# 仓库副本 (2026-10-14): 中间文件在 analysis/data/synthesis_1x1_2026-10-14/ (会话中原在临时目录); 逐集行为记录合并为 behav_out.jsonl。
"""1x1 全部有 10 种子评估的臂: 因素向量 + best / 最后检查点指标 + 训练后诊断 → JSON。只读。"""
import yaml, glob, os, re, json, numpy as np
PV = "analysis/data/pervehicle"; DG = "analysis/data/posttrain_diag"
SEEDS = list(range(123, 133)); SH = {"car": .965, "bus": .033, "ambulance": .0017}; WT = {"car": 1, "bus": 3, "ambulance": 5}
have = sorted({int(m.group(1)) for f in os.listdir(PV) if (m := re.match(r"exp(\d+)_ep\d+_seed\d+_best_coll\.json", f))})
def metrics(p):
    s = json.load(open(p))["collector"]["summary"]; g = lambda c, m: float(s.get(f"eval/system/{c}/{m}", np.nan))
    o = {f"stop_{c}": g(c, "avg_stopped_time_per_visit") for c in ("all", "car", "bus", "ambulance")}
    o.update({f"ev_{c}": g(c, "avg_stop_events_per_visit") for c in ("all", "car", "bus")})
    o["tl_all"] = g("all", "avg_time_loss_per_visit") if f"eval/system/all/avg_time_loss_per_visit" in s else np.nan
    o["J"] = sum(SH[c] * WT[c] * np.nan_to_num(o[f"stop_{c}"]) for c in SH); return o
def ckfiles(e, kind):
    if kind == "best":
        fs = glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll.json")
        if not fs: return None, {}
        ep = max(int(f.split("_ep")[1].split("_")[0]) for f in fs); fs = [f for f in fs if f"_ep{ep:05d}_" in f]
    else:
        fs = glob.glob(f"{PV}/exp{e}_ep*_seed*_ckpt_coll.json")
        if not fs: return None, {}
        ep = max(int(f.split("_ep")[1].split("_")[0]) for f in fs); fs = [f for f in fs if f"_ep{ep:05d}_" in f]
    d = {int(f.split("_seed")[1].split("_")[0]): f for f in fs}
    return ep, d
def factors(c):
    fr = c.get("frap") or {}
    gs = c.get("action_scheme") == "enum_frap"
    arch = ("8STD-MLP" if not gs else {"frap": "FRAP", "gsg": "GS-G", "menuq": "枚举条件化", "setq": "集合式相位头", "mtt": "MTT"}[fr.get("arch", "frap")])
    if gs and fr.get("score") == "pressure": arch += "+压力打分"
    em = c.get("enum_meta_file", "")
    act = "8标准(8STD路网)" if not gs else ("8标准(枚举路网)" if "std8" in em else ("枚举12" if em.endswith("syc_4phases_enum_meta.json") else os.path.basename(em)))
    fl = c.get("obs_fields", [])
    wait = "两种" if "mean_awt_log" in fl else ("对数" if c.get("obs_awt_log") else "线性")
    enc = c.get("obs_phase_state", "legacy")
    hdr = enc == "legacy" or c.get("obs_header", True) is not False
    ge = "无"
    if c.get("obs_green_elapsed"):
        ge = ("逐运动连续" + (f"cap{c.get('obs_green_elapsed_cap')}" if c.get("obs_green_elapsed_cap") else "") + ("精确" if c.get("obs_green_elapsed_exact") else "") +
              ("含黄" if c.get("obs_green_elapsed_include_yellow") else "") + ("相位清零" if c.get("obs_green_elapsed_reset") else ""))
    layers = fr.get("menu_layers", 2 if fr.get("arch") == "menuq" else None) if fr.get("arch") == "menuq" else (fr.get("setq_layers") if fr.get("arch") == "setq" else None)
    return dict(arch=arch, act=act, enc=enc, header=("有" if hdr else "无") if enc == "perphase" else "legacy头", slot_mg=bool(c.get("obs_slot_min_green")),
                wait=wait, since_green=bool(c.get("obs_since_green")), green_el=ge, psi=bool(c.get("obs_downstream")), occ=bool(c.get("obs_lane_occ")),
                F1=bool(c.get("mask_min_green")), layers=layers, embed=fr.get("embed_dim") if gs else None, pair=fr.get("pair_dim") if gs else None,
                gsg_ti=fr.get("gsg_time_inputs", "full") if fr.get("arch") == "gsg" else None, gsg_sign=fr.get("gsg_cost_sign"), header_in=bool(fr.get("header_in")),
                episodes=c.get("episodes"), seed=c.get("seed"))
out = {}
for e in have:
    fs = [f for f in glob.glob(f"configs/exp{e}_*.yaml") if not re.search("traindry|smoke", f)]
    if not fs: continue
    c = yaml.safe_load(open(fs[0]))
    if "1x1" not in c.get("net_file", ""): continue
    r = {"name": c["name"], "f": factors(c)}
    for kind in ("best", "final"):
        ep, d = ckfiles(e, kind)
        if ep is None or not all(s in d for s in SEEDS): r[kind] = None; continue
        ms = [metrics(d[s]) for s in SEEDS]
        r[kind] = {"ep": ep, **{k: [m[k] for m in ms] for k in ms[0]}}
    for tag in ("best", "ep1500", "ep3000"):
        p = f"{DG}/exp{e}_{tag}.json"
        if os.path.exists(p): r[f"diag_{tag}"] = json.load(open(p))["mean"]
    out[e] = r
json.dump(out, open("/home/xiaowen/sumo-rl/experiments/analysis/data/synthesis_1x1_2026-10-14/master_1x1.json", "w"), ensure_ascii=False)
print(len(out))
for e, r in out.items():
    b = r["best"]; fi = r["final"]; f = r["f"]
    bs = f"{np.mean(b['stop_all']):6.2f}" if b else "  -   "; fs_ = f"{np.mean(fi['stop_all']):6.2f}@{fi['ep']}" if fi else "   -  "
    print(e, f"{f['arch']:14s} {f['act']:14s} {f['enc']:8s} {f['header']:5s} mg{int(f['slot_mg'])} {f['wait']:3s} sg{int(f['since_green'])} ge:{f['green_el']:14s} F1{int(f['F1'])} L{f['layers']} E{f['embed']} ep{f['episodes']} s{f['seed']}  best {bs} (ep{b['ep'] if b else '-'}) final {fs_}")
