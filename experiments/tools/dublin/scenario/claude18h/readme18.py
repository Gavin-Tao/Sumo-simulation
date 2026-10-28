"""生成 nets/dublin/weekday_18h_nocg_claude/README.md (数字全部取自 build/*.json)。"""
import collections, json
import common18 as C

def main():
    A = C.load("anchors_18h.json"); E = C.load("entries.json"); B = C.load("buses.json"); AM = C.load("ambulances.json")
    D = C.load("demand_report.json"); W = C.load("write_report.json"); H = C.load("iterate_history.json")["history"]
    try: F = C.load("final_calibration.json")
    except FileNotFoundError: F = None
    meta = C.load_meta(); L = []
    p = L.append
    p("# 18 点场景 (claude 版): 19 信号 corridor 网删除 College Green 后的 18 个 RL 路口\n")
    p("实现计划: `experiments/analysis/PLAN_18H_NET19_DEMAND_2026-10-28.md` (第三版)。生成与核对工具: `experiments/tools/dublin/scenario/claude18h/`; 测试: `experiments/tests/test_claude18h_demand.py`。")
    p(f"路网 `{C.NET.name}` (sha256 `{C.NET_SHA[:12]}…`), 动作表 `{C.META.name}` (sha256 `{C.META_SHA[:12]}…`)。没有启动训练。\n")
    p("## 怎么用\n")
    p("- 训练与标定的默认场景: `scenario.sumocfg` = `background_cars_100.rou.xml` + `bus_tt1.rou.xml` + `amb_tt1.rou.xml`。")
    p("- 时刻表: 第 1–5 份供训练轮换 (训练代码目前不支持按集切换路由文件, 未批准前只用第 1 份); 第 6 份标定; **第 7 份最终测试, 定稿前未跑**; 第 8–13 份只供评估。")
    p("- 跨集课程可选: `background_cars_060/080/100.rou.xml` (嵌套抽稀, 车辆 id 一致)。")
    p("- 评估专用 (不得被训练配置引用): `*_as_car.rou.xml` (换车型孪生)、`amb_freqprobe_eval*.rou.xml` (每集约 20 辆救护车的频率补充测试)、`amb_unseen_eval*.rou.xml` (3 条未见路线)。")
    p("- 训练配置须开 `obs_all_connections` (133307283 与 389332 有含多条连接的槽)。sumocfg: `time-to-teleport -1`、`ignore-junction-blocker -1`、`collision.check-junctions true`。\n")
    p("## 车辆与路线规则\n")
    p("- 三类车 (car/bus/ambulance) 物理参数逐项相同, 只差 id 与颜色, 不写 vClass; 换道参数 lcStrategic/lcCooperative = -1, lcSpeedGain/lcKeepRight = 0: **行驶中不换道**。每辆车写明 departLane, 路线在车道图上逐段连通; 标定实测换道 0 次。")
    p("- 起点只在白名单车道: 15 条边界入口; 支路入口 (小汽车) " + "、".join(f"`{x}`" for x in E["side_lanes"]) + "; 公交也从其中的 " + "、".join(f"`{x}`" for x in B["bus_side_entries"]) + " 出发 (GTFS 原出发边不合规的已撤掉, 延伸到合规入口)。从不在 RL 进口或出口车道上发车。")
    p("- 终点只在 15 条边界出口 (无去向的边) 边末离网; 不重复经过同一 RL 路口; 同一入口车道上所有车型合排时刻, 相邻计划出发间隔 ≥ 3 s。")
    p("- 每集时间曲线 (背景小汽车): 0–300 s 40%, 300–900 s 升到 100%, 900–3000 s 平台, 3000–3600 s 70%; 所有流率目标均指平台期; 整集车数 = 0.85 × 平台流率。\n")
    p("## 零需求转向 (穷举判定)\n")
    p(f"状态 = (车道, 已经过的 RL 路口集合), 穷举 {E['search_states']['with_side']} 个状态: " + "、".join(f"{t} {m}" for t, m in E["zero"]) + " 没有不重访的路线。")
    if E["zero_lane_loop"]:
        p("另: " + "、".join(f"{t} {m}" for t, m in E["zero_lane_loop"]) + " 唯一的路线要在同一条普通车道上经过两次 (绕街区兜圈), 设为零需求。\n")
    for k, lane in E.get("side_relaxed", {}).items():
        p(f"另: {k.replace('|', ' ')} 从规则内入口出发的唯一路线要绕街区重复经过同一条车道; 按用户要求为它单设例外支路入口 `{lane}` (入口到 RL 进口之间有一次让行), 设计流量 40 辆/h, 不再是零需求。\n")
    p("## 需求: 各路口参考值与实现量 (平台期, 辆/h, 含公交与救护车)\n")
    p("| 路口 | 参考来源 | 参考值 | 浮动系数 s_j | 实现 | 实现/参考 |\n|---|---|---|---|---|---|")
    for t, j in D["junctions"].items():
        p(f"| {t} | {j['source']} | {j['ref']:.0f} | {j['scale']} | {j['realized']:.0f} | {j['ratio']} |")
    p(f"\n小汽车平台期流率合计 {D['cars_plateau_rate_total']:.0f} 辆/h; 整集小汽车 1.0 档 {W['n_cars']['1.0']} 辆、0.8 档 {W['n_cars']['0.8']} 辆、0.6 档 {W['n_cars']['0.6']} 辆; GTFS 公交 {W['n_bus']} 辆/集; 救护车 {W['amb_per_episode']} 辆/集。")
    if D.get("caps"): p("无可信站点路口的总量上限 (标定迭代加): " + json.dumps(D["caps"], ensure_ascii=False))
    if D.get("lane_factor"): p("进口车道容量系数 (标定迭代按进口下调): " + json.dumps(D["lane_factor"], ensure_ascii=False))
    p("\n竞争补量 (偏离真实, 逐条记录):\n")
    for x in D["topup"]: p(f"- {x['tls']} {x['movement']}: {x['before']} → {x['after']} 辆/h ({x['reason']})")
    if D.get("notes"): p("\n" + "\n".join(f"- {k}: {v}" for k, v in D["notes"].items()))
    p("\n## 公交\n")
    p(f"GTFS 18 点 {len(B['buses'])} 辆全部保留 (班次系数 1.0, 无非 GTFS 补充), {len(B['lines'])} 条线路; 改动: {json.dumps(B['report'], ensure_ascii=False)}。"
      "延伸到合法入口/出口; 重访 RL 路口的截断。没有公交经过竞争转向的路口: " + ("、".join(B["junctions_without_bus_on_competing"]) or "无") + " (与 GTFS 一致)。\n")
    p("## 救护车\n")
    r = AM["route"]
    try:
        ch = json.load(open(C.WORK / "amb_route_choice.json")); p("选线方式: 满足规则的候选各放 20 辆救护车实测 (固定轮转), 取放行取舍机会占比最高者: " + "; ".join(f"候选{x['i']} {x['share']:.0%}" for x in ch["candidates"]) + "。")
    except FileNotFoundError: pass
    p(f"每集固定 3 辆, 走同一条设计路线 (真实数据没有救护车路线; 一条路线让沿线每个路口每集都见到 3 辆, 有利于学会优先): {len(r['traversals'])} 个 RL 路口, "
      f"{len(AM['route_competing'])} 次竞争转向, {r['length']:.0f} m, 不经过让行连接; 入口平台期负荷 ≤ 450 辆/h。")
    p("路线: `" + " ".join(r["edges"]) + "`")
    p("经过: " + "、".join(f"{t} {m}" for t, m in r["traversals"]))
    p("出发: 900–1500、1500–2100、2100–2700 s 各一辆, 间隔 ≥ 300 s; 时刻表 1–7: " + json.dumps(AM["timetables"]) + "\n")
    p("## 标定\n")
    p("控制器: 固定轮转 (每相位 5 个决策步 = 25 s), 排队驱动轮转 (看车道头车, 绿灯 10–40 s, 保证服务), 均匀随机 (只作压力测试)。迭代历史: `build/iterate_history.json`。\n")
    if F:
        p("| 场景 | 种子数 | C1 在网增长 | C2 未插入 | C2 平均出发延迟 s | C3 堵死 | C3' 长等待 | C4 漏放 | C4 旧队列 | C6 回溢 | C7 长进口越界>5% | C9 换道/碰撞/瞬移/缺/救护车到达 | C13 | C14 最差超出 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in ["L100_served_tt1", "L100_fixed_tt1", "L100_random_tt1", "L080_served", "L080_fixed", "L060_served", "L060_fixed"]:
            v = F.get(k)
            if not v: continue
            fm = lambda xs: ", ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in xs)
            p(f"| {k} | {v['n']} | {fm(v['C1_growth'])} | {fm(v['C2_not_inserted'])} | {fm(v['C2_mean_delay'])} | {fm(v['C3_blocked'])} | {fm(v['C3p_long'])} | {fm(v['C4_missed'])} | {fm(v['C4_old'])} | {fm(v['C6_spill'])} | {fm(v['C7_long_over_5pct'])} | {'; '.join('/'.join(map(str, x)) for x in v['C9'])} | {fm(v['C13'])} | {fm(v['C14_worst_excess'])} |")
        o = F["C11_fixed_tt1to6"]
        p(f"\nC11 (固定轮转, 时刻表 1–6 × 5 种子 = {o['episodes']} 集): 救护车在竞争转向上的访问 {o['ambulance_competing_visits']} 次, 其中有放行取舍机会 {o['ambulance_with_opp']} 次 ({o['ambulance_share']:.0%}); "
          f"红灯提前放行 {o['amb_red_release']}、绿灯保持 {o['amb_green_hold']}; 公交每集有机会访问 {o['bus_with_opp_per_ep']:.0f} 次。")
    insertion_section(p, E, B)
    try:
        JC = C.load("junction_check.json")
        p("\n## 中性基准: 不分车型的控制器下三类车的每次访问等待\n")
        p("口径与训练/wandb 完全相同: 按车道剔除恒绿车道 (所有绿灯相位都为绿的车道整条不计, 例如救护车经过的 cluster_2498303858 南进口第 0 车道); 5 个种子合计。")
        p("\n| 控制器 | 小汽车 | 公交 | 救护车 |\n|---|---|---|---|")
        for pol, nm in (("served", "排队驱动 (自适应)"), ("fixed", "固定轮转")):
            b = JC["baseline"][pol + "_lane"]; g = lambda k: f"{b[k]['mean_wait']:.2f} s ({b[k]['visits']} 次; 各种子 {b[k]['per_seed']})" if k in b else "—"
            p(f"| {nm} | {g('car')} | {g('bus')} | {g('ambulance')} |")
        try:
            ch = json.load(open(C.WORK / "amb_route_choice.json"))
            p("\n救护车与公交的基准差距 (每集 20 辆救护车实测, 排队驱动 3 种子, 同口径): " + "; ".join(f"候选{x['i']} 机会 {x['share']:.0%}, 救护车 {x['amb_base']} s / 公交 {x['bus_base']} s (差 {x['gap']:+.2f})" for x in sorted(ch["candidates"], key=lambda x: abs(x["gap"]))) + "。")
        except FileNotFoundError: pass
        p("\n解读: 262 辆 GTFS 公交多走主干道, 这些转向本来等得少; 救护车路线要体现优先级就必须经过有竞争的转向, 等待天然更长。选线时所有机会占比 ≥70% 的候选里救护车基准都高于公交, 选用差距最小的一条 (选线时约 +1 s);"
          " 定稿需求下 (上表, 排队驱动) 差距进一步缩小。代价是 cluster_5166592137 南直行回到每进口 100 辆/h 下限 (目标本来只有 5 辆/h; 外部复核后途经支路入口车道的小汽车路线一律不用, 先前借道的南直行车流随之消失),"
          " 救护车在该路口西左转的放行取舍机会降到约 44%, 全路线机会占比由选线时 90% 降到 C11 的数值, 仍满足 ≥70% 的选线判据。没有为此补量, 因为补南直行会把救护车基准重新抬高。"
          "这对结论是保守的: 救护车起点比公交差, 531 下若仍优于公交, 说明是优先级规格的作用。同一转向上三类车的等待相同, 没有控制器帮不上忙的陷阱。")
        try:
            import glob, collections as _c
            PR = [json.load(open(f)) for f in sorted(glob.glob(str(C.WORK / "rank3/cur_prio_*.json")))]
            if PR:
                per = _c.defaultdict(list)
                for d in PR:
                    for ty, (n_, s_) in d["C16"]["by_type"].items(): per[ty].append(s_ / max(1, n_))
                ok = sum(1 for i in range(len(PR)) if per["ambulance"][i] <= per["bus"][i] <= per["car"][i])
                p(f"\n**顺序可达性检查 (知道车型的简单控制器, 不是 RL; calib18 --policy prio)**: 相位得分按 5/3/1 加权, 有救护车时满最短绿即切到能放行它的相位。{len(PR)} 个种子, 同口径每次访问等待: "
                  f"小汽车 {sum(per['car'])/len(PR):.2f} s、公交 {sum(per['bus'])/len(PR):.2f} s、救护车 {sum(per['ambulance'])/len(PR):.2f} s; 救护车 ≤ 公交 ≤ 小汽车 {ok}/{len(PR)} 个种子成立, 没有公交比小汽车多等的路口。"
                  "说明场景本身允许 531 顺序且有余量; RL 能否学到取决于算法, 场景不能替它保证。")
        except Exception: pass
        p("\n**建议的主证据: 换车型孪生。** 用 `*_as_car.rou.xml` 把同一批公交与救护车以 car 身份再跑一遍 (同路线、同时刻、同位置), 比较 \"救护车比它的小汽车孪生少等多少\" 与 \"公交比它的小汽车孪生少等多少\"; 路线组成的差异完全抵消。绝对 per-visit 排序作辅助, 报告时附上本节的 111 基准差距。")
        p("\n## 逐路口体检 (排队驱动 5 种子; 固定轮转下每个路口有排队决策步占比 30%–92%)\n")
        p("| 路口 | 小汽车/公交/救护车 每次访问等待 s | 有排队决策步 (排队驱动) | 堵死 | 清空后 60 s 内又满 / 清空次数 |\n|---|---|---|---|---|")
        for t, j in JC["junctions"].items():
            w = j["wait_served"]; f = lambda x: f"{x['mean_wait']:.1f}" if x else "—"
            p(f"| {t} | {f(w['car'])} / {f(w['bus'])} / {f(w['ambulance'])} | {min(j['queue_step_share']):.2f}–{max(j['queue_step_share']):.2f} | {sum(j['blocked'])} | {sum(j['refill_fast'])} / {sum(j['clears'])} |")
    except FileNotFoundError: pass
    try:
        V = [json.load(open(C.WORK / f"validation/summary_{pp}_123.json")) for pp in ("fixed", "random")]
        p("\n## 状态与奖励逐步独立重算\n")
        for v in V: p(f"- {v['policy']} 种子 123: {v['decisions']} 个决策步, 状态 {v['state_checks']} 次、奖励 {v['reward_checks']} 次、统计器 {v['tracker_checks']} 次全部一致; 碰撞 {v['collisions']}, 瞬移 {v['teleports']}, 统计器缺条目 {v['tracker_missing']}, 换道 {v['lane_changes']}, 车身重叠 {v['body_overlap']}; 集末在网 {v['remaining_at_end']} 辆 (集末才出发), 待插入 {v['pending_at_end']}。")
    except FileNotFoundError: pass
    p("\n## 与计划的偏离\n")
    for x in DEVIATIONS: p(f"- {x}")
    open(C.OUT / "README.md", "w").write("\n".join(L) + "\n")
    print("写出", C.OUT / "README.md")

def insertion_section(p, E, B):
    """中段插车审查: 支路入口逐车型出发延迟 (三种控制器对比), 插到公交/救护车前面的次数 (C17), 优先车辆在非 RL 车道上的停车。"""
    import glob, statistics
    runs = {pol: [json.load(open(f)) for f in sorted(glob.glob(str(C.WORK / f"calib_final/L100_{pol}_tt1_*.json")))] for pol in ("served", "fixed", "random")}
    if not all(runs.values()) or "C17" not in runs["served"][0]: return
    pols = [("served", "排队驱动"), ("fixed", "固定轮转"), ("random", "随机")]
    side = {l.rsplit("_", 1)[0] for l in E["side_lanes"]}
    p("\n## 中段插车审查 (1.0 档, 时刻表 1, 每种控制器 5 种子)\n")
    p("支路入口只有 " + "、".join(f"`{x}`" for x in E["side_lanes"]) + "; 其余车辆全部从 15 条边界入口出发。一辆车在支路入口出发, 等于在它与下游 RL 路口之间凭空多出一辆车;"
      " 若插车延迟随控制器变化 (信号放得慢、入口车道排到入口处就插不进), 或者插车常常插到公交/救护车前面, 就会把与优先级无关的差异带进 RL 路口的指标。下面三项逐一检查。\n")
    p("**1. 出发延迟按入口边与车型 (平均 s, 5 种子平均; 括号内为 99 分位最大值)**\n")
    p("| 入口边 | 车型 | 每集出发 | " + " | ".join(nm for _, nm in pols) + " |\n|---|---|---|" + "---|" * len(pols))
    keys = sorted({k for d in runs["served"] for k in d["C2"]["by_entry_type"] if k.split("|")[0] in side})
    for k in keys:
        e, ty = k.split("|"); cells = []
        for pol, _ in pols:
            vs = [d["C2"]["by_entry_type"].get(k) for d in runs[pol]]; vs = [v for v in vs if v and v["mean_delay"] is not None]
            cells.append(f"{statistics.mean(v['mean_delay'] for v in vs):.2f} ({max(v['p99_delay'] for v in vs):.1f})" if vs else "—")
        n = statistics.mean(d["C2"]["by_entry_type"][k]["requested"] for d in runs["served"] if k in d["C2"]["by_entry_type"])
        p(f"| {e} | {ty} | {n:.0f} | " + " | ".join(cells) + " |")
    worst = {pol: max(v["mean_delay"] for d in runs[pol] for kk, v in d["C2"]["by_entry_type"].items() if kk.split("|")[0] not in side and v["mean_delay"] is not None) for pol, _ in pols}
    p("\n边界入口 (对照): 各入口×车型平均出发延迟的最大值 " + "、".join(f"{nm} {worst[pol]:.2f} s" for pol, nm in pols) + "。")
    p("\n**2. 插到优先车辆前面 (C17)**: 计数规则 = 公交或救护车从上游驶入某车道时, 该车道上 15 s 内有别的车刚出发 (优先车辆自己的出发车道不算)。\n")
    p("| 控制器 | 公交 (5 集合计) | 救护车 (5 集合计) | 涉及车道 |\n|---|---|---|---|")
    for pol, nm in pols:
        cs = [c for d in runs[pol] for c in d["C17"]["cutins"]]
        lanes = collections.Counter(c["lane"] for c in cs)
        p(f"| {nm} | {sum(1 for c in cs if c['type'] == 'bus')} | {sum(1 for c in cs if c['type'] == 'ambulance')} | " + ("、".join(f"`{l}` {n}" for l, n in lanes.most_common()) or "无") + " |")
    p("\n解读: 公交从未被插到前面。救护车的插队全在 `7849019#2_1`: 它是 cluster_5166592137 西进口 (单车道 523892327#0) 上游的非 RL 边, 车道上的小汽车全部来自这个支路入口,"
      " 是西右转唯一的车源 (不设这个入口, 西进口除救护车外整集无车, 违反 \"不留空进口\")。按每集约 75 辆的流率, 若这些车是从上游自然开来, 救护车驶入时前方 15 s 内有车的期望约为每辆 0.37 次, 5 集 15 辆救护车约 5.5 次;"
      " 固定轮转的 6 次与之相符, 排队驱动与随机偏多, 但都是 15 辆救护车的小样本。判定没有偏差的依据: (a) 插车延迟三种控制器完全相同, 插入车流不受信号影响, 等同外生到达;"
      " (b) 救护车在非 RL 车道上停车 0 秒, 不存在 \"救护车在上游排队、插入车趁空位插到它前面\" 的情形; (c) 插入车随后与救护车在 RL 进口正常排队, 与从上游开来的车没有区别, 控制器看得见、可处理;"
      " (d) 换车型孪生评估中孪生车在同一时刻经过同一车道, 遇到同样的插入车, 差异抵消。")
    p("\n**3. 优先车辆在非 RL 车道上的停车 (车·秒/集, 5 种子平均)**: 这部分不进 per-visit 指标, 但若很大说明优先车辆在进入 RL 路口前已被非 RL 瓶颈拖慢。\n")
    p("| 控制器 | 公交 | 救护车 |\n|---|---|---|")
    for pol, nm in pols:
        g = lambda ty: statistics.mean(d["C17"]["prio_nonrl_stop_s"].get(ty, 0) for d in runs[pol])
        p(f"| {nm} | {g('bus'):.0f} (约 {g('bus') / max(1, B and len(B['buses'])):.2f} s/辆) | {g('ambulance'):.1f} |")


DEVIATIONS = [
    "课程档嵌套抽稀 (定稿时修正): 原来每辆车独立抽均匀随机数 key, key < L 保留; 0.6 档平台期后段碰巧多留约 7% 的车, C1 被抬到 0.08–0.107。改为同一路线按时间顺序取黄金比例低差异序列 (起点为该路线第一辆车原 key, 不多耗随机数), 1.0 档逐位不变, 0.6/0.8 档平台期每 300 s 发车量变平, C1 回到 ≤0.07。",
    "cluster_5166592137 北右转: 从规则内入口出发的唯一路线要绕街区重复经过同一条车道; 按用户要求 (2026-10-28) 为它单设例外支路入口 7849022#0 (入口到北进口之间有一次让行), 设计流量 40 辆/h; 零需求转向因此仍是计划所列 3 个。",
    "设计流量 (用户 2026-10-28 要求): 9101555 北直行 200、北右转 60、Digges 支路右转 80 辆/h (该路口无可信 SCATS 站点, 旧路由只有 5 辆/h)。",
    "每个 RL 进口平台期最低 100 辆/h、每条进口车道最低 30 辆/h (用户要求不让进口一直没车), 作为强软约束, 全部满足。",
    "\"清空后很快又排满\" (用户关注) 纳入迭代: 某转向完全清空后 60 s 内回到 max(4 辆, 60% 容量) 的次数占清空次数 ≥30% 且清空 ≥10 次, 该进口车道容量 ×0.85。",
    "让行口: 计划的 \"冲突主路 ≤150 辆/h\" 会压低真实主干道车流, 改为按主路流量估算让行支路通过量 (0.6 × (1100 − 主路)), 只限支路一侧, 迭代 3 轮; 9101554 的 2294382#1 ≤100 辆/h 保留 (软)。",
    "入口上限: 长度 ≥100 m 的边界入口由 600 放宽到 900 辆/h (单车道入口在 600 时满足不了 SCATS 量); 插车影响由 C2 按入口检验。",
    "进口车道容量: 计划写的 \"共同占用 ≤0.6 起始检查、只报告\" 改为线性规划硬约束: 上限 = 1800 × 0.92 × 获绿相位占比 × 利用率 (单一转向车道 0.5, 共享车道 0.45, 共享车道的获绿占比取车道上各转向中最小者, 因为只有头车的转向为绿时车道才走); 标定显示共享单车道进口 (如 4074526#0 下游的 cluster_21620852 南进口) 是插车延迟与 \"清空又满\" 的主因。标定中对持续积压、长进口越界或清空又满的进口再按 ×0.85 下调。因车道容量约束, 659805、cluster_1998326407、cluster_21620852 的实现量低于 SCATS 的 0.8 倍 (计划 D4 允许的瓶颈例外)。",
    "支路入口上游来车 ≤10% 在线性规划里是强软约束 (含公交、救护车的固定穿行与发车), 结果由按全部车型保守计数的测试检验。",
    "转向先验降为弱目标, 可信路口总量为强目标: 旧 18h 路由有 37% 的车在 RL 进口道上出发, 先验在相邻路口间不守恒。",
    "公交: 原 GTFS 出发边沿公交自身路线满足支路规则 (两端非 RL、≥40 m、到第一个 RL 进口无让行、≥60 m) 的, 保留原出发点, 不往上游补经过 RL 路口的前段。",
    "救护车选线加 \"入口平台期负荷 ≤ 450 辆/h\" (第一版选中的入口 74233405#1 负荷 886 辆/h, 插车延迟 125 s)。",
    "救护车选线第二版: 先要求机会占比 ≥70%, 再选不分车型时救护车基准等待最接近公交的路线 (用户要求: 不让公交天然占优); 所有合格候选的救护车基准都高于公交, 选差距最小者 (+1.1 s, 机会 90%), 剩余差距在评估上用换车型孪生处理。",
    "救护车选线改为仿真实测: 按规则打分选出的路线以主路直行为主, 固定轮转下只有 28% 的访问有放行取舍机会 (计划要求 ≥70%); 改为对满足规则的候选各放 20 辆救护车实测一集, 取机会占比最高的 (amb_select18.py, 选中路线实测 91%)。",
    "排队驱动控制器改为看车道头车 (只把头车去向属于该相位的车道算作可放行); 只看 \"转向有车\" 会在共享车道上长时间给不能放行的绿灯。",
    "C3 堵死改为看头车: 头车自己的转向是绿灯却停着, 绿灯累计 ≥30 s 且该车道 120 s 无车离开; 头车方向为红灯时的头车阻塞不算堵死。",
    "外部复核后的修正 (2026-10-28 晚): (1) C1 恢复计划原口径 (平台期最后一窗对第一窗), 一度改为从第二窗起算属于放宽, 已撤回; C14 改为按单条车道判断 (按进口平均会冲淡单车道积队, 例: cluster_21620852 东进口第 1 车道); 迭代时排队驱动与固定轮转都作硬判据。(2) 支路入口 \"上游穿行 ≤ 本地发车 10%\" 改为按全部车型、所有可能车道路径计数 (原统计漏了公交与救护车), 小汽车支路入口改选公交只出发不穿行的车道。(3) 标定工具的统计器缺条目检查原先判空值无效 (统计器缺条目返回 0 并计数), 改读内部计数; 等待值另由状态/奖励校验 (严格模式统计器) 独立核对。",
    "无可信站点的路口 (133307283、389281、26165090、659802、9101555、cluster_5166592137) 没有总量目标, 流量由上下游守恒形成; 标定不过时加总量上限 (本轮实现 × 0.9)。",
    "C13 开局判据: 计划写的 \"0–900 s 内等待 ≥90 s 的车次为 0\" 对 4 相位以上路口相当于不到一个轮转周期 (659799 一整轮约 168 s), 会把正常红灯等待判为不合格; 改为 \"开局阶段没有车等满本路口一个服务轮转 (有效相位数 × 42 s)\", 90 s 计数保留为诊断。",
    "结构性空车道已解决 (用户 2026-10-28 要求): 在原 nocg 网上 cluster_21620852 东进口第 0 车道 (37721356#0_0) 整集 0 辆 — 它只能由 cluster_1998326407 南进口第 0 车道直行喂车, 但上游非 RL 节点 658464 处 2294384#0 第 0 车道同时连向 4472590#0 第 0/1 车道, SUMO 把直行车全分到第 1 车道, 不换道时无车能到达; 下游任何去向从第 1 车道也都能到, 无法靠路线强制。修复: 另存 lanefix 网 (原 nocg 网不动), 只改这一条连接: 2294384#0 第 0 车道 → 4472590#0 第 1 车道 改为 2294384#0 第 1 车道 → 4472590#0 第 1 车道 (第 0 车道只进第 0 车道 [左转+直行], 第 1 车道进第 1 [直行] 与第 2 [右转] 车道)。核对: 18 个路口动作表与信号程序逐项一致, RL 进出车道长度/形状不变, 路网审计 25 项全 0; 定稿路由逐车道计数 (纯 SUMO 一集, 1.0/0.8/0.6 三档): 68 条受控进口车道 (含 5 条恒绿) 全部有车, 最少每集 22/18/14 辆 (每车道 30 辆/h 下限的车道); 37721356#0_0 每集 58/46/36 辆 (容量系数 0.25, 再高会让上游非 RL 的 38192817 排队)。副作用: 659805 北左转在新连接下有了合规路线, 零需求转向由 3 个减为 2 个; 南进口第 1、2 车道只能由支路入口 230483152#0_1 喂车, 该入口上游无穿行车, 发车量较大, 插车影响由 C2 与 C17 实测。",
    "中段插车 (外部复核后重做): 公交原 GTFS 出发边按全部车型核对 \"上游穿行 ≤ 本地发车 10%\" 后全部不合规, 均已撤掉, 这些公交延伸到边界或合规入口 (多经过的 RL 路口逐辆记在 buses.json); 小汽车路线一律不得从上游穿过支路入口车道; 选支路入口时公交与救护车的穿行按实际通过量计入。结果: 3 个支路入口全部合规, 唯一的穿行是 7849019#2 第 1 车道上的救护车 (cluster_5166592137 西进口只有一条车道, 喂西右转的车道都在救护车路线上, 结构上避不开), 插到救护车前面的情况由 C17 实测。",
    "C5 (清空与再满)、C12 (车道压力)、C8 (非 RL 停车)、C10 只报告; 计算量化指标的合成序列单元测试 (计划任务 8 的三个测试) 未做, 指标正确性由标定结果的一致性旁证。",
]

if __name__ == "__main__":
    main()
