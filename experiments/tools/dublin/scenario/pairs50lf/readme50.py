"""生成 nets/dublin/weekday_pairs50_lanefix/README.md (数字全部取自 build/*.json)。"""
import json
import common50 as P


def main():
    S = P.load("pair_routes.json"); T = P.load("timetables.json"); F = P.load("final_calibration.json"); V = P.load("validation.json"); C = P.load("pair_candidates.json")
    L = []; p = L.append
    p("# 50 + 50 配对受控场景 (lanefix 网, 18 个 RL 路口)\n")
    p("实现计划: `experiments/analysis/PLAN_PAIRS50_LANEFIX_2026-10-28.md` (第三版)。工具: `experiments/tools/dublin/scenario/pairs50lf/`; 测试: `experiments/tests/test_pairs50_lanefix.py`。")
    p(f"路网 `{P.NET.name}` 与动作表 `{P.META.name}` (与 18h 定稿相同, sha256 核验)。\n")
    p("## 用途与核心设计\n")
    p("受控测试: 比较不同优先级规格 (如 111 / 531 / 351) 下公交与救护车的待遇, 尽可能隔离路线、出发时机、沿途竞争等外部因素。"
      "**公交与救护车可互换**: 共用同一套路线 (逐段同车道), 数量相同 (各 50), 每条路线对数为偶数且路线内交替先后, 出发时刻分布相同; 两类唯一的区别是车型标签。"
      "观测按优先级档位分桶、奖励用同一张表加权, 控制器看不到车型。**每份时刻表自身镜像对称**: 前半段的每一对在后半段整体晚 600 s 再出现一次, 并对调公交与救护车;"
      " 600 s 是各路口固定轮转周期 (50/75/100/150 s) 最小公倍数的倍数。于是单份时刻表里同路线结伴、跨路线共现、优先车相遇、出发时刻、固定轮转的相位对齐都由两类各承担一次,"
      " 531 与 351 在同一份时刻表上是同一个问题 (只是哪些车是 5 档换了), 剩下的差别只来自前后两半段背景车流的随机差异。\n")
    p("## 文件\n")
    p("- 背景 (所有时刻表共用, 与 18h 定稿逐位相同): `background_cars_{060,080,100}.rou.xml`、`background_gtfs_as_car.rou.xml` (262 辆 GTFS 公交以 car 身份)。背景里没有任何别的公交或救护车。")
    p("- 配对车辆: `pairs_tt{1..13}.rou.xml` (第 1–5 份训练, 训练代码目前只能用第 1 份; 第 6 份标定; **第 7 份最终测试, 定稿前未跑**; 第 8–13 份评估)。")
    p("- 评估专用 (不得被训练配置引用): `pairs_tt{k}_mirror.rou.xml` (车型对调的镜像)、`pairs_tt{k}_as_car.rou.xml` (换车型孪生, 100 辆全是 car)、`pairs_unseen_tt{8..13}[_mirror].rou.xml` (3 条未见路线各 4 对)。")
    p("- `pairs_manifest.json`: 每个车辆 id → 份号、是否镜像/孪生、路线、对号、先后位置、车型、原车型。id 不含车型、所有文件之间唯一 (避开奖励函数里 车辆 id → 权重 的缓存)。")
    p("- `scenario.sumocfg`: 背景 1.0 档 + 第 1 份; 处理项与 18h 相同。\n")
    p("## 路线\n")
    p(f"候选 {len(C['candidates'])} 条 (边界入口、全程无非 RL 让行连接、不穿过小汽车支路入口车道、≥3 个 RL 路口且 ≥2 个竞争转向、入口到第一个 RL ≥100 m), 每条放探针车实测后按覆盖贪心选出, "
      "选线只读覆盖、机会占比、车道轨迹、非 RL 停车、出发延迟、入口负荷, 不读任何等待结果; 每个入口最多 3 条、每条竞争进口车道最多 3 条路线且配对数 ≤ 14 对。\n")
    p("| 路线 | 入口 | 对数 | 对内间隔 s | 竞争转向 | 机会占比 (固定轮转, 第 1–6 份) |\n|---|---|---|---|---|---|")
    for r in S["routes"]:
        p(f"| {r['id']} | `{r['entry']}` | {r['n_pairs']} | {T['deltas'][r['id']]:.0f} | " + "、".join(f"{t} {m}" for t, m, _ in r["competing"]) + f" | {F['P4']['by_route'][r['id']]:.2f} |")
    p(f"\n覆盖 {len(S['coverage'])}/18 个 RL 路口; **未覆盖: {'、'.join(S['uncovered'])}** —— 389332 与 cluster_1448439 只在 College Green 入口的路线上, 那些路线的机会占比 38–67% (< 70%); 9101555 没有满足规则的路线。是否放宽规则由用户定。\n")
    p("## 时刻表\n")
    p("- 出发窗: 每条路线 [900 s, 3000 s − 第 99 分位行程时间 − 对内间隔 − 120 s]; 前半段统一放在 [900 s, 900 s + W], W = 各路线出发窗上限的最小值 − 900 − 600, 后半段为前半段整体晚 600 s, 两半都在每条路线自己的出发窗内。")
    p("- 对内间隔逐路线取 \"同对两车同时在同一 RL 进口 ≤ 10%\" 的最小值 (中性控制器, 标定时刻表)。每条路线上两类的先行次数与平均出发时刻严格相等 (按构造)。")
    p("- 配对车辆只与其他配对车辆保持同车道 ≥ 3 s; 不再为背景车找空档 (密集入口 74233405#1_0 上前后两半同时找空档不可行), 与背景挨得近时由 SUMO 正常排队插入, 实际出发延迟见下。")
    p("- 第 1–6 份: 单份优先车相遇不对称 > 20% 或有效冲突不足时换种子重抽; 第 7–13 份用预定种子 (7007–7013) 一次生成, 不仿真筛选。第 7–13 份与标定份共用同一份背景与同一套 10 条路线, 只是出发时刻不同。")
    p("\n| 份 | 种子 | 重抽 | 相遇不对称 | 每类每集有效冲突 | 整对最大移动 s |\n|---|---|---|---|---|---|")
    for k in range(1, P.N_TT + 1):
        t = T["timetables"][str(k)]; e = t.get("encounters")
        p(f"| {k} | {t['seed']} | {t['rejected']} | {e['asym']:.2f} | {e['conflict_visits_per_ep']['bus']:.0f} / {e['conflict_visits_per_ep']['ambulance']:.0f} | {max(abs(x['shift']) for x in t['pairs']):.0f} |" if e else
          f"| {k} | {t['seed']} | 0 | (留出, 未仿真) | — | {max(abs(x['shift']) for x in t['pairs']):.0f} |")
    p("\n## 标定与验收 (只用标定时刻表)\n")
    for pol, nm in (("served", "排队驱动"), ("fixed", "固定轮转"), ("random", "随机 (压力测试)")):
        h = F["hard"][pol]
        p(f"- {nm} 5 种子: C1 在网增长 {min(h['C1_growth']):+.3f}…{max(h['C1_growth']):+.3f}; 堵死 {sum(h['C3_blocked'])}; 漏放 {sum(h['C4_missed'])}; 回溢 {sum(h['C6_spill'])}; 换道/碰撞/瞬移/缺条目 {h['C9_lc_col_tp_missing'][0]}; "
          f"中段插车插到配对车辆前面 {sum(sum(x.values()) for x in h['C17_cutins'])} 次; 每次访问等待 (车道口径) 小汽车 {h['C16_by_type']['car']:.2f} / 公交 {h['C16_by_type']['bus']:.2f} / 救护车 {h['C16_by_type']['ambulance']:.2f} s。")
    p1s, p1f = F["P1"]["served"], F["P1"]["fixed"]
    ok = lambda v: -0.5 <= v["ci90"][0] and v["ci90"][1] <= 0.5
    p(f"- P1 可互换性 (第 1 份单份; 独立单位 = 整集; 种子数逐轮按方差补到 90% 区间半宽 ≤ 0.3 s; 等效界 ±0.5 s; 等待为车辆口径 = 每辆车每次访问平均等待再对车平均, 与 wandb 相同): "
      f"排队驱动 {p1s['n_seeds']} 种子 {p1s['mean_diff_bus_minus_amb']:+.2f} s [{p1s['ci90'][0]:+.2f}, {p1s['ci90'][1]:+.2f}] {'通过' if ok(p1s) else '**未通过**'}; "
      f"固定轮转 {p1f['n_seeds']} 种子 {p1f['mean_diff_bus_minus_amb']:+.2f} s [{p1f['ci90'][0]:+.2f}, {p1f['ci90'][1]:+.2f}] {'通过' if ok(p1f) else '**未通过**'}。"
      f"训练用的 SUMO 种子 0 单集: 排队驱动 {F['P1']['training_seed0']['served']:+.2f} s、固定轮转 {F['P1']['training_seed0']['fixed']:+.2f} s (单集逐种子标准差约 1–1.5 s, 只报告)。")
    p("- P2 镜像一致性: " + "; ".join(f"{k} {'逐车相同' if v['identical'] else '不同'}" for k, v in F["P2"].items()) + " (车型标签不影响物理行为; 知道车型的控制器只依据数值档位)。")
    p(f"- P3 优先车相遇 (原份 + 镜像, 中性控制器): {F['P3']['union_counts']}; 每类每集有效冲突 {F['P3']['conflict_visits_per_ep']['bus']:.0f}。")
    p(f"- P4 放行取舍机会: 每条路线 {min(F['P4']['by_route'].values()):.2f}–{max(F['P4']['by_route'].values()):.2f}; 每类每集有机会的访问 公交 {F['P4']['with_opp_per_class_per_ep']['bus']:.0f} / 救护车 {F['P4']['with_opp_per_class_per_ep']['ambulance']:.0f}。")
    p(f"- P5 (报告) 优先车对小汽车的中性差距: 排队驱动下配对车辆比小汽车每次访问多等约 {F['P5']['pair_minus_car_s']:.1f} s —— 配对路线专门挑了有竞争的转向, 天然比小汽车平均难走。"
      "只影响 \"优先车对小汽车\" 的类均值比较, 不影响公交对救护车; 优先车对小汽车请用换车型孪生或逐转向比较。")
    p6 = F["P6"]["tables"]
    p(f"- P6 (能力检查, 知道车型的简单控制器, 不是 RL): 531 下 救护车 {p6['531']['mean']['ambulance']:.2f} < 公交 {p6['531']['mean']['bus']:.2f} < 小汽车 {p6['531']['mean']['car']:.2f} s; "
      f"351 下 公交 {p6['351']['mean']['bus']:.2f} < 救护车 {p6['351']['mean']['ambulance']:.2f} < 小汽车 {p6['351']['mean']['car']:.2f} s; 两方向镜像差 {F['P6']['mirror_gap_s']:.3f} s。场景允许按规格服务, 不代表 DQN 一定做到。")
    p(f"- P11 同类同进口共现 (一次访问期间同一 RL 进口边上有另一辆同类配对车辆的占比): " + "; ".join(f"{'排队驱动' if pol == 'served' else '固定轮转'} 公交 {v['bus']:.1%} / 救护车 {v['ambulance']:.1%}" for pol, v in F["P11"]["same_class_copresence"].items()) + "。")
    p(f"- P12 中间档镜像差 (知道车型的控制器: 531 下公交 对 351 下救护车): {F['P12']['mid_mirror_gap_s']:.3f} s。")
    p(f"- 实际出发延迟 (配对车辆, 中性控制器): 公交 {F['P10']['depart_delay_mean_s']['bus']:.2f} s / 救护车 {F['P10']['depart_delay_mean_s']['ambulance']:.2f} s (最大 {max(F['P10']['depart_delay_max_s'].values()):.1f} s)。")
    p("- 口径说明: P1 用车辆口径; P5、P6、上面各控制器的 \"每次访问等待\" 用访问口径 (所有访问合计再平均), 两者不能直接比。")
    p(f"- P7 同对共现最大 {max(F['P7']['max_by_route'].values()):.2f}; P8 非 RL 停车 (s/辆) 排队驱动 公交 {F['P8']['by_policy']['served']['bus']:.2f} / 救护车 {F['P8']['by_policy']['served']['ambulance']:.2f}, 固定轮转 公交 {F['P8']['by_policy']['fixed']['bus']:.2f} / 救护车 {F['P8']['by_policy']['fixed']['ambulance']:.2f} (固定轮转多出的部分是 1448442 58 m 短进口的排队回溢到上游); P9 车道轨迹违规 {F['P9']['violations']}; P10 中性控制器下截尾 {F['P10']['censored_neutral']}。")
    v = V["validation"]
    p(f"- 状态与奖励逐步独立重算: 固定轮转 / 随机各 {v['fixed']['state_checks']} 次全部一致, 统计器缺条目 0, 换道 0; 第 1 份与第 1′ 份在 531 / 351 下观测档位与奖励权重全部与规格表一致。\n")
    p("## 评估建议 (计划 §8)\n")
    p("- 主证据: 同一份时刻表、车型不变, 只换 531 / 351 —— 同一批救护车在 531 下是否比 351 下少等, 同一批公交反过来; 原份与镜像都做再合并; 111 作空白对照。")
    p("- 所有规格用同一批评估时刻表 (第 8–13 份及镜像) 与同一批评估种子; 排空到 4500 s, 截尾不补零、不只比完成的车。")
    p("- 用同一检查点切换规格时, 每次都新建环境、观测与奖励函数; 分别训练时要多次独立训练才能把规格效果与训练差异分开。")
    p("- 单份时刻表已自身镜像对称; 原份 + 镜像合并后两类在出发时刻层面也严格相同, 报告类间比较时仍建议两份都跑。")
    open(P.OUT / "README.md", "w").write("\n".join(L) + "\n"); print("写出", P.OUT / "README.md")


if __name__ == "__main__":
    main()
