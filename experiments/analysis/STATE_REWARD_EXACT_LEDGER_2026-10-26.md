# 状态与奖励修正：执行台账（2026-10-26）

实施计划 experiments/analysis/STATE_REWARD_EXACT_PLAN_2026-10-26.md 的执行记录：每个任务的完成行（提交范围与测试命令）、执行中的决定（Ruling）、外部审查的修正与延期项。原件为计划工作区的 progress.md，工作区已清理，此处为入库副本。

```
# SDD ledger — plan: experiments/analysis/STATE_REWARD_EXACT_PLAN_2026-10-26.md
Setup: Ruling: 不建 worktree, 在本工作区的分支 fix/state-reward-exact 上实施 — 无训练/调度进程在跑, 用户的实验路径固定为本目录; 代价: 若有人在此期间从本目录起训练会捡到半成品代码
Pre-flight: 任务1 产出 obs_kwargs_from_cfg/env_kwargs_from_cfg/reward_wait_basis/turnmap_from_meta → 任务2/3/5/6 的配置键 obs_all_connections/obs_slot_memo/obs_awt_basis=visit/reward_wait_basis 由它映射, 名称一致; 任务4 产出 VisitWaitTracker.get(ts_id, vid, lane) → 任务5/6 消费, 签名一致; 任务7 的 all_connections 环境参数与任务1 env_kwargs_from_cfg 的键一致; 无冲突
Task 0: Ruling: 快照 npz 不入库, 放在本计划工作区 stream_baseline/ (git 忽略), 工具入库 — 三份快照约 140 KB 但属一次性校验用途; 代价: 工作区被清后需用相同命令重建 (确定性已证: 重跑逐位相同)
Task 0: 发现: ckpt_env.build 不指定 ckpt 时取最后一个检查点 (457 为 ckpt_ep00800), 不是 best.pth; 本会话早先的 ψ 置零闭环、reward 重算等诊断用的都是 ep800 而非 best(ep505), 结论方向不受影响, 最终报告须更正措辞
Task 0: complete (commits e9adaf5..6e6dd6e, tests: python3 experiments/tools/dublin/obs_reward_dump.py --compare /home/xiaowen/sumo-rl/.superpowers/sdd/STATE_REWARD_EXACT_PLAN_2026-10-26/stream_baseline/exp457_s123.npz /home/xiaowen/sumo-rl/.superpowers/sdd/STATE_REWARD_EXACT_PLAN_2026-10-26/stream_baseline/exp457_s123_rerun.npz → compare exp457_s123.npz vs exp457_s123_rerun.npz: 键 39, 最大绝对差 0.000e+00, 有差的键 0)
Task 1: Ruling: frap_glue.py 第 33 行的 enum meta turnmap 不改用 turnmap_from_meta — GS 路径本次不动, 表达式相同; 代价: enum meta 若出现同索引标签不一致不会被该路径报错 (任务 9 审计会查)
Task 1: Ruling: ckpt_env 原来在无优先级键时显式传 priority_source=None, 共用模块改为不传 — 观测类默认即 None, 全部在用配置都有 priority_source, 流快照逐位相同; 代价: 无
Task 1: complete (commits 6e6dd6e..a40a052, tests: python3 -m pytest experiments/tests/test_exp_params.py experiments/tests/test_warmstart_obs_split.py -q → 15 passed, 13 warnings in 1.68s)
Task 2: complete (commits a40a052..db9ffdb, tests: python3 -m pytest experiments/tests/test_obs_all_connections.py -q → 5 passed in 0.40s)
Task 3: complete (commits db9ffdb..57465fb, tests: python3 -m pytest experiments/tests/test_obs_slot_memo.py -q → 3 passed in 0.55s)
Task 4: Ruling: 计划停车 (busStop) 的处理跟 SUMO 走 — SUMO 在刹停进站那一秒记 1 秒等待, 停靠期间不再累计; 统计器只读 getWaitingTime>0, 与 SUMO 累计等待逐秒相等; 测试改为断言停靠期间不增长且等于 SUMO 计数; 代价: 若论文把进站刹停那 1 秒视为非等待, 与 SUMO 自身口径不一致, 量级每次停靠 1 秒
Task 4: complete (commits 57465fb..6b3849d, tests: python3 -m pytest experiments/tests/test_visit_wait.py -q → 8 passed in 0.98s)
Task 5: complete (commits 6b3849d..a2582ad, tests: python3 -m pytest experiments/tests/test_obs_awt_visit.py -q → 5 passed in 0.98s)
Task 6: Ruling: ckpt_env 在 legacy 口径下沿用原做法 (不看 cfg.reward_fn, 一律建优先级奖励; 评估工具不用奖励值), 只在 visit 口径下按 reward_fn 选普通/优先级 — 保证旧评估路径逐位不变; 代价: legacy 下 458 的 ckpt_env 奖励仍是优先级奖励 (与训练不同), 仅影响 obs_reward_dump 这类读奖励的诊断
Task 6: Ruling: agent_arch=multihead 与 reward_wait_basis=visit 组合直接 sys.exit — 向量奖励胶水 (multihead_glue) 未接统计器, 本批不改; 代价: 多头臂暂不能用精确口径
Task 6: complete (commits a2582ad..db65500, tests: python3 -m pytest experiments/tests/test_reward_wait_visit.py -q → 6 passed in 1.14s)
Task 7: complete (commits db65500..c59ca3a, tests: python3 -m pytest experiments/tests/test_out_lanes_all_connections.py -q → 3 passed in 0.45s)
Task 8: Ruling: reindex_tls 改为返回 (phases, meta, moved) 三元组, experiments/tools/dcc/reindex_8std_net.py 的调用同步改成三元组解包 — 该工具是另一条 (DCC 真实信号) 管线, 行为不变; 代价: 若有未入库脚本仍按二元组解包会报错
Task 8: complete (commits c59ca3a..cc0242a, tests: python3 -m pytest experiments/tests/test_reindex_split_slots.py -q → 3 passed in 0.21s)
Task 9: Ruling: ckpt_env.build 加 cfg_override 参数 (只在内存里覆盖配置键, 不写文件) 供调试/双算工具换网换口径 — 默认 None 行为不变; 代价: 无
Task 9: 发现: 重算工具复现旧缓存时最初用全局字典 (观测类是每路口一个实例), 改为 (路口, 车辆) 后旧口径差异归零; 用旧策略跑分槽网 cluster_21620852 观测 3193→11053, 证实分槽网必须重训
Task 9: complete (commits cc0242a..e747233, tests: python3 experiments/tools/dublin/state_reward_debug.py 457 --basis visit --all-connections --no-memo --strict --steps 60 →   统计: {'first_rl_sighting': 379, 'first_rl_sighting_with_upstream_wait': 39, 'veh_on_shared_lane': 1678, 'veh_on_second_lane_of_multi_index': 62, 'dropped_slot_minus1': 3} | (车辆,路口) 重访数 0)
Task 10: Ruling: 双算参照的 since_green 起点取 reset 时刻并把 reset 时放行的槽记为刚放行 (与观测类首次观测的记账一致) — 修正后 315/313 since_green 差为 0; 代价: 无
Task 10: complete (commits e747233..52b4b66, tests: python3 -m pytest experiments/tests -q → 157 passed, 13 warnings in 6.57s)
Final review: 外部审查 (fable 子代理, 读 e9adaf5..52b4b66): Critical 0, Important 2, Minor 7; 审查者独立复现了测试/regress/流快照, 并在都柏林双车道进口上独立对账统计器 (158972 车辆秒差 0)
Final: fixed Important 1 (visit 配不支持的奖励名静默走 legacy) — test_reward_wait_basis_visit_rejects_unsupported_reward_fn RED→GREEN, suite 159/159
Final: fixed Important 2 (eval_unified 修正面被低估, "逐位不变" 未划界) — 验证报告 §2 改范围 + 新增 §2.1/§2.2, 测试名改为 includes_keys_eval_unified_used_to_drop; 提交 a40a052 的说明不可改, 以报告为准
Final: fixed Minor 4 提级为 Important (训练不检查 tracker.missing, 静默零值风险) — test_assert_tracker_clean RED→GREEN, train.py 每集结束调用 (try 之外), suite 159/159
Final: minor (deferred): 同一车道受两个信号控制时 _lane_key 静默覆盖 (现有三张网无此情况; 有了 missing 检查会响亮暴露)
Final: minor (deferred): 几何标签一致性检查在构造期触发, 换 "直行+右转共用索引" 的网会在 rebind 前失败
Final: minor (deferred): reindex_8std.py 手写 argv 解析脆弱 (--tag 后无值 IndexError)
Final: minor (deferred): 测试缺口 lane_copy+all_connections 组合无显式断言; eval_unified 与 train.py 无真实比流
Final: minor (deferred): env_extra_kwargs 按真值过滤, 将来默认值翻转时显式 False 会被丢掉 (现无影响)
Final: Ruling: 审查者搁置项 (multihead 不接 visit / ckpt_env legacy 奖励 / frap_glue turnmap / 下游 mean_awt / 约 30 个自建环境的旧工具未接 wait_tracker 会 RuntimeError / busStop 口径 / 重训范围 / 生成网 XML 未逐行审) — 全部维持原判: 前四项见任务 Ruling; 旧工具在 visit 配置上响亮失败是可接受的 (不会静默错), 需要时各工具自行接 env_extra_kwargs; 代价: 旧工具跑 visit 配置前须先接线
```

## 复查补修（2026-10-26 深夜）

- 第二轮外部审查两点属实并已修：独立评估入口补 assert_tracker_clean；重算工具 visit 参照改为自带计数器 + 与 SUMO 累计等待恒等式核对，退出判定纳入缺条目。
- 全仓库清点后补修：DQN sync_target_init（默认关）；train_strict_errors（默认关，只吞 SUMO/TraCI 异常）；info["step"] 每集清零、推进后写；regress_dryrun / fixed_cycle_eval 改用 exp_params。
- Ruling: gui_eval / edge_waiting_collect / net_waiting_all / traincoeff / trainorico / train_ppo 不改 — arm 专用或其它训练器，visit 配置在其上会响亮报错；代价: 要在它们上用新口径须先接 exp_params。
- Ruling: 漏掉独立评估入口检查与参照不独立两处的原因是只核了自己改动的路径；补修改为按清单逐处核对。
