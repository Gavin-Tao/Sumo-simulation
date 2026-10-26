# 状态与奖励修正：验证报告（2026-10-26）

实施计划：experiments/analysis/STATE_REWARD_EXACT_PLAN_2026-10-26.md。分支 fix/state-reward-exact。
核对报告：experiments/analysis/STATE_REWARD_DEBUG_2026-10-26.md。

## 1. 改了什么（全部有开关，默认 = 旧行为）

| 项 | 配置键 | 默认 | 位置 |
|---|---|---|---|
| 建表遍历每个信号索引下全部连接 | obs_all_connections | false | observations.py PriorityMovement |
| 运动缓存可关闭，每步按当前路线重查 | obs_slot_memo | true | 同上 |
| 等待口径 visit：逐秒访问统计器 | obs_awt_basis: visit | local/global 不变 | visit_wait.py + env.py + observations.py |
| 奖励等待口径 visit | reward_wait_basis | legacy | rewards.py |
| TrafficSignal.out_lanes 收齐全部连接 | all_connections | false | traffic_signal.py + env.py |
| cluster_21620852 分槽版 8STD 网 | 新文件 dublin_8std_splitslot.* | 旧文件不动 | reindex_8std.py --split-slots |
| 三个入口共用参数构造 | 无 | 行为不变 | exp_params.py |

## 2. 旧配置逐位不变的证据

1. 流快照：exp457（都柏林 18h）、exp274（1x1）、exp338（1x3）各 200 个决策步的全部 state 向量与 reward，在任务 0 记录基线，
   任务 1 到 10 每完成一项重跑一次比较，最大绝对差始终为 0。
2. train.py 回归摘要：regress_dryrun 对 exp208（都柏林 11h）与 exp136（1x3）两配置的摘要与 2026-07-02 基线一致。
3. 测试：新增 8 个测试文件 39 项，原有测试全部通过（全套 157 项）。

## 3. 新口径与定义一致的证据（任务 9）

exp457 种子 123 整集，旧/新口径 × 8std/8std_splitslot 四种组合，环境输出的每一组特征和奖励与独立重算的最大差 ≤ 5.2e-8（float32 舍入），
新口径下统计器缺条目 0。静态审计问题项：12（8std 首连接）→ 2（全连接，只剩 cluster_21620852 槽内灯色）→ 0（分槽网 + 全连接）。

## 4. 同一轨迹双算：旧口径驱动，新口径重算（任务 10）

| 实验 | 场景 | state 有差维占比 | state 最大差 | 奖励 新/旧 | 单步奖励最大差 |
|---|---|---|---|---|---|
| 457 | 都柏林 18h 改道第二版 | ≤ 0.15%（lane_occ 0.15%，下游 count 0.12%，进口 count 0.06%，mean_awt 0.02%） | lane_occ 1.32，进口 count 0.66 | 0.198 | 0.86 |
| 315 | 都柏林 11h | ≤ 0.12% | 0.66 | 0.500 | 6.76 |
| 313 | 都柏林 02h | ≤ 0.07% | 0.66 | 0.718 | 0.11 |
| 274 | 1x1 | 0 | 0 | 1.000 | 0 |
| 338 | 1x3 | 0 | 0 | 0.995 | 0.003 |

读法：
- 都柏林的 state 差异只落在 389332 的两个直行槽（容量减半、第二车道）和重访公交的槽，占比不到千分之二；is_green、槽内最短绿、since_green 完全相同。
- 都柏林的奖励变化很大，来源是旧口径把上游非 RL 路口的等待记给第一条 RL 车道：18h 的 cluster_1998326407 从 −158.4 变为 −2.8，
  11h 的 133307283 从 −133.9 变为 −7.5；其余路口变化 4% 到 94%。整集累计惩罚降到旧值的 20%（18h）、50%（11h）、72%（02h）。
- 网格：1x1 的 state 与奖励完全相同；1x3 的 state 相同，奖励差 0.5%（上一条 RL 车道最后一次采样之后的等待原来漏到下一条车道）。
  网格臂的训练输入不变、训练信号几乎不变。
- 313 的头部有 0.25% 的维差异大于 1e-6（最大 1.5e-6）：02h 绿灯很长，相位已持续时间 /100 超过 16 后 float32 的舍入即达 2e-6，属精度而非定义。

## 5. 重训范围建议

- 都柏林：采用新口径的所有臂重训。exp459（457 的新口径版）与 exp460（458 的新口径版）已建配置模板，未启动；
  449、456 的对应版本照 459 的改法建。GS 臂（301、341）用同一奖励函数，纳入比较时也要按新口径重训。
- 网格：state 不变；奖励 1x1 完全相同、1x3 差 0.5%。严格对比时 1x3 的臂按"奖励改了就重训"处理，1x1 可不重训。
- 归因：保留一个只开观测修复（obs_all_connections、obs_slot_memo=false、obs_awt_basis=visit）、奖励仍为 legacy 的臂作消融。

## 6. 口径说明（论文用）

- 等待 = SUMO 判定的等待步数（getWaitingTime > 0，速度低于 0.1 m/s 且非计划停车），按仿真步长累加；只计本次在本路口进口边上的秒数；
  路口内部车道与非 RL 路段不计给任何路口；离开再进入从 0 起；同一进口换道不清零；无时间窗。
- SUMO 在车辆刹停进站（busStop）的那一秒记 1 秒等待，停靠期间不再累计；统计器跟 SUMO 走。
- 奖励仍是"采样时刻在场车辆的平均（本次访问）等待"的水平量，聚合与权重公式不变。
- 本会话早先的诊断（ψ 置零闭环、奖励三口径重算、出口堵死统计等）用的是 457 的 ckpt_ep00800，不是 best.pth（ep505）；引用时注明。

## 7. 决定记录（rulings）

见本计划工作区 progress.md 的 Ruling 行；要点：不建 worktree 在分支上实施；流快照不入库；frap_glue 的 enum turnmap 不改；
计划停车跟 SUMO 口径；ckpt_env 在 legacy 下沿用原奖励构造；multihead 暂不支持 visit；reindex_tls 改三元组并同步 dcc 调用方；ckpt_env 加 cfg_override。
