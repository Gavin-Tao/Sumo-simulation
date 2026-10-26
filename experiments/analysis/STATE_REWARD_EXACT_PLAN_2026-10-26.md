# 状态与奖励修正实施计划（2026-10-26）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 修正 PriorityMovement 状态与 priority-avg-waiting / avg-waiting-time 奖励的六处缺陷，全部做成配置开关，旧配置逐位不变。

**架构：** 新增环境级逐秒访问等待统计器 VisitWaitTracker，状态与奖励在 `visit` 口径下只读它；观测建表改为可选遍历全部连接；运动缓存可选关闭；8STD 重索引工具加 split-slots 开关生成新 net/meta；三个入口共用一个参数构造模块。

**技术栈：** Python 3.9、libsumo、pytest（experiments/tests，需要 SUMO）。

**规格：** 本次会话定稿的设计（公式见本文第 0 节），对应评审清单第 1 到 6 项。

## 全局约束

- 每个新行为都有开关，默认值 = 旧行为；不写新键的配置逐位不变（用第 0 项的流快照校验）。
- 代码注释、提交说明、文档一律中文；提交说明末尾加 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。
- 不启动任何训练、wandb；新开的 Python 进程一律 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
- 改 train.py 后必须跑 `experiments/tools/frap/regress_dryrun.py` 与基线对账。
- 测试放 experiments/tests/，用 `python -m pytest experiments/tests/<file> -q` 运行。
- 新配置键：`obs_all_connections`、`obs_slot_memo`、`obs_awt_basis: visit`、`reward_wait_basis`、`all_connections`。

## 0. 公式

Δ = simulation.getDeltaT()，I_v(τ) = 1 当且仅当 getWaitingTime(v) > 0。

```
逐秒，键 (路口 j, 车辆 v, 进口边 e)：
    v 在 τ 出现在 j 的 ts.lanes 中属于边 e 的车道上：新访问 W = I·Δ，否则 W += I·Δ
    v 在 τ 不在 e 上：删除 W[j,v,e]
state：mean_awt_{m,p} = Σ_{v∈V_{m,p}} W[j,v,e(v)] / |V_{m,p}|，空集 0；feature = log(1+mean)/log(3601)
reward 456/457：r_j = −(1/100) Σ_p w_p Σ_{v∈V_p} W / |V_p|，空等级不计
reward 458：    r_j = −(1/100) Σ_{v∈V} W / |V|，空集 0
```

## 审查焦点

1. 步长不等于 1 秒：W 必须按 getDeltaT 累加（任务 4 测试）。
2. 车辆被 SUMO 传送（teleport）离开进口道再出现在别处：访问应关闭，不得残留（任务 4 测试）。
3. lane_copy 口径与 all_connections 同开：共享车道的容量只加一次（任务 2 测试）。
4. reward_scale / reward_floor 包装器与 visit 口径叠加仍成立（任务 6 测试）。
5. 评估入口与训练入口在同一仿真状态下输出逐位相同（任务 1 测试）。

---

### Task 0：分支与旧行为流快照

**文件：**
- 新建：`experiments/tools/dublin/obs_reward_dump.py`

**接口：**
- 产出：`python experiments/tools/dublin/obs_reward_dump.py <exp> --seed S --steps N --out F.npz`：用 ckpt_env.build 构造环境与策略，跑 N 个决策步，保存每步每路口 state 向量与 reward；`--compare A.npz B.npz` 逐位比较并打印最大差。

- [ ] 步骤 1：`git checkout -b fix/state-reward-exact`。
- [ ] 步骤 2：写 dump 工具；对 exp457（都柏林）、exp274（1x1）、exp338（1x3）各跑 200 步，输出到 `experiments/analysis/data/stream_baseline/`。
- [ ] 步骤 3：同一命令再跑一次，`--compare` 结果差为 0，证明流是确定的。
- [ ] 步骤 4：提交工具与快照。

### Task 1：参数构造共用模块（评审第 5 项）

**文件：**
- 新建：`experiments/exp_params.py`
- 修改：`experiments/train.py:182-227`、`experiments/tools/dublin/ckpt_env.py:33-56`、`experiments/eval_unified.py:221-250`
- 测试：`experiments/tests/test_exp_params.py`

**接口：**
- 产出：`obs_kwargs_from_cfg(cfg: dict) -> dict`（train.py 现有全部键，含 obs_awt_log、obs_header、obs_slot_min_green，另加 obs_all_connections→all_connections、obs_slot_memo→slot_memo）；`env_kwargs_from_cfg(cfg) -> dict`（wait_tracker、all_connections）；`reward_wait_basis(cfg) -> str`；`turnmap_from_meta(tls_meta: dict) -> dict[int, tuple[str, str]]`，同一索引下各连接标签不一致时 raise ValueError。

- [ ] 步骤 1：写测试：exp457 配置经 obs_kwargs_from_cfg 得到的字典等于手写期望；缺键时不出现；turnmap_from_meta 对 18h meta 全部通过，对构造的不一致 meta 抛错。
- [ ] 步骤 2：跑测试，失败。
- [ ] 步骤 3：实现模块；三个入口改为调用它。
- [ ] 步骤 4：跑测试通过；跑 regress_dryrun 对账；任务 0 的三条流 `--compare` 差为 0。
- [ ] 步骤 5：提交。

### Task 2：观测建表遍历全部连接（评审第 1 项）

**文件：**
- 修改：`sumo_rl/environment/observations.py` PriorityMovementObservationFunction 构造函数与 `_build_movement_tables`（734-795 行）
- 测试：`experiments/tests/test_obs_all_connections.py`

**接口：**
- 产出：构造参数 `all_connections: bool = False`；True 时对每个索引的每条连接登记进口车道、出口车道、边对；车道集合去重，容量每车道每槽只加一次。

- [ ] 步骤 1：写测试（都柏林 18h 网，重绑 8std meta）：False 时 389332 的 E-T 槽容量 1.51、出口 1 条、lane_occ 车道 1 条；True 时 3.02、2 条、2 条；133307283 的 N-T 出口 True 时 2 条；其余 16 路口两种取值下 `_slot_cap/_slot_out/_slot_in_lanes` 相等；1x1 网两种取值全等。
- [ ] 步骤 2：跑测试，失败。
- [ ] 步骤 3：实现。
- [ ] 步骤 4：测试通过；任务 0 流比较差为 0。
- [ ] 步骤 5：提交。

### Task 3：运动缓存可关闭（评审第 2 项）

**文件：**
- 修改：`observations.py` `_vehicle_slot`（832-842 行）、`__call__` 开头的阀门（847-850 行）
- 测试：`experiments/tests/test_obs_slot_memo.py`

**接口：**
- 产出：构造参数 `slot_memo: bool = True`；False 时每次按 (进口边, 路线下一条边) 重查，不写 `_vid_slot`，不触发阀门。

- [ ] 步骤 1：写测试（1x1 网）：跑到某辆车在进口道上，用 `vehicle.changeTarget` 改它的出口边；slot_memo=True 时该车槽不变，False 时槽随新路线变；两种取值在不改路线时流逐位相同。
- [ ] 步骤 2：失败。
- [ ] 步骤 3：实现。
- [ ] 步骤 4：通过；任务 0 流差为 0。
- [ ] 步骤 5：提交。

### Task 4：VisitWaitTracker 与环境钩子

**文件：**
- 新建：`sumo_rl/environment/visit_wait.py`
- 修改：`sumo_rl/environment/env.py` 构造参数（91 行起）、reset（288-320 行）、`_sumo_step`（452 行）
- 测试：`experiments/tests/test_visit_wait.py`

**接口：**
- 产出：`VisitWaitTracker(sumo, ts_lanes: dict[str, list[str]], strict: bool = False)`，属性 `dt`、`W: dict[(ts_id, vid, edge) -> float]`、`missing: int`；方法 `step()`（同一仿真时刻重复调用直接返回）、`get(ts_id, vid, lane) -> float`（strict 缺键 raise KeyError，否则 0 且 missing+1）、`reset()`。环境参数 `wait_tracker: bool = False`，属性 `env.visit_tracker`（关时 None）；reset 后重建；`_sumo_step` 在 simulationStep 之后调用 step()。

- [ ] 步骤 1：写测试（1x1 网，waiting_time_memory=100000）：跑 300 秒，测试内独立按秒记录每辆车在进口道上的 acc 增量，与 tracker 各次访问结束时的 W 之和逐车相等；同一时刻重复 step() 不重复累加；reset 后 W 为空；strict 缺键抛错；步长改 0.5 秒时 W 按 0.5 累加；用 `vehicle.remove` 模拟传送，条目在下一秒消失。
- [ ] 步骤 2：失败。
- [ ] 步骤 3：实现。
- [ ] 步骤 4：通过；wait_tracker 缺省时任务 0 流差为 0。
- [ ] 步骤 5：提交。

### Task 5：状态 visit 口径

**文件：**
- 修改：`observations.py` 构造函数校验（awt_basis 允许 visit）、`__call__` 两个分支（866-900 行）、下游 mean_awt 分支（986 行）
- 测试：`experiments/tests/test_obs_awt_visit.py`

**接口：**
- 消费：`ts.env.visit_tracker.get(ts.id, vid, lane)`。
- 产出：awt_basis="visit" 时 w 取 tracker；不碰 `_awt_entry`；下游字段含 mean_awt 且 visit → ValueError。

- [ ] 步骤 1：写测试（1x1）：visit 口径下每槽每等级 mean_awt 等于测试内按 tracker 独立聚合的值；awt_basis=local 时与任务 0 快照逐位相同；visit 与 local 的差只出现在车辆进入后 4 秒内停下的访问；visit 口径配 downstream mean_awt 抛错；visit 口径下 `_awt_entry` 始终为空。
- [ ] 步骤 2：失败。
- [ ] 步骤 3：实现。
- [ ] 步骤 4：通过；流差为 0。
- [ ] 步骤 5：提交。

### Task 6：奖励 visit 口径

**文件：**
- 修改：`sumo_rl/environment/rewards.py` `make_priority_avg_waiting_reward`（896 行）、新增 `make_avg_waiting_reward`；`experiments/train.py` 奖励解析（230-282 行）；`ckpt_env.py`、`eval_unified.py` 同步
- 测试：`experiments/tests/test_reward_wait_visit.py`

**接口：**
- 产出：`make_priority_avg_waiting_reward(table: dict, wait_basis: str = "legacy")`；`make_avg_waiting_reward(wait_basis: str = "legacy") -> Callable`，legacy 时返回现有 `_avg_waiting_reward`；visit 时两者只读 tracker、不读写 env.vehicles。配置键 `reward_wait_basis`，入口按它决定 wait_tracker。

- [ ] 步骤 1：写测试（1x1）：visit 口径 reward 等于测试内按公式独立计算的值（含空等级跳过、空集 0）；legacy 时与快照逐位相同；visit 下 `env.vehicles` 中无 reward 写入痕迹（由 reward 单独调用验证）；reward_scale 与 reward_floor 包装后仍等于包装公式。
- [ ] 步骤 2：失败。
- [ ] 步骤 3：实现。
- [ ] 步骤 4：通过；regress_dryrun 对账；流差为 0。
- [ ] 步骤 5：提交。

### Task 7：TrafficSignal.out_lanes 全连接（评审第 6 项）

**文件：**
- 修改：`sumo_rl/environment/traffic_signal.py` 构造函数、107-118 行；`env.py` 两处 TrafficSignal 构造传参
- 测试：`experiments/tests/test_out_lanes_all_connections.py`

**接口：**
- 产出：TrafficSignal 构造参数 `all_connections: bool = False`；环境参数 `all_connections` 透传。

- [ ] 步骤 1：写测试：都柏林 389332 的 out_lanes 默认 2 条、True 时 4 条；1x1 两种取值相等。
- [ ] 步骤 2：失败。
- [ ] 步骤 3：实现（去掉 107-114 行的废弃调试变量）。
- [ ] 步骤 4：通过；流差为 0。
- [ ] 步骤 5：提交。

### Task 8：cluster_21620852 分槽版 8STD 网（评审第 4 项）

**文件：**
- 修改：`experiments/tools/dublin/reindex_8std.py` main
- 新建：`nets/dublin/dublin_8std_splitslot.net.xml`、`nets/dublin/dublin_8std_splitslot_meta.json`、`nets/dublin/weekday_18h_reroute_v2/dublin_weekday_18h_probe11h_8std_splitslot.sumocfg`
- 测试：`experiments/tests/test_reindex_split_slots.py`

**接口：**
- 产出：命令行 `--split-slots --tag splitslot`；输出文件名带 tag，不覆盖旧文件；meta 顶层加键 `slot_split`。

- [ ] 步骤 1：写测试：生成结果中除 cluster_21620852 外 17 个路口的 tlLogic 与 meta 与现有文件相等；该路口 links 2→(E,R)、6→(S,T)，mask 为 [T,T,T,T,F,F,T,T]。
- [ ] 步骤 2：失败。
- [ ] 步骤 3：实现开关并生成文件；写 sumocfg。
- [ ] 步骤 4：通过；sumo 加载新网无 Error。
- [ ] 步骤 5：提交。

### Task 9：全网审计与都柏林调试运行

**文件：**
- 新建：`experiments/tools/dublin/state_audit_full.py`、`experiments/tools/dublin/state_reward_debug.py`
- 新建：`experiments/analysis/STATE_REWARD_DEBUG_2026-10-26.md`

**接口：**
- 产出：审计工具：静态表对全连接核对、槽内灯色一致性、丢弃车辆统计、索引标签一致性；调试工具：跑一集，对每个路口每步独立重算 count/queue/mean_awt/下游/lane_occ/is_green/最短绿与 reward，与环境输出逐位比对，并记录车辆跨 RL/非 RL 路口的转换、共享车道双槽归属、多连接索引的车道覆盖。

- [ ] 步骤 1：实现两个工具。
- [ ] 步骤 2：在旧网与分槽网、legacy 与 visit 四种组合上各跑一集 457 策略；所有重算差为 0；记录新发现的问题。
- [ ] 步骤 3：写报告文档；提交。

### Task 10：同轨迹双算与新配置模板

**文件：**
- 新建：`experiments/tools/dublin/dual_run_compare.py`
- 新建配置：`experiments/configs/exp459_dublin18h_reroute_v2_amb11h_541_8std_slot12_f1_exact.yaml`（457 + 全部新键 + 分槽网）及 458 对应版
- 新建：`experiments/analysis/STATE_REWARD_EXACT_VERIFICATION_2026-10-26.md`

- [ ] 步骤 1：双算工具：旧口径驱动策略，同时算新 state、新 reward，输出逐维差异分布与逐路口 reward 差；对 457（18h）、315（11h）、313（02h）、274、338 各跑一集。
- [ ] 步骤 2：写配置模板（只建文件，不启动）。
- [ ] 步骤 3：写验证报告：旧配置逐位不变的证据、各项差异量级、重训范围建议；提交。
