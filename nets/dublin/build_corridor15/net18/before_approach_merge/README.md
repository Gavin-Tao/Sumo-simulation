# Corridor 路网阶段交付（2026-10-28）

当前交付只涉及 net、meta 和连通性验证。用户已要求先完成路网，正式车流重建、需求标定、训练均未开展到这个 18 信号版本。

- 路网：[dublin_8std_splitslot18_southclosed_corridor.net.xml](../../dublin_8std_splitslot18_southclosed_corridor.net.xml)
- 动作表：[dublin_8std_splitslot18_southclosed_corridor_meta.json](../../dublin_8std_splitslot18_southclosed_corridor_meta.json)
- 基础：原 southclosed → corridor 合并版本 → 局部参考 `nets/dcc/dcc.net.xml`。
- 原 southclosed net/meta 未覆盖；算法、环境、训练、评估代码没有修改。
- 18 指 15 个原保留信号加本次 3 个；与原始 18 信号网的路口集合不同。

## 新增控制点

| 路口 | 本次内容 | 有效标准动作 |
|---|---|---|
| 659799 | 补东、北臂，接通西进口；四臂车道与转向参考 DCC | a0、a1、a2、a7 |
| 659805 | 保留原车道和 6 条连接，恢复信号控制 | a0、a1、a3 |
| cluster_1426051358_1426100603_3206752725_3783453672_3783453675 | 接通 389332 出口后续道路，补 Foster Place 及东出口；按 DCC 恢复左右转 | a1、a2、a4、a6、a7 |

四臂不保证八种标准动作都有效：仍按允许转向、冲突、兼容扩展和相位去重生成 mask。所有连接至少被一个有效相位服务；绿相位没有许可小写 g，没有冲突绿或同时汇入同一目标车道的绿连接。

659799 的车道：东进口 lane 0 左转/直行共享、lane 1 直行；北进口 lane 0/1/2 分别左转/直行/右转；西进口一条直行；南进口一条左转/右转共享；东、西出口均两条车道。

College Green 的西进口/出口使用现有合并边 `304634278#0` / `-304634278#0`，没有恢复参考网中间的短边节点。北臂 Foster Place 保留参考网的尽端回转连接，不能按普通贯通干道理解。东侧不导入参考网的 gneE66/gneE68 短绕行环和额外信号。

659799 东、北进口和 College Green 东进口各有 60 米上游缓冲段，逐车道直连，没有新增交叉让行。它们是模型边界缓冲，不计入 RL 进口长度，也没有为其生成正式注入车流。

133307292、9101555、cluster_135109528_9101656 本次均未增加 RL。133307292 的受控合流方案等待用户确认。

## 已合并的道路与长度

26912118、26911837 的北支关闭，主路合并；原 southclosed 已完成的 1448442—389281 连通及 cluster_389667 注入分离继续保留。

| RL 进口/道路 | 原 southclosed | 最终 net18 |
|---|---:|---:|
| 1448436 西，668344588，每条车道 | 10.30 m | 47.69 m |
| 同一西侧通道的上游缓冲 | 无独立缓冲 | 40.00 m，不计 RL 容量 |
| 1448436 东，-3791745#3，每条 | 61.80 m | 120.08 m |
| 389332 西，3791745#4，每条 | 47.07 m | 119.84 m |
| 389332 东，-304634278#0，每条 | 88.24 m | 87.74 m |
| 1448442 南，532427444#7，每条 | 110.24 m | 110.24 m |
| 389281 北，-532427444#3 | 110.04 m | 110.04 m |
| 1430847303 东，3791031#0，每条 | 124.44 m | 122.74 m |
| 659802 北，4074528#0 | 162.04 m | 161.65 m |

最后两处及 389332 东侧的少量缩短来自相邻路口边界重建，不是手填长度。容量始终读取 SUMO 的实际车道长度；测试中据此独立重算 state。

1448436 西侧未额外在原边界节点引入一股穿过现有转弯流的车，而将合并通道前 40 米保留为缓冲，所以 RL 进口是 47.69 米，不能声称整段约 88 米都纳入观测。

新 659799 的四臂进口长度分别是北 60.88 m、东 305.34 m、西 123.37 m、南 161.65 m。659805 进口仍为 193.05 m 和两条 111.11 m。新 College Green 的 Church Lane 进口仍是两条 35.15 m，北臂 74.98 m，西臂两条 87.61 m，东臂两条 65.79 m。

完整长度差异见 [all_rl_length_changes.json](all_rl_length_changes.json)。缓冲段及其他上游非 RL 停车仍不在本车道 visit 奖励中；本次没有扩展观测范围，后续车流标定必须另查这些等待。

## cluster_2498303858 的几何修正

修正 `357012332#0` 的上游端点，消除内部连接向进口折返。52879956 两条进口由 54.20 m 变为 53.43 m。出口几何变更使自动标准转向由 S-T 变为 S-L，mask 从 a0/a2 变为 a1/a2；物理绿灯状态仍是 `GGGGr` / `rrrGG`，该连接仍恒绿。

因此不能说相对于原 southclosed 所有动作编号都不变。除这一处外，原 15 个路口的标准动作映射保持不变；从 corridor15 再增加三个路口时，原 corridor15 的映射全部保持不变。必须使用本 net 随附的 meta。

## 验证范围与结果

`validate_net18.py` 生成的是局部连通性探针，不是正式需求：覆盖全部 116 条受控连接，分别用 car/bus/ambulance 通行，共 348 辆；另 3 辆验证 Foster Place 回转及同一路口再次访问。

| 控制方式 | 车辆到达 | state 逐位重算 | reward 重算 | 碰撞/teleport/换道/tracker 缺失 |
|---|---:|---:|---:|---|
| 固定轮转，种子 123 | 351/351 | 4320 次通过 | 4320 次通过 | 全为 0 |
| 随机，种子 123 | 351/351 | 4320 次通过 | 4320 次通过 | 全为 0 |
| 随机，种子 124 | 351/351 | 4320 次通过 | 4320 次通过 | 全为 0 |

每次运行 1200 仿真秒，18 个路口均输出 336 维观测。等待参照使用 SUMO 长窗口累计等待的逐秒增量，独立于 tracker；覆盖非零等待。新三个路口及 cluster_2498303858 每秒检查车身矩形，测试样本未发现交叠。这是采样验证，不宣称所有车流与连续时刻均已证明无交叠。

没有用这些稀疏诊断车辆证明“不拥堵”或“优先级效果好”；正式车流、训练及配对评价仍待下一阶段。

构建及复核：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/build_net.py
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/build_net17.py
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/build_net18.py
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/validate_net18.py fixed 123
```

结果在 `validation/*_report.json`。`preservation_audit.json` 核对原 southclosed 快照：只有此前已授权的 27 辆终止公交路线修复，不存在额外源文件变化。
