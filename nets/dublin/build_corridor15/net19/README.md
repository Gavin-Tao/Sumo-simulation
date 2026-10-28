# 最终路网：19 个 RL 信号（2026-10-28）

本轮路网阶段定稿：保留之前 18 个路口，增加 DCC 参考网中已有信号定义的 **9101555**，不替换任何现有 RL 路口。两个新 corridor 场景共用此 net/meta。原 southclosed 场景、算法和训练代码未改，没有生成正式车流或启动训练。

- [最终 net](../../dublin_8std_splitslot19_southclosed_corridor.net.xml)
- [配套 meta](../../dublin_8std_splitslot19_southclosed_corridor_meta.json)
- [全网图](final_review/final_net_overview.png)，[矢量图](final_review/final_net_overview.svg)
- [18h GUI 预览](../../weekday_18h_15_southclosed_corridor/net_preview.sumocfg)
- [pairs50 GUI 预览](../../weekday_pairs50_15_southclosed_corridor/net_preview.sumocfg)

两个目录名仍带 `15`，是建立副本时的历史名称；`net_preview.sumocfg` 已加载 19 信号网，不加载车流。目录中旧的 `scenario.sumocfg`、路由、时刻表、需求文件仍是未标定的 15 信号草稿，不能用于新网训练。

## 本次决定与改动

加入 9101555 的依据是明确的实验用途：用独立支路优先请求与 Mercer 主流形成竞争，并与上游 26165090 组成连续控制路段。旧场景有没有车流不作为判据，因为正式车流会重建。它的下游约束没有因此消失，见后文。

| 9101555 道路 | 车道 | 实际长度 |
|---|---:|---:|
| Mercer 主进口 `15464767#0`，来自 26165090 | 2 | 87.89 m |
| Digges 支进口 `14039867#0` | 1 | 86.94 m |
| Mercer 出口 `15464767#5`，到 9101554 | 2 | 70.85 m |

有效标准动作是 **a0 / a3**，物理绿灯字符串分别是 `GGGr` / `rrrG`。前三条连接是主路两条直行及主路右转，第四条是支路右转。当前 SUMO 的 foes 矩阵将支路右转与前三条连接都记为冲突。因此“主路第一车道直行不与支路冲突”不符合本 net；没有将其作为恒绿运动。

只改 9101555 的控制属性并用重索引工具生成相位。相对于已完成的 net18：

- 原 18 个路口的完整 meta 逐项一致。
- 全部既有普通车道的长度、形状一致；物理连接集合不变。
- 其他节点连接路权不变，包括保留 9101554 的真实让行关系。
- 受控连接数从 116 增为 120，原非 RL 的四条连接纳入观测与控制。

构建断言和输入哈希见 [build_report.json](build_report.json)。19 个 RL 节点均能在 DCC 参考网中找到 traffic_light 定义；这不等于现场现状核查。

## 已完成的基础改造继续保留

道路合并、389332/1448442—389281 的连接修正、cluster_2498303858 的内部折返修正，以及边界入口扩建，继承 [net18 阶段记录](../net18/README.md)。本次没有撤销或重新切分道路。

- 1448436 西进口为完整 82.74 m；College Green 东进口为完整 125.79 m。
- College Green 北臂保留信号控制和 74.98 m 进口；3206753875 已改为独立人工入口/出口，掉头连接删除。
- College Green 北/东及 1448436 西都有独立上游入口；全网 16 条合法边界入口都先经过非 RL 道路。正式路由仍须明确使用这些入口，不能直接在 RL 进口出发。
- 2397311044 保留为边界出口，不加反向注入。
- 133307292 不加入 RL；其他未确认候选不加入，不删除原路口。

## 新副本的禁换道参数修正

完整路线验证发现，原来的 `lcStrategic=0`、`lcCooperative=0` 并非完全禁换道；旧参数下两种控制方式均记录到 5 次换道。SUMO 文档明确区分 0 和 -1：[车辆及换道参数定义](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#lane-changing_models)。

已仅将两份新 corridor 副本的 `vtypes_same.add.xml` 改为 `lcStrategic=-1`、`lcCooperative=-1`，速度收益和靠边换道参数仍为 0。三类车除 id/颜色外参数逐项一致，两份副本也一致；其他物理参数不变。原场景文件未动。新参数下重新运行局部及完整路线检查，换道数均为 0。

旧参数诊断记录保留在 `../net18/validation/full_routes/first_run_lanechanges/`，修改前的新副本类型文件保存在 `../net18/final_review/before_types_fix/`。

## 验证

| 测试 | 控制/种子 | 每轮到达 | 每轮 state / reward 独立核对 |
|---|---|---:|---:|
| 局部连接及新入口 | 固定 123、随机 123、随机 124 | 381 / 381 | 各 4560 次 |
| 非 RL 边界入口到真正边界出口 | 固定 123、随机 123 | 360 / 360 | 各 13680 次 |

所有五轮均为：0 碰撞、0 teleport、0 换道、0 tracker 缺条目、集末 0 在网车、0 pending。检查区域内未出现逐秒车身矩形交叠。state 固定 336 维；等待参照使用长窗口 SUMO 累计等待的逐秒增量，独立于 visit tracker。

静态车道图中全部 120 条受控连接均有外部入口到边界出口的合法不换道路径。完整路线车辆仍由 SUMO 决定中间连接的实际目标车道；不把静态指定路径当作每条连接都被逐车强制复现。完整诊断路线可能重访道路，用于检验连通和计数；它们不是正式需求或配对路线。

输出见 `validation/*_report.json`、`validation/full_routes/*_report.json`、[final_net_audit.json](final_review/final_net_audit.json) 和最终哈希清单。测试证明连接与 state/reward 链路工作，未证明正式车流不会过饱和或训练必然产生某种优先级排序。

## 下一阶段车流必须满足的条件

9101554 的 Mercer 方向仍让行于 `2294382#1`。可以把后者设为低需求或零需求作为受控实验起点，但必须检查 9101554 及其更下游的实际接收能力：低竞争来车也不能保证下游队伍不会回顶。要累计整集所有车辆在出口和上游非 RL 路段的停车，不能只看 RL 进口或集末剩余车辆。

Digges 支路 100–150 辆/h、主支比 0.3–0.5 只是待标定的起始候选，不是已验证容量。主路车辆应经 26165090，公交与救护车在支路使用相同的路线和出发时刻分布；111/531/351 比较固定同一需求，检查每类相对自身基线的变化。

真实短进口仍有五处：1448436 南 21.61 m、26165090 北 40.17 m、College Green 南 35.15 m、cluster_1998326407 北 45.27 m、cluster_21620852 西 43.33 m。保留其真实分支和连接；后续必须检查队尾是否超出观测范围。既有非 RL 合流/让行点也未全部删除，清单在最终审计 JSON 中。

标定应同时检查：请求/插入/完成车数、pending 与出发延迟、在网车辆是否持续增长、上游和出口停车、下游阻塞、分转向的有效优先级竞争机会。新增缓冲道路不会自动限流；路网阶段完成后再固定正式需求，并统一供各对照臂使用。

## 重现

已有 net18 构建产物后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/build_net19.py
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/validate_net19.py fixed 123
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/validate_net19.py random 123 --full-routes
PYTHONDONTWRITEBYTECODE=1 /home/xiaowen/anaconda3/envs/sumo_rl/bin/python -B nets/dublin/build_corridor15/review_final_net19.py
```
