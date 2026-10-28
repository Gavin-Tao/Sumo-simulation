# 18h 派生 corridor 副本：目前仅完成新路网

2026-10-28，用户要求先做 net。当前采用 **18 个 RL 信号的 nocg 版本**：由 net19 撤销 College Green 的 RL 控制，保留 9101555。它与较早的 splitslot18 版本不是同一个路口集合。目录名中的 15 沿用建立副本时的名称。

查看新路网请打开 [net_preview.sumocfg](net_preview.sumocfg)。它只加载新 net，不加载车辆。

当前 [net](../dublin_8std_splitslot18_southclosed_corridor_nocg.net.xml)、[meta](../dublin_8std_splitslot18_southclosed_corridor_nocg_meta.json) 和 [构建差异](../build_corridor15/net18_nocg/build_report.json)。net19 及原 southclosed 场景均保留；历史构建说明见 [net19](../build_corridor15/net19/README.md)。

College Green 按用户最新要求恢复 **v2 的局部拓扑与路权**：参考 `dublin_8std_splitslot.net.xml`，取消信号灯，去掉后来补出的北臂（含入口、出口及缓冲边）和东向出口 `4476041#0`。保留原东进口与南进口；东进口两条连接向西且为主路 M，南进口三条连接向西且为让行 m。五条连接在合并边 ID 映射后与 v2 完全一致。此前合并的 389332 道路和东侧上游缓冲保留。15 条边界入口先进入非 RL 道路；剩余 21 条 passenger-only 车道权限已与其他道路统一，允许 bus/emergency。

其余 18 个路口完整 meta 不变；三臂路口边界重建后，389332 东进口从 net19 的 87.74 m 变为 88.24 m，东出口从 87.61 m 变为 88.10 m，其他 RL 进口长度不变。没有拆分 2110697#0，仍为 297.36 m。构建报告列出全部端点差异。

与 v2 一样，西侧来路 `304634278#0` 在本节点没有后续连接；这项截断是恢复旧布局的一部分。后续正式需求必须明确是否使用此处作为研究边界出口，不能误当贯通道路，也不能沿用上一版经 `4476041#0` 的路线。南进口让行停车仍需标定。上一版保留新增出口的 nocg 文件已备份到 `build_corridor15/net18_nocg_before_v2_restore/`。

本目录现存的 `scenario.sumocfg`、route 文件、`generation_report.json`、`demand_targets.json` 是增加信号之前的 **15 信号草稿**，未完成标定，尚未适配当前路网。不要把它们直接配到新网或用于训练。正式车流等待后续重做；没有启动训练。

`vtypes_same.add.xml` 已修正禁换道设置：三类车统一 `lcStrategic=-1`、`lcCooperative=-1`，其余属性不变；先前的 0 仍允许部分换道。car/bus/ambulance 除 id、颜色外参数一致，旧场景文件未改。

原 18h 的 262 辆公交及 1 辆救护车不能直接作为救护车优先的充分样本；本场景角色、流量与配对评价设计待车流阶段明确。
