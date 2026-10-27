# weekday_18h_reroute_v2_samevtype — 18:00 高峰改道第二版的 "三车类同物理参数" 版

建立 2026-10-27 (用户令 "先做一个新的 18H, 三个物理参数一致, 跟 CAR 原来一样")。原文件夹 `../weekday_18h_reroute_v2/` 保留未动。
依据: `experiments/analysis/DESIGN_UPSTREAM_AND_PRIORITY_AGG_2026-10-27.md` §7 第 1 点 (用户决定: 三类车物理参数完全一致, 只差 vType id)。
训练期间不得改动本文件夹的路由文件 (SUMO 每集重读)。

## 改了什么 (只改 vType 定义行; 车辆、路线、出发时刻、sumocfg 与原文件夹逐字节相同)

| 文件 | 原 vType | 现 vType |
|---|---|---|
| bus_weekday_18h.rou.xml | `id="bus" vClass="bus" length="12.0" color="0,0,1" lc*…` | `id="bus" color="0,0,1" lc*…` |
| amb_weekday_18h_probe11h.rou.xml, amb_weekday_18h.rou.xml, probe_amb_18h.rou.xml | `id="ambulance" vClass="emergency" guiShape="emergency" color="0,1,0" lc*…` | `id="ambulance" color="0,1,0" lc*…` |
| car_weekday_18h.rou.xml, probe_bus/probe_car/busprobe | 不变 (car: `id="car" color="1,0,0" lc*…`; 另三个文件不定义 vType) | 不变 |

三个 vType 现在只差 id 与颜色 (颜色只影响 GUI), 物理参数全部取 SUMO passenger 类默认 (与 car 原来一样):
车长 5.0 m / 车宽 1.8 m, accel 2.6, decel 4.5, emergencyDecel 9.0 m/s², maxSpeed 55.6 m/s, speedDev 0.1, tau 1.0, vClass passenger;
换道参数 lcStrategic/lcCooperative/lcSpeedGain/lcKeepRight = 0 三者原本就一致。
原来的差别 (bus: 12 m、accel 1.2、maxSpeed 27.8、可走公交道; ambulance: emergency 类, 可右侧超车/借对向车道/无优先路口可超车) 全部消失。
已核对分槽网 `../dublin_8std_splitslot.net.xml`: 没有 allow="bus" 车道、没有 disallow passenger 车道、没有 emergency 标记, 所以改 vClass 不会让任何路线失效。

## 固定配时整小时加载检查 (路网默认配时, seed 123, dublin_weekday_18h_probe11h_8std_splitslot.sumocfg)

| 文件夹 | 插入辆 (需求 6092) | 小时末在网 |
|---|---|---|
| 原 weekday_18h_reroute_v2 | 5782 | 275 |
| 本文件夹 | 5772 | 277 |

(差别来自公交车长 12 → 5 m 改变了插入与跟驰; 三份路由全部加载无错误。)

## 用途

- 回答 "权重的作用" 而不是 "12 米公交的通行": 与原 18h 公交结果不可直接相比, 这是有意为之。
- 对应实验配置: exp469 (= exp461 只换本文件夹的 sumocfg 与三份路由; 531)。111 / 351 臂按同样做法再建。
- 换车型孪生评估用本文件夹的 probe_{amb,bus,car}_18h.rou.xml (车型已同参数)。
