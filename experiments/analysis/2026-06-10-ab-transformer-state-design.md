# A / B / Transformer State 设计 Spec

- 日期: 2026-06-10
- 作者: taox@tcd.ie (TCD PhD)
- 状态: 设计待评审 (brainstorming → spec)
- 相关文件:
  - `state_design_AB_worked_example.txt` (A vs B + 堵塞探测甲/乙 的逐格推演)
  - `generic_agent_state_design.txt` (lane-token Transformer 的完整泛化设计)
  - `State_Design_A_B_Transformer.pptx` (common skeleton 示意)
  - 现有代码: `sumo_rl/environment/observations.py`, `traffic_signal.py`, `metrics.py`, `bnf_parser.py`, `BNF/traffic_rules.bnf`

---

## 0. 本次设计敲定的决策 (decision log)

| 决策点 | 选择 | 理由 |
|---|---|---|
| 重设计目的 | **实际实现并训练**，对比 A vs B vs Transformer | 不是纯论文叙事；schema 必须落到现有数据可直接 build |
| 优先级维度组织 | **5 个固定优先级槽 {1..5} + active(bnf_present)** | 论文核心 contribution: agent 看"优先级桶"不看车型 → 换 BNF zero-shot |
| 堵塞探测 servable/blocked | **第一版不做**（φ 只有 count/queue/awt） | 不是必须；保留为 Phase-2 反应式甲 的预留位 |
| 路口规模 | **单路口先行，架构预留多路口** | 现有 metrics/lanes 是 1x1；encoder 做成 set-based 以后不返工 |
| BNF context | **第一版用固定 `type→priority` 表** | 立即可 build+train；context 留作"动态 BNF"实验 |
| 走法 | 先口头过 encoder/action，已确认 → 落 spec | (本文档) |

---

## 1. 背景与目标

在单个 SUMO 信号路口上，实现并对比**三种 state 组织方式**，隔离"状态组织"这一个变量，回答"按相位 / 按转向 / 按车道组织 state，对带优先级的信号控制各有什么影响"。

三种共用：同一个优先级分桶特征器 `φ`、同一个全局头、同一个动作语义（选绿相位）、同一个 reward、同一套 min/max-green 掩码、同一套评估指标 (`EpisodeMetricsCollector`)。**唯一变量 = 车辆聚合粒度**（相位 / 转向 / 车道）。

---

## 2. 当前数据现状（盘点，设计必须尊重）

**已有、可直接复用:**

- 4 个车型: `car, truck, bus, ambulance` (`metrics.py` TYPES)
- `TrafficSignal` 真实 API:
  - `ts.lanes` (incoming 去重)、`ts.out_lanes`
  - `ts.signal_controlled_lanes` / `ts.always_green_lanes`（已区分自由右转）
  - `ts.lanes_length`、`ts.MIN_GAP`
  - `ts.green_phases[k].state`（G/g/r/s/y 字符串）、`ts.num_green_phases`、`ts.green_phase`
  - `ts.time_since_last_phase_change`、`ts.min_green`、`ts.yellow_time`
  - `sumo.trafficlight.getControlledLanes(id)`（raw，索引对齐 phase.state）
  - `sumo.trafficlight.getControlledLinks(id)`、`sumo.lane.getLinks(lane)`（含 direction）
  - `sumo.lane.getShape(lane)`、`sumo.junction.getPosition(id)`（几何，用于 approach_angle）
  - `sumo.vehicle.getTypeID / getSpeed / getAccumulatedWaitingTime / getRoute`
- 现有 obs 类（flat per-lane per-type counts/waits）: `PriorityCtrlBCAObservationFunction` 等
- 评估指标 `EpisodeMetricsCollector`（per-type / per-ts / ambulance / 局部 wait / throughput）

**缺口（要新增）:**

1. `vtype(+context) → priority ∈ {1..5}` 的映射函数（第一版固定表）
2. 优先级分桶特征器 `φ`（把"按车型数"换成"按优先级桶数"）
3. 相位→放绿车道集 / 车→转向 / 车道几何 三个解析器
4. 三个新 `ObservationFunction` 子类 + 三个网络主体 + 可选 FRAP 动作头

---

## 3. 核心设计: 一个 φ，三种聚合

```
                       ┌─────────────────────────┐
   一组车辆 V  ───────▶ │  φ(V) → 20 维优先级桶向量 │
                       └─────────────────────────┘
                                  ▲
       ┌──────────────┬───────────┴───────────┬──────────────┐
   A: 按相位放绿车道分组   B: 按转向分组          T: 按车道分组
   K 个相位单元          12 个转向槽            N 个车道 token
```

A / B / Transformer **只差"V 怎么分组"**，φ、全局头、动作、reward 全相同。

---

## 4. 固定 type→priority 表（第一版）

```python
# sumo_rl/environment/priority_map.py
TYPE_TO_PRIORITY = {
    "ambulance": 5,
    "bus":       3,
    "truck":     2,
    "car":       1,
}
DEFAULT_PRIORITY = 1          # 未知车型兜底
PRIORITY_LEVELS = (1, 2, 3, 4, 5)   # 固定取值空间

def vehicle_priority(sumo, vid) -> int:
    return TYPE_TO_PRIORITY.get(sumo.vehicle.getTypeID(vid), DEFAULT_PRIORITY)
```

- `bnf_present`（全局头里的 active）= 这张表用到了哪几级 → 本表 = `{1,2,3,5}` → `[1,1,1,0,1]`。
- Phase-2「动态 BNF」: 把 `vehicle_priority` 换成真正调用 `BNFParser` + 车辆 context 属性；**agent / φ / 网络都不用改**，只换映射函数 → 这就是 zero-shot 不变性的落点。

> ⚠️ 待解决: `BNF/traffic_rules.bnf` 中 `policy17 / policy22` 输出 `"6"`，超出 `<priority> ::= 1..5`。落动态 BNF 前必须修：clamp 6→5 或修正为预期级别。第一版固定表不受影响。

---

## 5. φ — 优先级分桶特征器（20 维）

布局 = 5 级 × 4 字段 `[count, queue, mean_awt, max_awt]`，按 p=1..5 顺序排开。

```python
def phi(sumo, vehicle_ids, normalize=True, capacity=None, awt_scale=100.0):
    """把一组车辆压成 20 维优先级桶向量。
    字段: 每级 [count, queue, mean_awt, max_awt]
    normalize=True 时: count,queue /= capacity ; mean_awt,max_awt /= awt_scale
    """
    import numpy as np
    cnt   = {p: 0   for p in PRIORITY_LEVELS}
    queue = {p: 0   for p in PRIORITY_LEVELS}
    sumw  = {p: 0.0 for p in PRIORITY_LEVELS}
    maxw  = {p: 0.0 for p in PRIORITY_LEVELS}
    for vid in vehicle_ids:
        p = vehicle_priority(sumo, vid)
        w = sumo.vehicle.getAccumulatedWaitingTime(vid)
        cnt[p]  += 1
        sumw[p] += w
        if w > maxw[p]:
            maxw[p] = w
        if sumo.vehicle.getSpeed(vid) < 0.1:
            queue[p] += 1
    out = []
    for p in PRIORITY_LEVELS:
        c = cnt[p]
        mean = (sumw[p] / c) if c else 0.0
        if normalize:
            cap = capacity if capacity and capacity > 0 else 1.0
            out += [c / cap, queue[p] / cap, mean / awt_scale, maxw[p] / awt_scale]
        else:
            out += [c, queue[p], mean, maxw[p]]
    return np.asarray(out, dtype=np.float32)   # shape (20,)
```

字段含义:

| 字段 | 含义 | 为什么 |
|---|---|---|
| `count_p` | 该桶车数 | 需求量 |
| `queue_p` | 其中 halting (speed<0.1) 数 | 区分排队 vs 通过 |
| `mean_awt_p` | 该桶累计等待均值 | 普通车等待压力 |
| `max_awt_p` | 该桶累计等待最大值 | **稀疏高优先车不被均值稀释**（单辆救护车 awt 大，mean 会被同桶拉平，max 保住它） |

- "该桶在这组车里没有" = `count=0`，不需额外 flag（全局头的 `bnf_present` 已标"BNF 用不用第 p 级"）。
- **归一化**: 展示/调试用原始值（见 §8 worked example）；训练用 `normalize=True`，`capacity = lanes_length / (MIN_GAP + 车长)`（与 `get_lanes_density` 一致），`awt_scale=100`。归一化是跨车道/跨路口迁移的前提（worked example 注脚）。
- **预留位**: 以后加反应式甲 blocked-bit → 每级 `[count, queue, mean_awt, max_awt, blocked_belief]` = 5×5=25，现在 mask 掉，不破坏结构。

---

## 6. 全局头（7 维，三种共用，prepend 一次）

```python
def global_header(ts):
    bnf_present = [1.0 if p in active_levels(ts) else 0.0 for p in PRIORITY_LEVELS]  # 长度5
    min_green_ok = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
    phase_elapsed = ts.time_since_last_phase_change / 100.0   # 归一化
    return bnf_present + [min_green_ok, phase_elapsed]        # 长度 7
```
`active_levels(ts)` 第一版返回固定表的级别集合 `{1,2,3,5}`。

---

## 7. 三个解析器（路口加载时缓存一次）

### 7.1 相位 → 放绿车道集（A 用）

```python
def phase_served_lanes(ts):
    """返回 {k: [lane, ...]}: 第 k 个绿相位放绿(G/g)的 incoming 车道（去重，按 ts.lanes 序）。"""
    raw = ts.sumo.trafficlight.getControlledLanes(ts.id)   # 索引对齐 phase.state
    served = {}
    for k, ph in enumerate(ts.green_phases):
        lanes = []
        seen = set()
        for i, ch in enumerate(ph.state):
            if ch in ('G', 'g') and raw[i] not in seen:
                seen.add(raw[i]); lanes.append(raw[i])
        served[k] = [l for l in ts.lanes if l in seen]      # 保持 ts.lanes 顺序
    return served
```
> 注: 共享车道（既左又直）会出现在多个相位的放绿集里（车道级聚合 → 不分转向 → 不需要转向信息）。这是 A "粗"的来源。

### 7.2 车 → 转向（B 用）

```python
def lane_allowed_dirs(ts, lane):
    """该车道允许的转向集合 {'L','T','R','U'}（来自 SUMO connection direction）。"""
    dirs = set()
    for link in ts.sumo.lane.getLinks(lane):
        d = link[6]                # ⚠️按你的 traci 版本核对 direction 的 index
        dirs.add({'l':'L','L':'L','r':'R','R':'R','t':'U','s':'T'}.get(d, 'T'))
    return dirs

def vehicle_movement(ts, vid, lane):
    """车 vid 在 lane 上的转向: 用 route 的下一条 edge 匹配 link 的 toLane 所属 edge。"""
    route = ts.sumo.vehicle.getRoute(vid)
    idx = ts.sumo.vehicle.getRouteIndex(vid)
    if idx + 1 >= len(route):
        return 'T'                 # 末段，默认直行
    next_edge = route[idx + 1]
    for link in ts.sumo.lane.getLinks(lane):
        to_lane = link[0]
        if to_lane.rsplit('_', 1)[0] == next_edge:
            d = link[6]
            return {'l':'L','L':'L','r':'R','R':'R','t':'U','s':'T'}.get(d, 'T')
    return 'T'
```
转向槽 = `{N,E,S,W} × {L,T,R}` = 12（U-turn 第一版并入 L 或单列，按 net 实际）。进口方向 N/E/S/W 由 §7.3 的 approach_angle 离散化得到。

### 7.3 车道几何 / approach_angle（Transformer 用）

```python
import math
def approach_angle(ts, lane):
    shape = ts.sumo.lane.getShape(lane)           # [(x,y),...] 末点靠近路口
    (x0,y0), (x1,y1) = shape[-2], shape[-1]
    return math.atan2(y1 - y0, x1 - x0)           # 行驶方位角(弧度)

def lane_static_features(ts, lane):
    length = ts.lanes_length[lane]
    cap = length / (ts.MIN_GAP + 5.0)             # 5m 近似车长，或用 getLastStepLength
    dirs = lane_allowed_dirs(ts, lane)
    return [
        length / 200.0,                           # 归一化长度
        approach_angle(ts, lane),                 # 连续角度（替代离散 N/E/S/W）
        cap / 50.0,                               # 归一化容量
        1.0 if 'L' in dirs else 0.0,
        1.0 if 'T' in dirs else 0.0,
        1.0 if 'R' in dirs else 0.0,
        1.0 if lane in ts.always_green_lanes else 0.0,
    ]                                             # 长度 7
```

---

## 8. 三种完整 state（用同一份快照展开）

**快照设定**（聚焦 N/S 进口，W/E 当前绿、残余空；NS 红灯，车全排队 → queue=count）:

| 车 | 车道 | 转向 | 车型→优先级 | awt |
|---|---|---|---|---|
| v1 | n_t_0(左+直共享) | 左 | car→1 | 20 |
| v2 | n_t_0 | 直 | car→1 | 15 |
| v3 | n_t_0 | 直 | amb→5 | 12 |
| v4 | n_t_1(直) | 直 | car→1 | 30 |
| v5 | n_t_1 | 直 | car→1 | 25 |
| v6 | s_t_0(直) | 直 | car→1 | 18 |
| v7 | s_t_0 | 直 | bus→3 | 16 |
| v8 | s_t_1(左) | 左 | car→1 | 22 |

当前相位 = WE直，已绿 15s，min_green 满足。
全局头(7) = `[1,1,1,0,1, 1, 15]`（展示用 phase_elapsed 原值 15；训练时 /100 → 0.15）。
φ 每级布局 `[count, queue, mean_awt, max_awt]`（下方用**原始值**展示，训练时归一化）。

### 8.A 方案 A — 相位级（K 行，每行 22 维）

按相位放绿车道聚合（车道级，不分转向）。

```
header(7) = [1,1,1,0,1, 1, 15]

NS直 [is_cur=0, 0] + φ(放绿车道 n_t_0,n_t_1,s_t_0 → v1..v7):
  p1[5,5,21.6,30] p2[0,0,0,0] p3[1,1,16,16] p4[0,0,0,0] p5[1,1,12,12]
NS左 [0, 0] + φ(放绿车道 n_t_0,s_t_1 → v1,v2,v3,v8):
  p1[3,3,19,22]  p2[0,0,0,0] p3[0,0,0,0]   p4[0,0,0,0] p5[1,1,12,12]
WE直 [is_cur=1, 15] + φ(空): 全 0
WE左 [0, 0] + φ(空): 全 0
```
`dim_A = 7 + K×(2+20)`，K=4 → **95**。
> 共享道 n_t_0 整条进 NS直 和 NS左 → v3 在两行都出现。去掉 servable/blocked 后，"v3 在 NS直 走不了、换全绿才走"state 看不到，agent 自学。

### 8.B 方案 B — 转向级（12 槽，每槽 22 维，空槽 mask）

按车的转向聚合（需转向信息）。非空槽:

```
header(7) = [1,1,1,0,1, 1, 15]
N-L [is_green=0, sec] + φ(v1):           p1[1,1,20,20]
N-T [0, sec] + φ(v2,v3,v4,v5):           p1[3,3,23.3,30]  p5[1,1,12,12]
S-T [0, sec] + φ(v6,v7):                 p1[1,1,18,18]   p3[1,1,16,16]
S-L [0, sec] + φ(v8):                    p1[1,1,22,22]
N-R,E-L,E-T,E-R,S-R,W-L,W-T,W-R: φ=0 + mask=1
```
`dim_B = 7 + 12×(2+20)` = **271**。
> 转向分得最细，但看不到"放 N-L → v1 走 → v2/v3 解锁"的跨槽非线性，agent 自学。

### 8.T Transformer — 车道 token（每 token 29 维 + CLS）

`token = [静态几何7 | is_green2 | φ20]`。非空 token:

```
CLS = 可学习向量(29维参数)
n_t_0: 静态[len,angle_N,cap, L=1,T=1,R=0, ag=0] | [0,sec] | φ(v1,v2,v3): p1[2,2,17.5,20] p5[1,1,12,12]
n_t_1: 静态[..,L=0,T=1,R=0, ag=0]               | [0,sec] | φ(v4,v5):    p1[2,2,27.5,30]
s_t_0: 静态[..,L=0,T=1,R=0, ag=0]               | [0,sec] | φ(v6,v7):    p1[1,1,18,18] p3[1,1,16,16]
s_t_1: 静态[..,L=1,T=0,R=0, ag=0]               | [0,sec] | φ(v8):       p1[1,1,22,22]
其余车道: φ=0 token（仍参与注意力）
```
`state_T = (N_lanes, 29) 矩阵 + CLS`。
> 唯一保留车道身份: n_t_0 token 同时带 v1(左,p1)/v2(直,p1)/v3(直,p5)，注意力可建模"同车道左转挡直行"。

---

## 9. 网络主体

### 9.A A-flat（推荐 A 起步，drop-in 你现有 DQN）
```
state_A(95) → MLP[256,256] → K 个 Q 值（每相位一个）
```
输入维度绑死 K → 不跨路口迁移（这是它 vs Transformer 的代价）。
可选 A-FRAP 变体: 每相位行 → 共享 phase-MLP → 1 分，argmax（参数与 K 无关）。

### 9.B B（FRAP 简化版）
```
每转向槽(22) → 共享 movement-MLP → e_m (d)
phase_k 模板(从 green_phases 解析: 该相位放绿哪些转向)
phase_score_k = MLP( pool_{m∈phase_k} e_m , phase_meta ) → argmax
```
只做 pool-then-score，不做完整 FRAP 两两竞争（更重，且其主要收益是已 defer 的非线性解堵）。

### 9.T Transformer（推荐配 FRAP 动作头）
```
每 token(29) → lane-MLP(29→128) → prepend CLS → Transformer(2层,4头,FF256)
CLS 输出 = h_int(128)  ← 与 N_lanes 无关
phase_k: phase_token = pool(h_lane: lane∈served(k))
Q_k = MLP(concat(h_int, phase_token)) → argmax
```
参数全 per-token/per-phase → 跨任意 N_lanes/K 共享 → 唯一能 zero-shot 迁移、唯一能建模车道交互。
多路口预留: `h_int` → CoLight GAT(邻居 attention) → `h_int'`，不返工。
Fallback: Transformer 训不稳 → DeepSets(φ→sum→ρ)。

---

## 10. 两种动作头（任意 state 可配）

| | 固定相位 + mask | FRAP phase scorer |
|---|---|---|
| 输出 | 固定 N 个相位 Q，不存在的 mask | 每候选相位独立打分 |
| 相位数 | 绑死 | 可变 |
| 跨路口迁移 | ❌ | ✅ |
| 推荐搭配 | A-flat / B | Transformer |
| 动作语义 | 都是 `Discrete(num_green_phases)`，选绿相位 | 同 |

动作契约统一 → 两种动作头与三种 state 可自由组合；min/max-green 掩码沿用 `set_next_phase` 现有逻辑。

---

## 11. 要新增的 ObservationFunction 子类

加到 `sumo_rl/environment/observations.py`，签名与现有类一致（`__call__` 返回 np.ndarray，`observation_space` 返回 `spaces.Box`）:

```python
class PriorityPhaseObservationFunction(ObservationFunction):    # 方案 A
    # __call__: header(7) + 每相位 [is_cur, sec] + φ(phase_served_lanes[k])
    # observation_space dim = 7 + num_green_phases * 22

class PriorityMovementObservationFunction(ObservationFunction): # 方案 B
    # __call__: header(7) + 12 槽 [is_green, sec] + φ(该转向车) + mask
    # observation_space dim = 7 + 12 * 22  (返回 obs 与 mask)

class PriorityLaneTokenObservationFunction(ObservationFunction): # Transformer
    # __call__: 返回 (N_lanes, 29) 矩阵 (+ 静态几何缓存)；CLS 在网络里
    # observation_space: Box(shape=(max_lanes, 29)) 或变长 → 网络处理 padding+mask
```

三者复用同一 `phi()` 与 `global_header()`，只换分组。Transformer 的变长用 padding + attention mask（或每路口固定 N）。

---

## 12. A/B/T 对比实验协议

- **控制变量**: 同 reward（保留 BNF 优先级加权 `-Σ p×mean_awt - β·switch`，p_max 用 max）、同 demand、同 min/max-green、同探索、同评估种子（注意 memory: eval 种子不要污染训练）。
- **唯一变量**: state 组织 (A-flat / B / Transformer)。
- **指标**（`EpisodeMetricsCollector`，per-type 重点看 ambulance/bus）:
  - system avg_wait / throughput / completion_rate
  - **ambulance avg_wait & max_wait**（优先级核心诉求）
  - bus avg_wait
  - 切换频率（β 项）
- **消融**: (a) 5 优先级槽 vs 4 车型槽; (b) 加/不加 max_awt; (c) Phase-2: 加反应式甲 blocked-bit。
- **结论轴**: action 对齐 / 车道交互 / 跨路口参数共享（见 §0 三方对比表）。

---

## 13. 数据依赖与待解决问题

1. **BNF priority "6" 越界**（policy17/22）: 落动态 BNF 前修；第一版固定表绕过。
2. **车辆 context 属性缺失**: 第一版固定 `type→priority` 表绕过；动态 BNF 需在 route 生成时给车 task/location/time/state 属性。
3. **归一化口径**: count/queue ÷ capacity；awt ÷ 100。需对照现有 reward 量纲，避免 obs/reward 尺度打架。
4. **`getLinks` direction index**: §7.2 伪代码的 `link[6]` 按本机 traci 版本核对。
5. **共享车道在 A 的重复计入**: 已知设计取舍（车道级聚合），文档化即可，不修。

---

## 14. 实施 checklist

**基础设施**
- [ ] `priority_map.py`: `TYPE_TO_PRIORITY` + `vehicle_priority()` + `active_levels()`
- [ ] `phi()` 优先级分桶特征器（带 normalize）
- [ ] `global_header()`
- [ ] `phase_served_lanes()` / `vehicle_movement()` / `lane_static_features()` 三解析器（缓存）

**观测类**
- [ ] `PriorityPhaseObservationFunction` (A) + observation_space
- [ ] `PriorityMovementObservationFunction` (B) + mask
- [ ] `PriorityLaneTokenObservationFunction` (Transformer) + padding/mask

**网络**
- [ ] A-flat MLP head（drop-in 现有 DQN）
- [ ] B movement-embed → phase pool-score
- [ ] Transformer encoder(2层4头) + CLS + FRAP phase scorer
- [ ] (可选) FRAP 动作头通用化；(预留) CoLight GAT

**实验**
- [ ] 三种 state 单路口训练脚本（控制变量）
- [ ] 评估报表: system + ambulance/bus per-type
- [ ] 消融: 5槽 vs 4车型 / max_awt / 甲 blocked-bit

---

## 15. 未来扩展（非本期）

- **反应式甲 blocked-bit**: 绿+有车+队首没走 → 标记，带 staleness；φ 每级 +1 位（预留）。salvage A 的"换全绿解堵"可见性。
- **动态 BNF**: `vehicle_priority` 换成真 BNF + 车辆 context → agent/φ/网络不变 → zero-shot 不变性实验（论文核心）。
- **多路口 CoLight GAT**: `h_int` 上挂邻居 attention，邻接图从 net.xml 自动推导。
- **预测式乙 servable/blocked**: 需转向+次序，作为 oracle 上界做"甲 vs 乙"消融。
