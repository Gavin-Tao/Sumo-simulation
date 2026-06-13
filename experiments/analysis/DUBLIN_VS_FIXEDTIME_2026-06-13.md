================================================================================
  Dublin: exp208(γ.95) vs 固定配时 全指标对比 + 优先级倒挂判定  (2026-06-13)
  exp208=尾段20平均(真收敛); fixed-time=3seed; 同受控车道口径; wandb命名
================================================================================

## 核心结论
1. ★exp208 全面优于固定配时: 几乎所有指标改善 +6%~+107% (仅 stop_events_per_visit
   的 car 项 -29%, 即 car 更频繁地短停, 但每次停更短 → 总停驶时间仍大降)。
2. ★优先级倒挂: 固定配时本身【严重倒挂】(amb 22.33 > car 18.35 停驶/visit, 因 amb
   路线长穿越多路口、固定配时不优待 → 反而最差); exp208 在所有 fair 口径【完全消除倒挂】
   并建立正确阶梯 amb<bus<car。这是优先级机制最有力的证据。
3. amb 改善最大 (+96%停驶/visit, +107%速度) = 优先级机制精准作用于最高优先级。
4. amb 样本 n≈3/ep, 尾段std仍大(±0.7~3.0), 阶梯方向稳健但绝对值需≥50seed精确化。
# Dublin: fixed-time vs exp208(γ.95) 对比 + 优先级倒挂判定

注: exp208用尾段20平均(真收敛值); fixed-time 3seed; 同 EpisodeMetricsCollector 受控车道口径.
改善: 速度越高越好,其余(停驶/停次)越低越好,completion越高越好.

## per_visit + xts 指标 (重点)

**`eval_system/<class>/avg_stopped_time_per_visit`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 18.07 | 4.76±1.0 | +74% |
| car | 18.35 | 4.95±1.1 | +73% |
| bus | 12.98 | 1.31±0.2 | +90% |
| ambulance | 22.33 | 0.86±0.7 | +96% |

**`eval_system/<class>/avg_stop_events_per_visit`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 0.46 | 0.58±0.1 | -25% |
| car | 0.47 | 0.60±0.1 | -29% |
| bus | 0.42 | 0.18±0.0 | +57% |
| ambulance | 0.50 | 0.14±0.1 | +73% |

**`eval_system/<class>/xts_avg_speed`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 2.84 | 5.44±0.1 | +91% |
| car | 2.82 | 5.37±0.1 | +90% |
| bus | 3.57 | 6.33±0.1 | +77% |
| ambulance | 3.12 | 6.46±0.4 | +107% |

**`eval_system/<class>/xts_avg_stopped_time`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 22.99 | 3.04±0.4 | +87% |
| car | 23.00 | 3.12±0.4 | +86% |
| bus | 19.45 | 1.76±0.3 | +91% |
| ambulance | 27.60 | 0.94±0.8 | +97% |

**`eval_system/<class>/xts_avg_stop_events`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 0.57 | 0.38±0.0 | +33% |
| car | 0.57 | 0.40±0.0 | +31% |
| bus | 0.56 | 0.20±0.0 | +64% |
| ambulance | 0.67 | 0.14±0.1 | +78% |

## 基础指标

**`eval_system/<class>/avg_speed`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 4.66 | 5.04±0.1 | +8% |
| car | 4.67 | 4.96±0.1 | +6% |
| bus | 4.64 | 6.49±0.1 | +40% |
| ambulance | 4.00 | 6.32±0.4 | +58% |

**`eval_system/<class>/avg_stopped_time`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 33.92 | 6.97±1.1 | +79% |
| car | 33.15 | 7.13±1.2 | +78% |
| bus | 47.66 | 4.13±0.6 | +91% |
| ambulance | 86.25 | 3.44±3.0 | +96% |

**`eval_system/<class>/completion_rate`**

| class | fixed-time | exp208(尾段20) | 改善 |
|---|---|---|---|
| all | 0.96 | 0.97±0.0 | +1% |
| car | 0.96 | 0.97±0.0 | +1% |
| bus | 0.93 | 0.96±0.0 | +3% |
| ambulance | 1.00 | 1.00±0.0 | +0% |

## 优先级倒挂判定 (期望 amb<bus<car, 越低越好)

| 指标 | 体系 | amb | bus | car | 结论 |
|---|---|---|---|---|---|
| 停驶/visit★fair | fixed | 22.33 | 12.98 | 18.35 | amb>car严重倒挂! |
| 停驶/visit★fair | exp208 | 0.86 | 1.31 | 4.95 | 无倒挂✓ |
| 停次/visit★fair | fixed | 0.50 | 0.42 | 0.47 | amb>car严重倒挂! |
| 停次/visit★fair | exp208 | 0.14 | 0.18 | 0.60 | 无倒挂✓ |
| xts停驶 | fixed | 27.60 | 19.45 | 23.00 | amb>car严重倒挂! |
| xts停驶 | exp208 | 0.94 | 1.76 | 3.12 | 无倒挂✓ |

速度(越高越好,期望amb≥bus≥car):

| 指标 | 体系 | amb | bus | car | 结论 |
|---|---|---|---|---|---|
| xts速度 | fixed | 3.12 | 3.57 | 2.82 | amb最快✓ |
| xts速度 | exp208 | 6.46 | 6.33 | 5.37 | amb最快✓ |
