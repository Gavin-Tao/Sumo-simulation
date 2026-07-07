probes_priority_eval_2026-07 — 复算脚本索引
(配套文档: experiments/analysis/BASELINE_VERDICT_AND_ROADMAP_2026-07-07.txt §8;
 均从 repo 根或 experiments/ 下运行, SUMO_RL_LIBSUMO=1, 详见各脚本头注释)

wandb 拉取 / 窗口统计
  pull_wandb.py            Dublin 207/211/215 eval 序列拉取
  pull_1x1.py              1x1 217/219/220 全量 history 拉取
  window_stats.py          末窗 IQR 剔异常 + 逐类 per-visit + 反转检验

1x1 批次 (BASELINE §2, §6.3, §6.5, §6.5a)
  probe_1x1.py             统一探针: {217|219|220} × {base|blind|w421b|swap351}
                           — 指标+指纹+amb停车分类; blind=obs致盲,
                           w421b=4-2-1经由桶, swap351=3-5-1桶交换
  probe_219_sweep.py       multihead 决策权重扫描 (flat/default/amb15/amb50/倒置)
                           注意: 改权重必须同步 agent._w_t (缓存张量)
  probe_219_sweep_gated.py       presence 门控合成版扫描
  probe_219_sweep_recontract.py  换约点 4-2-1 / 3-5-1 (裸)
  probe_219_sweep_gated_recontract.py  同上 (门控)
  smoke_exp220.py          exp220 无 wandb 冒烟 (mask/维度/loss 断言)

Dublin 批次 (EXP211_RESULTS_ANALYSIS §5a; BASELINE §3)
  probe_guard_metrics.py   {207|211} × {base|guard}: 空进口道→保持 外挂前后
  probe_guard207.py        guard 初版 (含 window_closed 计数)
  probe_amb_mech.py        amb 逐车停车机制分类 (denied/green_blocked/
                           green_head/mid_queue) + 拒绝时绿灯方向占用
  probe_ambblind.py        obs 致盲 (amb→l1, 仅 obs 表)
  probe_v350_base.py / probe_v350_blind.py   211 ckpt_ep350 谷底配对
  probe_215_emptyswitch.py 215 空进口道切相位计数 (结果: 4843 空步 0 切换)
  probe_behavior.py        行为指纹 (dqn8std|frap|moe): 换相率/保持分布
