"""训练中评估的逐秒采样钩子 (2026-10-26, 用户令: 加开关 eval_collector_1s, 默认关, 只给新臂用; best.pth 选择依据不动)。

原来训练中评估的 EpisodeMetricsCollector 每个决策步 (5 秒) 采一次: 采样那一秒车停着就记 5 秒, 两次采样之间停了又走的不计,
短进口道上几秒就过去的车看不见。eval_pervehicle --collector-1s 用逐秒口径出 best×10 报表, 两边口径不同。
PerSecondCollector 在评估期间把收集器挂到 env._sumo_step, 每个仿真步推进之前采一次 (收集器要求在 simulationStep 之前调用),
退出后恢复原函数, 训练步不受影响。收集器的 delta_time 由调用方按 simulation.getDeltaT() 传入。"""


class PerSecondCollector:
    def __init__(self, env, collector):
        self.env, self.collector = env, collector
        self._orig = None
        self.calls = 0

    def start(self):
        if self._orig is not None:
            return self
        self._had_override = "_sumo_step" in self.env.__dict__     # 之前是否已有实例级覆盖 (如 eval_pervehicle 的钩子)
        orig = self.env._sumo_step

        def hooked():
            self.collector.collect_step(self.env.sumo)   # 本步推进之前采样
            self.calls += 1
            orig()

        self._orig = orig
        self.env._sumo_step = hooked
        return self

    def stop(self):
        if self._orig is not None:
            if self._had_override:
                self.env._sumo_step = self._orig        # 恢复原来的实例级覆盖
            else:
                del self.env.__dict__["_sumo_step"]     # 恢复为类方法本身
            self._orig = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
