"""逐秒的本次访问等待统计器 (2026-10-26, 状态/奖励修正实施计划任务 4)。

背景: SUMO 只提供按车、跨车道、滚动窗 (--waiting-time-memory) 内的累计等待 getAccumulatedWaitingTime, 没有
"在当前进口道上等了多久"。观测原来用 "当前累计 − 第一次被观测到时的累计" 近似 (起点最多晚 4 秒, 重入沿用旧基线),
奖励原来用 "当前累计 − 其他 RL 车道的记录" (上游非 RL 路口的等待会整段记给第一条 RL 车道)。两者都受滚动窗过期影响。

本统计器不用累计等待: 每个仿真秒在 simulationStep 之后扫全部 RL 进口道 (ts.lanes, 与观测和奖励的范围一致),
键 (路口, 车辆, 进口边):
    新出现 → W = I·Δ;  在场 → W += I·Δ;  不在场 → 删除
其中 I = 1 当且仅当 SUMO 判定本步在等待 (getWaitingTime > 0, 计划停车由 SUMO 自行排除), Δ = simulation.getDeltaT()。
按进口边记键: 同一进口内换道不清零; 离开再进入从 0 起; 路口内部车道与非 RL 路段的等待不计给任何路口。
state 与 reward 在 visit 口径下只读 get(), 不再各自记账。"""


class VisitWaitTracker:
    def __init__(self, sumo, ts_lanes: dict, strict: bool = False):
        """sumo: 连接句柄; ts_lanes: {路口: [进口车道]} (用 ts.lanes); strict: get() 缺条目时抛 KeyError (验证阶段开)。"""
        self.sumo = sumo
        self.strict = bool(strict)
        # 静态表: 车道 → (路口, 进口边)。进口边 = 车道 id 去掉末尾 "_<index>"。
        self._lane_key = {}
        for ts_id, lanes in ts_lanes.items():
            for lane in lanes:
                self._lane_key[lane] = (ts_id, lane.rsplit("_", 1)[0])
        self.dt = float(sumo.simulation.getDeltaT())     # 步长只读一次 (整个仿真固定)
        self.W: dict = {}            # (路口, 车辆, 进口边) -> 本次访问等待秒数
        self._prev: set = set()      # 上一秒在场的键
        self._last_time = None       # 已处理的仿真时刻 (同一时刻重复调用直接返回)
        self.missing = 0             # 非严格模式下 get() 缺条目的次数 (预期恒为 0)

    def step(self) -> None:
        now = float(self.sumo.simulation.getTime())
        if self._last_time is not None and now <= self._last_time:
            return
        self._last_time = now
        W, dt = self.W, self.dt
        lane_vehicles = self.sumo.lane.getLastStepVehicleIDs
        waiting = self.sumo.vehicle.getWaitingTime
        present = set()
        for lane, (ts_id, edge) in self._lane_key.items():
            for vid in lane_vehicles(lane):
                k = (ts_id, vid, edge)
                present.add(k)
                W[k] = W.get(k, 0.0) + (dt if waiting(vid) > 0 else 0.0)
        for k in self._prev - present:
            W.pop(k, None)
        self._prev = present

    def get(self, ts_id: str, vid: str, lane: str) -> float:
        k = (ts_id, vid, lane.rsplit("_", 1)[0])
        w = self.W.get(k)
        if w is not None:
            return w
        if self.strict:
            raise KeyError(k)
        self.missing += 1
        return 0.0

    def reset(self) -> None:
        self.W.clear()
        self._prev = set()
        self._last_time = None
        self.missing = 0
