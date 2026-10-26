"""VisitWaitTracker (2026-10-26, 任务 4): 逐秒的本次访问等待统计器 + 环境钩子。
键 (路口, 车辆, 进口边); 每仿真秒在 simulationStep 之后更新: 新出现建条目, SUMO 判定等待 (getWaitingTime>0) 的加一个步长,
不在场的删除。对账恒等式: waiting_time_memory 远大于集长时, W == 当前累计等待 − 进入前一秒的累计等待。需要 SUMO。
运行: python -m pytest experiments/tests/test_visit_wait.py -q"""
import os, sys, functools, yaml, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")
pytestmark = pytest.mark.skipif(not os.path.exists(CFG274), reason="1x1 配置缺失")


def _env(wait_tracker=True, extra=None, route_file=None, num_seconds=400):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from exp_params import obs_kwargs_from_cfg
    cfg = yaml.safe_load(open(CFG274)); os.chdir(os.path.join(_REPO, "experiments"))
    kw = obs_kwargs_from_cfg(cfg)
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=route_file or cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None,
                          use_gui=False, num_seconds=num_seconds, min_green=5, single_agent=False, yellow_time=2, delta_time=5,
                          reward_fn="avg-waiting-time", observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw),
                          sumo_seed=0, sumo_warnings=False, waiting_time_memory=100000, additional_sumo_cmd=extra, wait_tracker=wait_tracker)
    env.reset(0); return env, env.ts_ids[0]


class _Ref:
    """独立参照: 用 SUMO 的累计等待做对账。进入前一秒的累计等待作为起点, W_ref = acc − 起点。"""
    def __init__(self, env, t):
        self.env, self.t, self.sumo = env, t, env.sumo; self.lanes = env.traffic_signals[t].lanes
        self.acc_prev = {}; self.entry = {}; self.checked = 0; self.max_diff = 0.0; self.frac_half = 0
    def record(self):
        sumo = self.sumo; tr = self.env.visit_tracker; present = set()
        for lane in self.lanes:
            edge = lane.rsplit("_", 1)[0]
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                k = (self.t, vid, edge); present.add(k); acc = sumo.vehicle.getAccumulatedWaitingTime(vid)
                if k not in self.entry:
                    self.entry[k] = self.acc_prev.get(vid, 0.0)
                w = tr.get(self.t, vid, lane); ref = acc - self.entry[k]
                self.max_diff = max(self.max_diff, abs(w - ref)); self.checked += 1
                if abs(w - round(w)) > 1e-9: self.frac_half += 1
        for k in [k for k in self.entry if k not in present]:
            del self.entry[k]
        for vid in sumo.vehicle.getIDList():
            self.acc_prev[vid] = sumo.vehicle.getAccumulatedWaitingTime(vid)


def _hook(env, ref):
    orig = env._sumo_step
    def hooked():
        orig(); ref.record()
    env._sumo_step = hooked


def _drive(env, t, steps):
    for i in range(steps):
        env.step(action={t: (i // 3) % 4})


def test_env_creates_tracker_only_when_enabled():
    from sumo_rl.environment.visit_wait import VisitWaitTracker
    env, t = _env(wait_tracker=False); assert env.visit_tracker is None; env.close()
    env, t = _env(wait_tracker=True); assert isinstance(env.visit_tracker, VisitWaitTracker)
    assert env.visit_tracker.dt == 1.0; env.close()


def test_W_matches_sumo_accumulated_wait_increments():
    env, t = _env(); ref = _Ref(env, t); _hook(env, ref); _drive(env, t, 70); env.close()
    assert ref.checked > 2000 and ref.max_diff == 0.0


def test_step_twice_at_same_time_does_not_double_count():
    env, t = _env(); _drive(env, t, 30); tr = env.visit_tracker
    before = dict(tr.W); assert before
    tr.step(); assert tr.W == before; env.close()


def test_reset_clears_entries():
    env, t = _env(); _drive(env, t, 30); assert env.visit_tracker.W
    env.reset(0); assert env.visit_tracker.W == {} and env.visit_tracker.missing == 0; env.close()


def test_strict_get_raises_and_lenient_counts_missing():
    from sumo_rl.environment.visit_wait import VisitWaitTracker
    env, t = _env(); lanes = {t: env.traffic_signals[t].lanes}
    strict = VisitWaitTracker(env.sumo, lanes, strict=True)
    with pytest.raises(KeyError):
        strict.get(t, "no_such_vehicle", "w_t_0")
    lenient = VisitWaitTracker(env.sumo, lanes)
    assert lenient.get(t, "no_such_vehicle", "w_t_0") == 0.0 and lenient.missing == 1; env.close()


def test_half_second_step_length_accumulates_half_seconds():
    env, t = _env(extra="--step-length 0.5"); assert env.visit_tracker.dt == 0.5
    ref = _Ref(env, t); _hook(env, ref); _drive(env, t, 40); env.close()
    assert ref.max_diff == 0.0 and ref.frac_half > 0


def test_removed_vehicle_closes_visit():
    env, t = _env(); _drive(env, t, 30); tr = env.visit_tracker
    key = next(iter(tr.W)); vid = key[1]
    env.sumo.vehicle.remove(vid); env.step(action={t: 0})
    assert key not in tr.W; env.close()


def test_scheduled_bus_stop_is_not_waiting(tmp_path):
    add = tmp_path / "stop.add.xml"; rou = tmp_path / "bus.rou.xml"
    add.write_text('<additional><busStop id="bs" lane="w_t_0" startPos="100" endPos="130"/></additional>')
    rou.write_text('<routes><vType id="bus"/><vehicle id="busX" type="bus" depart="0" departLane="0"><route edges="w_t t_e"/>'
                   '<stop busStop="bs" duration="60"/></vehicle></routes>')
    env, t = _env(extra=f"--additional-files {add}", route_file=str(rou)); tr = env.visit_tracker; sumo = env.sumo
    stopped_seconds = 0; w_during = []; acc_during = []
    orig = env._sumo_step
    def hooked():
        nonlocal stopped_seconds
        orig()
        if "busX" in sumo.vehicle.getIDList() and sumo.vehicle.isStopped("busX"):
            stopped_seconds += 1; w_during.append(tr.get(t, "busX", "w_t_0")); acc_during.append(sumo.vehicle.getAccumulatedWaitingTime("busX"))
    env._sumo_step = hooked
    _drive(env, t, 30); env.close()
    # SUMO 在刹停进站的那一秒记 1 秒等待, 停靠期间不再累计; 统计器跟 SUMO 走: 停靠期间 W 不增长, 且与 SUMO 累计等待相等
    assert stopped_seconds >= 50
    assert max(w_during) == min(w_during) and max(acc_during) == min(acc_during)
    assert w_during == acc_during and max(w_during) <= tr.dt
