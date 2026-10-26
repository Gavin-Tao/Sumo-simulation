"""eval_collector_1s (2026-10-26, 用户令 "只改这个"): 训练中评估的指标收集器可按仿真秒采样。
PerSecondCollector 在评估期间把收集器挂到 env._sumo_step: 每个仿真步推进之前采一次 (收集器要求在 simulationStep 之前调用,
与 eval_pervehicle --collector-1s 同一口径), 退出后恢复, 训练步不再采样。默认关, 旧配置逐位不变。需要 SUMO。
运行: python -m pytest experiments/tests/test_eval_collector_1s.py -q"""
import os, sys, functools, yaml, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")
pytestmark = pytest.mark.skipif(not os.path.exists(CFG274), reason="1x1 配置缺失")


def _env(num_seconds=400):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from exp_params import obs_kwargs_from_cfg
    cfg = yaml.safe_load(open(CFG274)); os.chdir(os.path.join(_REPO, "experiments"))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                          num_seconds=num_seconds, min_green=5, single_agent=False, yellow_time=2, delta_time=5, reward_fn="avg-waiting-time",
                          observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **obs_kwargs_from_cfg(cfg)),
                          sumo_seed=0, sumo_warnings=False)
    env.reset(0); return env, env.ts_ids[0]


class _Stub:
    def __init__(self):
        self.times = []
    def collect_step(self, sumo):
        self.times.append(float(sumo.simulation.getTime()))


def test_hook_samples_once_per_simulation_second_before_the_step_and_restores():
    from eval_hooks import PerSecondCollector
    env, t = _env(); stub = _Stub(); orig = env._sumo_step
    with PerSecondCollector(env, stub) as psc:
        for i in range(10):
            env.step(action={t: (i // 3) % 4})
        assert psc.calls == 50
    assert stub.times == [float(x) for x in range(50)]          # 每秒一次, 且在该秒推进之前 (第一次在 0 秒)
    assert env._sumo_step == orig and "_sumo_step" not in env.__dict__   # 退出后恢复为类方法 (绑定方法对象每次新建, 用 == 比)
    env.step(action={t: 0}); assert len(stub.times) == 50       # 训练步不再采样
    env.close()


def test_real_collector_per_second_matches_independent_stopped_count():
    from eval_hooks import PerSecondCollector
    from sumo_rl.environment.metrics import EpisodeMetricsCollector
    env, t = _env(); sumo = env.sumo
    ts_lane_map = {t: env.traffic_signals[t].signal_controlled_lanes}; always_green = set(env.traffic_signals[t].always_green_lanes)
    mc = EpisodeMetricsCollector(ts_lane_map, delta_time=float(sumo.simulation.getDeltaT()), excluded_lanes=always_green)
    ref = {"stopped_veh_sec": 0}
    lanes = [l for l in ts_lane_map[t] if l not in always_green]
    class Counter:                                              # 独立参照: 每秒数受控进口道上速度 < 0.1 的车辆
        def collect_step(self, s):
            for l in lanes:
                for vid in s.lane.getLastStepVehicleIDs(l):
                    if s.vehicle.getSpeed(vid) < 0.1:
                        ref["stopped_veh_sec"] += 1
            mc.collect_step(s)
    with PerSecondCollector(env, Counter()):
        for i in range(60):
            env.step(action={t: (i // 3) % 4})
    Counter().collect_step(sumo); mc.finalize(sumo)             # 末尾补采一次 (Counter 内部已调 mc.collect_step, 不能再单独调一次)
    recs = dict(getattr(mc, "_finalized", {})); recs.update(getattr(mc, "_active", {}))
    total = sum(sum(r.get("per_ts_stopped_sec", {}).values()) for r in recs.values())
    assert ref["stopped_veh_sec"] > 100 and total == ref["stopped_veh_sec"]
    env.close()
