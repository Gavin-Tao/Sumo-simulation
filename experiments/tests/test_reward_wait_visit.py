"""reward wait_basis=visit (2026-10-26, 任务 6): priority-avg-waiting 与 avg-waiting-time 两个奖励改读逐秒访问统计器。
聚合公式不变 (按等级均值 × 权重 / 100 取负; 普通版全体均值 / 100 取负), 只换等待秒数来源; 不读写 env.vehicles。
legacy 默认逐位不变 (有无统计器相同)。需要 SUMO。运行: python -m pytest experiments/tests/test_reward_wait_visit.py -q"""
import os, sys, functools, yaml, numpy as np, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")
pytestmark = pytest.mark.skipif(not os.path.exists(CFG274), reason="1x1 配置缺失")
TABLE = {"ambulance": 5, "bus": 4, "car": 1}


def _env(reward_fn, wait_tracker, info=True, num_seconds=400):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    cfg = yaml.safe_load(open(CFG274)); os.chdir(os.path.join(_REPO, "experiments"))
    kw = dict(fields=("count", "queue", "mean_awt"), phase_state="perphase", priority_source=TABLE)
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                          num_seconds=num_seconds, min_green=5, single_agent=False, yellow_time=2, delta_time=5, reward_fn=reward_fn,
                          observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw), sumo_seed=0, sumo_warnings=False,
                          waiting_time_memory=100000, wait_tracker=wait_tracker, add_system_info=info, add_per_agent_info=info)
    env.reset(0); return env, env.ts_ids[0]


def _ref_priority(env, t):
    sumo, tr = env.sumo, env.visit_tracker; tot, cnt = {}, {}
    for lane in env.traffic_signals[t].lanes:
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            w = float(TABLE[sumo.vehicle.getTypeID(vid)]); tot[w] = tot.get(w, 0.0) + tr.get(t, vid, lane); cnt[w] = cnt.get(w, 0) + 1
    return -sum(w * tot[w] / cnt[w] for w in tot) / 100.0


def _ref_plain(env, t):
    sumo, tr = env.sumo, env.visit_tracker; s = 0.0; n = 0
    for lane in env.traffic_signals[t].lanes:
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            s += tr.get(t, vid, lane); n += 1
    return -(s / n) / 100.0 if n else 0.0


def test_visit_priority_reward_equals_formula():
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    env, t = _env(make_priority_avg_waiting_reward(TABLE, wait_basis="visit"), True); nonzero = 0
    for i in range(60):
        _, r, _, _ = env.step(action={t: (i // 3) % 4}); ref = _ref_priority(env, t)
        assert abs(r[t] - ref) < 1e-12; nonzero += r[t] != 0.0
    env.close(); assert nonzero > 20


def test_visit_plain_reward_equals_formula_and_empty_is_zero():
    from sumo_rl.environment.rewards import make_avg_waiting_reward
    fn = make_avg_waiting_reward("visit"); env, t = _env(fn, True)
    assert fn(env.traffic_signals[t]) == 0.0                 # reset 后进口道无车 → 0
    for i in range(60):
        _, r, _, _ = env.step(action={t: (i // 3) % 4}); assert abs(r[t] - _ref_plain(env, t)) < 1e-12
    env.close()


def test_legacy_factories_keep_old_callables_and_ignore_tracker():
    from sumo_rl.environment.rewards import make_avg_waiting_reward, make_priority_avg_waiting_reward, _avg_waiting_reward
    assert make_avg_waiting_reward() is _avg_waiting_reward and make_avg_waiting_reward("legacy") is _avg_waiting_reward
    seqs = []
    for wt in (False, True):
        env, t = _env(make_priority_avg_waiting_reward(TABLE), wt, num_seconds=300); seq = []
        for i in range(50):
            _, r, _, _ = env.step(action={t: (i // 3) % 4}); seq.append(r[t])
        env.close(); seqs.append(seq)
    assert seqs[0] == seqs[1]


def test_visit_reward_does_not_write_env_vehicles():
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    env, t = _env(make_priority_avg_waiting_reward(TABLE, "visit"), True, info=False)
    for i in range(30):
        env.step(action={t: (i // 3) % 4})
    assert env.vehicles == {}; env.close()
    env, t = _env(make_priority_avg_waiting_reward(TABLE), False, info=False)
    for i in range(30):
        env.step(action={t: (i // 3) % 4})
    assert env.vehicles; env.close()                          # legacy 仍写旧记账


def test_scale_and_floor_wrappers_compose_with_visit():
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    base = make_priority_avg_waiting_reward(TABLE, "visit")
    def wrapped(ts, _f=base):
        return max(_f(ts) * 2.0, -0.05)
    env, t = _env(wrapped, True); clipped = 0
    for i in range(60):
        _, r, _, _ = env.step(action={t: (i // 3) % 4}); ref = max(_ref_priority(env, t) * 2.0, -0.05)
        assert abs(r[t] - ref) < 1e-12; clipped += r[t] == -0.05
    env.close(); assert clipped > 0


def test_bad_basis_rejected():
    from sumo_rl.environment.rewards import make_avg_waiting_reward, make_priority_avg_waiting_reward
    with pytest.raises(ValueError):
        make_avg_waiting_reward("bogus")
    with pytest.raises(ValueError):
        make_priority_avg_waiting_reward(TABLE, "bogus")
