"""obs_fields 可含 mean_awt_log (2026-10-12, 用户令 '线性与对数等待两个都给'): 默认字段逐位不变; 加入后每桶多一维, 值 = log(1+原始平均等待)/log(3601), 与线性 mean_awt (/100) 同时存在。需要 SUMO。"""
import os, sys, functools, numpy as np, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
D = os.path.join(_REPO, "nets/syc/1x1/Eq_350_BC_Wstr_Estr-Wleft_Eleft/Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb")
NET, ROU, CFG = (os.path.join(D, "syc_4phases_8std8.net.xml"), os.path.join(D, "Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb.rou.xml"), os.path.join(D, "Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb.sumocfg"))
pytestmark = pytest.mark.skipif(not os.path.exists(NET), reason="1x1 net missing")
def _roll(fields, n=12):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    kw = dict(fields=fields, phase_state="perphase", priority_source={"ambulance": 5, "bus": 3, "car": 1})
    env = SumoEnvironment(net_file=NET, route_file=ROU, cfg_file=CFG, out_csv_name=None, use_gui=False, num_seconds=200, min_green=5, single_agent=False, yellow_time=2, delta_time=5,
                          reward_fn=make_priority_avg_waiting_reward({"ambulance": 5, "bus": 3, "car": 1}), observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw), sumo_seed=0, sumo_warnings=False)
    s = env.reset(0); t = list(env.ts_ids)[0]; seq = [np.asarray(s[t], float)]
    for i in range(n): s, _, _, _ = env.step(action={t: (i // 3) % 4}); seq.append(np.asarray(s[t], float))
    env.close(); return seq
def test_both_fields():
    base = _roll(("count", "queue", "mean_awt")); both = _roll(("count", "queue", "mean_awt", "mean_awt_log"))
    assert len(both[0]) == len(base[0]) + 12 * 5
    for x, y in zip(base, both):
        assert np.allclose(x[:2], y[:2])
        for sl in range(12):
            bx = x[2 + sl * 16: 2 + sl * 16 + 16]; by = y[2 + sl * 21: 2 + sl * 21 + 21]
            assert bx[0] == by[0]
            for p in range(5):
                assert np.allclose(bx[1 + p * 3: 4 + p * 3], by[1 + p * 4: 4 + p * 4])            # count, queue, 线性 mean_awt 不变
                lin = bx[1 + p * 3 + 2]; lg = by[1 + p * 4 + 3]
                assert np.isclose(lg, np.log1p(lin * 100.0) / np.log1p(3600.0), atol=1e-5)          # 对数版与线性版对应同一原始等待
