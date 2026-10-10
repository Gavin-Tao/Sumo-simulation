"""obs_green_elapsed (2026-10-10, 用户批准): PriorityMovement 每槽追加 "本次连续放行秒数" 特征 (放行中计时, 未放行 0, 上限 600/100)。
默认关 → 维度与数值逐位不变。需要 SUMO (用 1x1 8std 网跑十几步), 无 SUMO 则跳过。"""
import os, sys, functools, numpy as np, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
pytest.importorskip("libsumo")
D = os.path.join(_REPO, "nets/syc/1x1/Eq_350_BC_Wstr_Estr-Wleft_Eleft/Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb")
NET, ROU, CFG = (os.path.join(D, "syc_4phases_8std8.net.xml"), os.path.join(D, "Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb.rou.xml"),
                 os.path.join(D, "Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb.sumocfg"))
pytestmark = pytest.mark.skipif(not os.path.exists(NET), reason="1x1 net missing")

def _env(**obs_kw):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    kw = dict(fields=("count", "queue", "mean_awt"), phase_state="perphase", priority_source={"ambulance": 5, "bus": 3, "car": 1},
              include_since_green=True); kw.update(obs_kw)
    obs = functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
    return SumoEnvironment(net_file=NET, route_file=ROU, cfg_file=CFG, out_csv_name=None, use_gui=False, num_seconds=200, min_green=5,
                           max_green=50, use_max_green=True, single_agent=False, yellow_time=2, delta_time=5,
                           reward_fn=make_priority_avg_waiting_reward({"ambulance": 5, "bus": 3, "car": 1}), observation_class=obs, sumo_seed=0, sumo_warnings=False)

def _layout(env, t):
    of = env.traffic_signals[t].observation_fn; per = 1 + 15 + (1 if of.include_since_green else 0) + (1 if of.include_green_elapsed else 0)
    return of, per

def test_default_off_unchanged_dim_and_values():
    a = _env(); sa = a.reset(0); t = list(a.ts_ids)[0]; dim_off = len(sa[t]); seq_off = [sa[t]]
    for _ in range(4):
        sa, _, _, _ = a.step({t: 0}); seq_off.append(sa[t])
    a.close()
    b = _env(include_green_elapsed=False); sb = b.reset(0); seq_b = [sb[t]]
    for _ in range(4):
        sb, _, _, _ = b.step({t: 0}); seq_b.append(sb[t])
    b.close()
    assert dim_off == 2 + 12 * 17 and all(np.array_equal(x, y) for x, y in zip(seq_off, seq_b))   # 关 = 显式 False = 旧布局 206 维

def test_on_semantics():
    e = _env(include_green_elapsed=True); s = e.reset(0); t = list(e.ts_ids)[0]; of, per = _layout(e, t)
    assert len(s[t]) == 2 + 12 * 18
    def ge(vec, slot): return float(vec[2 + slot * per + per - 1])     # 槽末位 = green_elapsed
    def ig(vec, slot): return float(vec[2 + slot * per])
    # 连续保持相位 0 四步: 放行槽的 green_elapsed 每步 +0.05, 未放行槽恒 0
    prev = None
    for k in range(4):
        s, _, _, _ = e.step({t: 0}); v = s[t]
        for slot in range(12):
            if ig(v, slot) == 1.0:
                assert ge(v, slot) >= 0.0 and (prev is None or ge(v, slot) >= prev[slot]) and ge(v, slot) <= 6.0
            else:
                assert ge(v, slot) == 0.0
        prev = {slot: ge(v, slot) for slot in range(12)}
    served0 = {slot for slot in range(12) if ig(v, slot) == 1.0}; assert served0 and max(prev[sl] for sl in served0) >= 0.15
    # 切到相位 1: 原相位独有的槽归 0, 新放行槽从 0 起
    for _ in range(3): s, _, _, _ = e.step({t: 1})
    v = s[t]; served1 = {slot for slot in range(12) if ig(v, slot) == 1.0}
    assert served1 != served0
    for slot in served0 - served1: assert ge(v, slot) == 0.0
    for slot in served1 - served0: assert 0.0 <= ge(v, slot) <= 0.2
    e.close()
