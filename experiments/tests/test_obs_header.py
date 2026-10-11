"""obs_header / include_header (2026-10-11, 用户批准): perphase 编码下可关闭头部两维 [min_green_ok, elapsed/100]。
默认开 → 维度与数值逐位不变; 关掉 → 维度减 2, 其余逐位等于默认向量的 [2:]; legacy 编码不允许关。需要 SUMO (1x1 8std 网), 无 SUMO 则跳过。"""
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
              awt_log=True, include_green_elapsed=True); kw.update(obs_kw)
    obs = functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
    return SumoEnvironment(net_file=NET, route_file=ROU, cfg_file=CFG, out_csv_name=None, use_gui=False, num_seconds=200, min_green=5,
                           max_green=50, use_max_green=True, single_agent=False, yellow_time=2, delta_time=5,
                           reward_fn=make_priority_avg_waiting_reward({"ambulance": 5, "bus": 3, "car": 1}), observation_class=obs, sumo_seed=0, sumo_warnings=False)

def _rollout(env, n=6):
    s = env.reset(0); t = list(env.ts_ids)[0]; seq = [np.asarray(s[t], dtype=np.float64)]
    for i in range(n):
        s, _, done, _ = env.step(action={t: (i // 2) % env.traffic_signals[t].num_green_phases}); seq.append(np.asarray(s[t], dtype=np.float64))
        if done["__all__"]: break
    env.close(); return seq

def test_default_header_present_and_unchanged():
    a = _env(); t = list(a.ts_ids)[0] if False else None
    seq = _rollout(a); assert len(seq[0]) == 2 + 12 * 17          # 头 2 + 12 槽 × (is_green + φ15 + green_elapsed)
    b = _env(include_header=True); seq_b = _rollout(b)
    for x, y in zip(seq, seq_b): assert np.array_equal(x, y)       # 显式 True 与默认完全相同

def test_header_off_drops_exactly_two_dims():
    on = _rollout(_env()); off = _rollout(_env(include_header=False))
    assert len(off[0]) == len(on[0]) - 2 == 12 * 17
    for x, y in zip(on, off): assert np.array_equal(x[2:], y)      # 其余逐位相同

def test_header_off_space_matches_vector():
    e = _env(include_header=False); s = e.reset(0); t = list(e.ts_ids)[0]
    assert e.traffic_signals[t].observation_space.shape[0] == len(s[t]) == 12 * 17; e.close()

def test_legacy_rejects_header_off():
    with pytest.raises(ValueError):
        e = _env(phase_state="legacy", include_header=False); e.reset(0); e.close()
