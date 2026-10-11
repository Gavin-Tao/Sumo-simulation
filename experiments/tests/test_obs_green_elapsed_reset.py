"""obs_green_elapsed_reset / green_elapsed_reset (2026-10-11, 用户批准): True 时相位切换把新相位成员的 green_elapsed 重新起算 (相位级语义)。
默认 False → 与原 green_elapsed 逐位相同; True 时切换后所有绿槽值相等且 = 头部已持续时间 − 5 s; 无切换时两种语义相同。需要 SUMO。"""
import os, sys, functools, numpy as np, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
D = os.path.join(_REPO, "nets/syc/1x1/Eq_350_BC_Wstr_Estr-Wleft_Eleft/Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb")
NET, ROU, CFG = (os.path.join(D, "syc_4phases_enum.net.xml"), os.path.join(D, "Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb.rou.xml"),
                 os.path.join(D, "Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb_enum.sumocfg"))
pytestmark = pytest.mark.skipif(not os.path.exists(NET), reason="1x1 enum net missing")
SD = 17   # is_green + φ15 + green_elapsed

def _env(**obs_kw):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    kw = dict(fields=("count", "queue", "mean_awt"), phase_state="perphase", priority_source={"ambulance": 5, "bus": 3, "car": 1}, awt_log=True, include_green_elapsed=True); kw.update(obs_kw)
    return SumoEnvironment(net_file=NET, route_file=ROU, cfg_file=CFG, out_csv_name=None, use_gui=False, num_seconds=300, min_green=5, max_green=50, use_max_green=True,
                           single_agent=False, yellow_time=2, delta_time=5, reward_fn=make_priority_avg_waiting_reward({"ambulance": 5, "bus": 3, "car": 1}),
                           observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw), sumo_seed=0, sumo_warnings=False)

def _roll(env, actions):
    s = env.reset(0); t = list(env.ts_ids)[0]; seq = [np.asarray(s[t], float)]
    for a in actions: s, _, _, _ = env.step(action={t: a}); seq.append(np.asarray(s[t], float))
    env.close(); return seq

def test_default_off_identical_to_plain_green_elapsed():
    acts = [0, 0, 1, 1, 1, 2, 2, 0]
    a = _roll(_env(), acts); b = _roll(_env(green_elapsed_reset=False), acts)
    for x, y in zip(a, b): assert np.array_equal(x, y)

def test_reset_makes_all_green_slots_equal_phase_elapsed():
    acts = [0, 0, 1, 1, 1, 2, 2, 2, 0, 0]
    seq = _roll(_env(green_elapsed_reset=True), acts)
    for v in seq[2:]:                                              # 跳过起始步
        head_elapsed = v[1]; ge = [v[2 + sl * SD + SD - 1] for sl in range(12) if v[2 + sl * SD] > 0.5]
        assert ge and max(ge) - min(ge) < 1e-9                     # 绿槽一致
        # 切换后的相位: 计时从切换后的第一个观测起算, = 头部 − 5 s; 起始相位: 从 reset 的观测起算, = 头部
        assert min(abs(ge[0] - (head_elapsed - 0.05)), abs(ge[0] - head_elapsed)) < 1e-6 or head_elapsed > 6.0

def test_reset_differs_from_continuous_only_after_switch_with_shared_movements():
    acts = [0, 0, 0, 1, 1, 1]
    a = _roll(_env(), acts); b = _roll(_env(green_elapsed_reset=True), acts)
    for x, y in zip(a[:4], b[:4]): assert np.array_equal(x, y)     # 切换前 (含切换当步) 相同
    assert not np.array_equal(a[5], b[5])                          # 切换后: 跨相位共享的运动在连续语义下更大
    ge_a = max(a[5][2 + sl * SD + SD - 1] for sl in range(12)); ge_b = max(b[5][2 + sl * SD + SD - 1] for sl in range(12)); assert ge_a > ge_b

def test_exact_anchor_is_true_green_time():
    """green_elapsed_exact + reset: 切换后绿槽值 == 头部相位计时器 − 黄灯 (0.02), 即真实放行时间; 集首相位 (无黄灯) == 头部; min_green ⇔ 值 ≥ 0.05。"""
    acts = [0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 3, 3]
    seq = _roll(_env(green_elapsed_reset=True, green_elapsed_exact=True), acts); switched = False; prev = None
    for v in seq[1:]:
        ge = [v[2 + sl * SD + SD - 1] for sl in range(12) if v[2 + sl * SD] > 0.5]; cur = tuple(int(v[2 + sl * SD] > 0.5) for sl in range(12))
        if prev is not None and cur != prev: switched = True
        prev = cur
        assert ge and max(ge) - min(ge) < 1e-9
        assert abs(ge[0] - (v[1] - (0.02 if switched else 0.0))) < 1e-6
        if switched: assert (ge[0] >= 0.05 - 1e-9) == bool(v[0] > 0.5)   # 切换后: min_green_ok ⇔ 真实放行 ≥ 5 s (集首相位环境按 7 s 计, 不检验)

def test_exact_default_off_unchanged():
    acts = [0, 0, 1, 1, 2, 2]
    a = _roll(_env(), acts); b = _roll(_env(green_elapsed_exact=False), acts)
    for x, y in zip(a, b): assert np.array_equal(x, y)
