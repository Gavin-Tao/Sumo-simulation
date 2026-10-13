"""枚举条件化 (arch menuq) 支持 legacy 观测 (2026-10-13, 用户批准)。
legacy 头部 = 路口全部绿相位独热 + min_green_ok, 槽里没有 is_green; menuq 的 "候选是否当前相位" 改从头部独热读。
检验: is_current 等于头部独热且不受槽内数值影响; 非法取值报错; FRAP/GS-G 不能用 legacy; 各路口相位数不同时拒绝; 智能体选动作与学习走通。
默认 (current_from=slots) 路径与改前逐位相同, 已另行用旧模块比对。"""
import os, sys, types, numpy as np, torch, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
from sumo_rl.agents.menu_qnet import MenuQNet
from sumo_rl.agents.frap_agent import FRAPAgent
import frap_glue

K, SD = 12, 15                 # 12 个枚举相位; 槽 = φ 15 维 (legacy 无 is_green)
H = K + 1                      # 头部 = 12 位独热 + min_green_ok
META = os.path.join(_REPO, "nets/syc/1x1/Eq_350_BC_Wstr_Estr-Wleft_Eleft/Eq_350_20-80-99-1-Bus-Car-Amb_WEstr_only_NSamb/syc_4phases_enum_meta.json")


def _x(cur, B=3, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.rand(B, H + 12 * SD, generator=g)
    x[:, :K] = 0; x[torch.arange(B), cur] = 1.0; x[:, K] = 1.0
    return x


def test_is_current_comes_from_header_onehot_only():
    net = MenuQNet(H, SD, K, hidden=16, n_layers=1, current_from="legacy_header")
    pm = (torch.rand(3, K, 12, generator=torch.Generator().manual_seed(1)) > 0.5).float(); ex = torch.ones(3, 12)
    cur = torch.tensor([0, 5, 11]); d = net.descriptors(_x(cur), pm, ex)
    assert torch.equal(d[:, :, 12], torch.nn.functional.one_hot(cur, K).float())
    x2 = _x(cur, seed=9); x2[:, :H] = _x(cur)[:, :H]                    # 槽内数值全换, 头部不变
    assert torch.equal(net.descriptors(x2, pm, ex)[:, :, 12], d[:, :, 12])
    assert torch.equal(d[:, :, :12], pm)                                   # 相位说明书不变


def test_invalid_options_rejected():
    with pytest.raises(ValueError): MenuQNet(H, SD, K, current_from="bad")
    with pytest.raises(ValueError): MenuQNet(0, SD, K, current_from="legacy_header")


def _tables():
    return frap_glue.load_enum_tables(META)


def _env(n_phases=(12,)):
    ids = [f"t{i}" for i in range(len(n_phases))]
    return types.SimpleNamespace(observation_space=types.SimpleNamespace(shape=(H + 12 * SD,)), ts_ids=ids,
                                 traffic_signals={t: types.SimpleNamespace(num_green_phases=k) for t, k in zip(ids, n_phases)})


def _cfg(arch):
    return {"obs_phase_state": "legacy", "frap": {"arch": arch, "menu_layers": 1}, "lr": 1e-3, "gamma": 0.99, "epsilon": 0.0,
            "capacity": 100, "mini_size": 10, "batch_size": 8, "eps_start": 0.0, "eps_end": 0.0, "eps_decay": 1}


def test_glue_opens_legacy_only_for_menuq():
    tables = _tables()
    with pytest.raises(AssertionError): frap_glue.build_frap_agent(_cfg("frap"), tables, _env(), "cpu")
    with pytest.raises(AssertionError): frap_glue.build_frap_agent(_cfg("gsg"), tables, _env(), "cpu")
    with pytest.raises(SystemExit): frap_glue.build_frap_agent(_cfg("menuq"), tables, _env((12, 8)), "cpu")
    with pytest.raises(ValueError):
        FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors=tables["tls"], lr=1e-3, gamma=0.99, epsilon=0.0, target_update=10, capacity=100,
                  mini_size=10, batch_size=8, eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", k_max=K, menu_current_from="legacy_header")   # arch 默认 frap → 拒绝


def test_legacy_menuq_agent_acts_and_learns():
    tables = _tables(); torch.manual_seed(0)
    a = frap_glue.build_frap_agent(_cfg("menuq"), tables, _env(), "cpu")
    assert a.q_net.current_from == "legacy_header" and a.q_net.header_dim == H and a.q_net.slot_dim == SD
    tid = list(tables["tls"])[0]; a._idx = {"t0": a._idx[tid]} if "t0" not in a._idx else a._idx
    rng = np.random.default_rng(0)
    for _ in range(30):
        s = rng.random(H + 12 * SD).astype(np.float32); ns = rng.random(H + 12 * SD).astype(np.float32)
        a.replay_buffer.add(s, int(rng.integers(K)), -1.0, ns, False, "t0")
    a.learn_step(); assert np.isfinite(a.loss)
    assert 0 <= a.take_action(rng.random(H + 12 * SD).astype(np.float32), "t0") < K
