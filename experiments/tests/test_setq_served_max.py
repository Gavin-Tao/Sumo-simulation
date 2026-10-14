"""集合式相位头 "放行运动另取最大值" 开关 (frap.setq_served_max, 2026-10-14; 默认关), 与 setq_unserved_max 一起用即两侧 "平均 + 最大"。
检验: 参数只多相位头第一层 E×hidden; 两侧都开时特征宽 4E+3, 末尾依次为 u_max、s_max; s_max 等于放行运动编码逐维最大值; 放行集合为空时取 0;
梯度有限; 运动槽重排后 Q 不变; 智能体能选动作与学习; 非 setq 结构打开报错。"""
import os, sys, numpy as np, torch, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.setq_qnet import SetQNet
from sumo_rl.agents.frap_agent import FRAPAgent

H, SD, K, E = 0, 17, 6, 32


def _tables(B=2):
    pm = torch.zeros(B, K, 12)
    pm[:, 0, [0, 1, 6, 7]] = 1; pm[:, 1, [0, 2, 6]] = 1; pm[:, 2, [3, 4, 9, 10]] = 1; pm[:, 3, [3, 5]] = 1; pm[:, 4, [1, 7]] = 1   # 候选 5 不放行任何运动
    return pm, torch.zeros(B, 12, 12, dtype=torch.long), torch.ones(B, 12)


def _x(B=2, seed=3):
    g = torch.Generator().manual_seed(seed); return torch.rand(B, H + 12 * SD, generator=g)


def test_param_count_and_width():
    n0 = sum(p.numel() for p in SetQNet(H, SD, E, K).parameters())
    n1 = sum(p.numel() for p in SetQNet(H, SD, E, K, served_max=True).parameters())
    n2 = sum(p.numel() for p in SetQNet(H, SD, E, K, unserved_max=True, served_max=True).parameters())
    assert n1 - n0 == E * 128 and n2 - n0 == 2 * E * 128
    net = SetQNet(H, SD, E, K, unserved_max=True, served_max=True); pm, rel, ex = _tables()
    assert net.features(_x(), pm, ex).shape == (2, K, 4 * E + 3)


def test_smax_equals_elementwise_max_of_served_codes():
    net = SetQNet(H, SD, E, K, unserved_max=True, served_max=True); pm, rel, ex = _tables(); x = _x()
    f = net.features(x, pm, ex); e = net.encode(x)
    for b in range(2):
        for k in range(K - 1):
            si = (pm[b, k] > 0.5).nonzero().flatten(); ui = (pm[b, k] < 0.5).nonzero().flatten()
            assert torch.allclose(f[b, k, -E:], e[b, si].max(0).values, atol=1e-6)            # 最后一段 = s_max
            assert torch.allclose(f[b, k, -2 * E:-E], e[b, ui].max(0).values, atol=1e-6)      # 倒数第二段 = u_max
    assert torch.equal(f[:, K - 1, -E:], torch.zeros(2, E))                                   # 放行集合为空 → s_max = 0


def test_gradients_finite_and_permutation_invariant():
    torch.manual_seed(0); net = SetQNet(H, SD, E, K, unserved_max=True, served_max=True); pm, rel, ex = _tables()
    x = _x().requires_grad_(True); net(x, pm, rel, ex).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None) and torch.isfinite(x.grad).all()
    x = _x(); perm = torch.tensor([5, 3, 11, 0, 7, 1, 9, 2, 10, 4, 8, 6]); xp = x.reshape(2, 12, SD)[:, perm].reshape(2, -1)
    assert torch.allclose(net(x, pm, rel, ex), net(xp, pm[:, :, perm], rel, ex[:, perm]), atol=1e-5)


def _agent(**kw):
    pmn = np.zeros((K, 12), np.float32); pmn[0, [0, 1, 6, 7]] = 1; pmn[1, [3, 4, 9, 10]] = 1
    rel = np.full((12, 12), -1, np.int64); np.fill_diagonal(rel, 0)
    tt = {"J": dict(pm=pmn, rel=rel, exist=np.ones(12, np.float32), mask=np.array([True, True] + [False] * (K - 2)))}
    return FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors=tt, lr=1e-3, gamma=0.99, epsilon=0.0, target_update=10, capacity=100,
                     mini_size=10, batch_size=8, eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", k_max=K, embed_dim=E, **kw)


def test_agent_acts_and_learns_both_sides():
    torch.manual_seed(0); a = _agent(arch="setq", setq_unserved_max=True, setq_served_max=True)
    assert a.q_net.unserved_max and a.q_net.served_max
    rng = np.random.default_rng(0)
    for _ in range(30):
        a.replay_buffer.add(rng.random(H + 12 * SD).astype(np.float32), int(rng.integers(2)), -1.0, rng.random(H + 12 * SD).astype(np.float32), False, "J",
                            next_mask=np.array([True] + [False] * (K - 1)))
    a.learn_step(); assert np.isfinite(a.loss)
    assert a.take_action(rng.random(H + 12 * SD).astype(np.float32), "J") in (0, 1)


def test_rejected_for_other_architectures():
    with pytest.raises(ValueError): _agent(arch="frap", setq_served_max=True)
    with pytest.raises(ValueError): _agent(arch="menuq", setq_served_max=True)
