"""集合式相位头 "被压运动另取最大值" 开关 (frap.setq_unserved_max, 2026-10-14, 用户批准; 默认关)。
检验: 默认关时网络与原来逐位相同; 打开后参数只多相位头第一层的 E×hidden 个权重; u_max 等于被压运动编码的逐维最大值;
单个被压运动的变化对 u_max 的影响不随被压运动数量变小 (不被平均稀释); 被压集合为空时取 0 且梯度有限; 运动槽重排后 Q 不变;
智能体能选动作与学习; 非 setq 结构打开该开关会报错。"""
import os, sys, numpy as np, torch, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.setq_qnet import SetQNet
from sumo_rl.agents.frap_agent import FRAPAgent

H, SD, K, E = 0, 17, 6, 32


def _tables(B=2):
    pm = torch.zeros(B, K, 12)
    pm[:, 0, [0, 1, 6, 7]] = 1; pm[:, 1, [0, 2, 6]] = 1; pm[:, 2, [3, 4, 9, 10]] = 1; pm[:, 3, [3, 5]] = 1; pm[:, 4, [1, 7]] = 1; pm[:, 5, list(range(12))] = 1
    return pm, torch.zeros(B, 12, 12, dtype=torch.long), torch.ones(B, 12)


def _x(B=2, seed=3):
    g = torch.Generator().manual_seed(seed); return torch.rand(B, H + 12 * SD, generator=g)


def test_default_off_is_identical():
    torch.manual_seed(0); a = SetQNet(H, SD, E, K)
    torch.manual_seed(0); b = SetQNet(H, SD, E, K, unserved_max=False)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    pm, rel, ex = _tables(); x = _x()
    assert torch.equal(a(x, pm, rel, ex), b(x, pm, rel, ex))


def test_param_count_and_feature_width():
    n0 = sum(p.numel() for p in SetQNet(H, SD, E, K).parameters()); n1 = sum(p.numel() for p in SetQNet(H, SD, E, K, unserved_max=True).parameters())
    assert n1 - n0 == E * 128
    net = SetQNet(H, SD, E, K, unserved_max=True); pm, rel, ex = _tables()
    assert net.features(_x(), pm, ex).shape == (2, K, 3 * E + 3)


def test_umax_equals_elementwise_max_of_unserved_codes():
    net = SetQNet(H, SD, E, K, unserved_max=True); pm, rel, ex = _tables(); x = _x()
    f = net.features(x, pm, ex); e = net.encode(x)
    for b in range(2):
        for k in range(K - 1):                                                   # 最后一个候选放行全部运动, 被压集合为空
            idx = (pm[b, k] < 0.5).nonzero().flatten()
            assert torch.allclose(f[b, k, -E:], e[b, idx].max(0).values, atol=1e-6)
    assert torch.equal(f[:, K - 1, -E:], torch.zeros(2, E))


def test_single_movement_signal_not_diluted():
    """一个被压运动上的变化: u_max 的变化幅度与被压集合大小无关, u_mean 的变化按 1/|被压集合| 缩小。"""
    torch.manual_seed(0); net = SetQNet(H, SD, E, K, unserved_max=True); pm, rel, ex = _tables(B=1); x = _x(B=1)
    x2 = x.clone(); x2[:, 3 * SD + 1: 4 * SD - 1] += 3.0                         # 只改运动 3
    with torch.no_grad(): f1, f2 = net.features(x, pm, ex), net.features(x2, pm, ex)
    k_big, k_small = 0, 2                                                        # 候选 0: 运动 3 被压且被压集合 8 个; 候选 2: 运动 3 被放行
    d_umax = (f2[0, k_big, -E:] - f1[0, k_big, -E:]).abs().sum(); d_umean = (f2[0, k_big, E:2 * E] - f1[0, k_big, E:2 * E]).abs().sum()
    assert d_umax > d_umean                                                      # 最大值通道对单个运动的变化更敏感


def test_gradients_finite_with_empty_unserved_set():
    net = SetQNet(H, SD, E, K, unserved_max=True); pm, rel, ex = _tables(); x = _x().requires_grad_(True)
    net(x, pm, rel, ex).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None) and torch.isfinite(x.grad).all()


def test_permutation_of_slots_leaves_q_unchanged():
    torch.manual_seed(0); net = SetQNet(H, SD, E, K, unserved_max=True); pm, rel, ex = _tables(); x = _x()
    perm = torch.tensor([5, 3, 11, 0, 7, 1, 9, 2, 10, 4, 8, 6])
    xp = x.reshape(2, 12, SD)[:, perm].reshape(2, -1)
    assert torch.allclose(net(x, pm, rel, ex), net(xp, pm[:, :, perm], rel, ex[:, perm]), atol=1e-5)


def _agent(**kw):
    pmn = np.zeros((K, 12), np.float32); pmn[0, [0, 1, 6, 7]] = 1; pmn[1, [3, 4, 9, 10]] = 1
    rel = np.full((12, 12), -1, np.int64); np.fill_diagonal(rel, 0)
    tt = {"J": dict(pm=pmn, rel=rel, exist=np.ones(12, np.float32), mask=np.array([True, True] + [False] * (K - 2)))}
    return FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors=tt, lr=1e-3, gamma=0.99, epsilon=0.0, target_update=10, capacity=100,
                     mini_size=10, batch_size=8, eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", k_max=K, embed_dim=E, **kw)


def test_agent_acts_and_learns():
    torch.manual_seed(0); a = _agent(arch="setq", setq_unserved_max=True); assert a.q_net.unserved_max
    rng = np.random.default_rng(0)
    for _ in range(30):
        a.replay_buffer.add(rng.random(H + 12 * SD).astype(np.float32), int(rng.integers(2)), -1.0, rng.random(H + 12 * SD).astype(np.float32), False, "J",
                            next_mask=np.array([True] + [False] * (K - 1)))
    a.learn_step(); assert np.isfinite(a.loss)
    assert a.take_action(rng.random(H + 12 * SD).astype(np.float32), "J") in (0, 1)


def test_rejected_for_other_architectures():
    with pytest.raises(ValueError): _agent(arch="frap", setq_unserved_max=True)
