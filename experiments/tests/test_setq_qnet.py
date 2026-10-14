"""集合式相位头 (arch setq, 2026-10-14): 形状与参数量; 运动槽重排 (特征与相位表一起换) 后 Q 不变; is_current 只在与 is_green 一致的候选上为 1;
被压运动的特征会影响 Q (u_k); 智能体选动作、学习、F1 掩码走通; 不支持的组合报错。"""
import os, sys, numpy as np, torch, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.setq_qnet import SetQNet
from sumo_rl.agents.frap_agent import FRAPAgent

H, SD, K = 0, 17, 6


def _tables(B=2):
    pm = torch.zeros(B, K, 12)
    pm[:, 0, [0, 1, 6, 7]] = 1; pm[:, 1, [0, 2, 6]] = 1; pm[:, 2, [3, 4, 9, 10]] = 1; pm[:, 3, [3, 5]] = 1; pm[:, 4, [1, 7]] = 1; pm[:, 5, [9, 10]] = 1
    return pm, torch.zeros(B, 12, 12, dtype=torch.long), torch.ones(B, 12)


def _x(cur_row, B=2, seed=3):
    g = torch.Generator().manual_seed(seed); x = torch.rand(B, H + 12 * SD, generator=g); x[:, H::SD] = cur_row
    return x


def test_shapes_and_param_count():
    torch.manual_seed(0); net = SetQNet(H, SD, embed_dim=32, k_max=K)
    assert sum(p.numel() for p in net.parameters()) == (SD * 32 + 32 + 64) + ((2 * 32 + 3) * 128 + 128) + (128 + 1)
    pm, rel, ex = _tables(); assert net(_x(pm[0, 0]), pm, rel, ex).shape == (2, K)


def test_permutation_of_slots_leaves_q_unchanged():
    torch.manual_seed(0); net = SetQNet(H, SD, k_max=K); pm, rel, ex = _tables(); x = _x(pm[0, 0])
    perm = torch.tensor([5, 3, 11, 0, 7, 1, 9, 2, 10, 4, 8, 6])
    xp = x.reshape(2, 12, SD)[:, perm].reshape(2, -1); pmp = pm[:, :, perm]; exp_ = ex[:, perm]
    assert torch.allclose(net(x, pm, rel, ex), net(xp, pmp, rel, exp_), atol=1e-5)


def test_is_current_flag():
    net = SetQNet(H, SD, k_max=K); pm, rel, ex = _tables(); f = net.features(_x(pm[0, 2]), pm, ex)
    assert f[:, :, -1].tolist() == [[0, 0, 1, 0, 0, 0]] * 2


def test_unserved_movements_affect_q():
    torch.manual_seed(0); net = SetQNet(H, SD, k_max=K); pm, rel, ex = _tables(); x = _x(pm[0, 0]); x2 = x.clone()
    x2[:, H + 3 * SD + 1: H + 4 * SD - 1] += 2.0                       # 只改运动 3 (不在候选 0 里) 的车流特征
    with torch.no_grad(): q, q2 = net(x, pm, rel, ex), net(x2, pm, rel, ex)
    assert not torch.allclose(q[:, 0], q2[:, 0])                       # 候选 0 的 Q 也随被压运动改变


def _agent(**kw):
    pmn = np.zeros((K, 12), np.float32); pmn[0, [0, 1, 6, 7]] = 1; pmn[1, [3, 4, 9, 10]] = 1
    rel = np.full((12, 12), -1, np.int64); np.fill_diagonal(rel, 0)
    tt = {"J": dict(pm=pmn, rel=rel, exist=np.ones(12, np.float32), mask=np.array([True, True] + [False] * (K - 2)))}
    return FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors=tt, lr=1e-3, gamma=0.99, epsilon=0.0, target_update=10, capacity=100,
                     mini_size=10, batch_size=8, eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", k_max=K, embed_dim=32, **kw)


def test_agent_acts_learns_with_f1_mask():
    torch.manual_seed(0); a = _agent(arch="setq"); assert isinstance(a.q_net, SetQNet)
    rng = np.random.default_rng(0)
    for _ in range(30):
        s = rng.random(H + 12 * SD).astype(np.float32); ns = rng.random(H + 12 * SD).astype(np.float32)
        a.replay_buffer.add(s, int(rng.integers(2)), -1.0, ns, False, "J", next_mask=np.array([True] + [False] * (K - 1)))
    a.learn_step(); assert np.isfinite(a.loss)
    m = np.zeros(K, bool); m[1] = True
    assert a.take_action(rng.random(H + 12 * SD).astype(np.float32), "J", mask=m) == 1


def test_unsupported_combinations_rejected():
    with pytest.raises(ValueError): _agent(arch="setq", hold_bias=True)
    with pytest.raises(ValueError): _agent(arch="setq", header_in=True)
