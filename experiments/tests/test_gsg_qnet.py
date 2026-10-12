"""GS-G (2026-10-12): arch gsg = 原 FRAP 学习打分 + 按运动的截断/起动代价。检验: 代价关掉时等于 FRAP 核心; 当前绿灯集合对应的相位无代价;
gt 只影响有截断的候选; 头部 (tau, mg) 只经代价进入; 梯度可达; 默认 FRAP 路径不变; 缺头部或缺 gt 时拒绝。"""
import os, sys, numpy as np, torch, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.gsg_qnet import GSGQNet
from sumo_rl.agents.frap_agent import FRAPAgent, FRAPQNet
H, SD, K = 2, 17, 6        # 槽 = is_green + φ15 + gt

def _tables():
    pm = torch.zeros(1, K, 12)
    pm[0, 0, [0, 1, 6, 7]] = 1; pm[0, 1, [0, 2, 6]] = 1; pm[0, 2, [3, 4, 9, 10]] = 1; pm[0, 3, [3, 5]] = 1
    rel = torch.randint(0, 4, (1, 12, 12), generator=torch.Generator().manual_seed(0)); rel[0].fill_diagonal_(0)
    return pm, rel, torch.ones(1, 12)

def _x(cur_row, B=2):
    torch.manual_seed(3); x = torch.rand(B, H + 12 * SD); x[:, H::SD] = cur_row          # is_green = 当前相位成员
    return x

def _net():
    torch.manual_seed(0); return GSGQNet(H, SD, k_max=K)

def test_costs_off_equals_core():
    net = _net(); pm, rel, ex = _tables(); x = _x(pm[0, 0]); P, R, E = pm.expand(2, -1, -1), rel.expand(2, -1, -1), ex.expand(2, -1)
    for m in (net.kappa_cut, net.kappa_new): torch.nn.init.constant_(m[2].bias, -40.0); torch.nn.init.zeros_(m[2].weight)
    feat = x[:, H:].reshape(2, 12, SD)[:, :, :SD - 1].reshape(2, -1)
    assert torch.allclose(net(x, P, R, E), net.core(feat, P, R, E), atol=1e-6)

def test_current_phase_has_no_cost_and_partial_switch_cheaper():
    net = _net(); pm, rel, ex = _tables(); x = _x(pm[0, 0]); P, R, E = pm.expand(2, -1, -1), rel.expand(2, -1, -1), ex.expand(2, -1)
    feat = x[:, H:].reshape(2, 12, SD)[:, :, :SD - 1].reshape(2, -1)
    d = net.core(feat, P, R, E) - net(x, P, R, E)                       # 每个候选付的总代价
    assert torch.allclose(d[:, 0], torch.zeros(2), atol=1e-7)          # 当前相位 0 代价
    assert (d[:, 1] < d[:, 2]).all()                                    # 半切换 (截 1 新 1) 比整体换相 (截 4 新 4) 便宜

def test_gt_and_header_enter_only_costs():
    net = _net(); pm, rel, ex = _tables(); x = _x(pm[0, 0]); P, R, E = pm.expand(2, -1, -1), rel.expand(2, -1, -1), ex.expand(2, -1)
    x2 = x.clone(); x2[:, H + SD - 1::SD] = 0.9; x2[:, :2] = torch.tensor([1.0, 0.4])
    q1, q2 = net(x, P, R, E), net(x2, P, R, E)
    assert torch.allclose(q1[:, 0], q2[:, 0], atol=1e-7)                # 当前相位不受时间量影响
    assert not torch.allclose(q1[:, 2], q2[:, 2])

def test_grad_and_agent_plumbing():
    net = _net(); pm, rel, ex = _tables(); x = _x(pm[0, 1]); net(x, pm.expand(2, -1, -1), rel.expand(2, -1, -1), ex.expand(2, -1)).sum().backward()
    assert net.kappa_cut[0].weight.grad is not None and net.kappa_new[0].weight.grad is not None
    pmn = np.zeros((K, 12), np.float32); pmn[0, 0] = 1; pmn[1, 1] = 1
    reln = np.full((12, 12), -1, np.int64); reln[:2, :2] = 3; np.fill_diagonal(reln, 0)
    exn = np.zeros(12, np.float32); exn[:2] = 1; mask = np.array([True] * 2 + [False] * (K - 2))
    kw = dict(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors={"J": dict(pm=pmn, rel=reln, exist=exn, mask=mask)}, lr=1e-3, gamma=0.95, epsilon=0.0,
              target_update=10, capacity=100, mini_size=10, batch_size=8, eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", k_max=K)
    assert isinstance(FRAPAgent(**kw).q_net, FRAPQNet)                  # 默认不变
    a = FRAPAgent(**kw, arch="gsg", gsg_kw=dict(gt_in_slot=True)); assert isinstance(a.q_net, GSGQNet)
    assert a.take_action(np.random.default_rng(0).random(H + 12 * SD).astype(np.float32), "J") in (0, 1)
    with pytest.raises(ValueError): FRAPAgent(**kw, arch="gsg", gsg_kw=dict(gt_in_slot=False))
    with pytest.raises(ValueError): GSGQNet(0, SD)

def test_free_sign_variant():
    pm, rel, ex = _tables(); x = _x(pm[0, 0]); P, R, E = pm.expand(2, -1, -1), rel.expand(2, -1, -1), ex.expand(2, -1)
    torch.manual_seed(0); net = GSGQNet(H, SD, k_max=K, cost_sign="free")
    feat = x[:, H:].reshape(2, 12, SD)[:, :, :SD - 1].reshape(2, -1)
    assert torch.allclose(net(x, P, R, E), net.core(feat, P, R, E), atol=1e-7)     # 零初始化 → 起点无代价
    torch.nn.init.constant_(net.kappa_cut[2].bias, -1.0)
    d = net.core(feat, P, R, E) - net(x, P, R, E); assert (d[:, 2] < 0).all()      # 代价可为负 (切换可"加分")
    with pytest.raises(ValueError): GSGQNet(H, SD, cost_sign="bad")
