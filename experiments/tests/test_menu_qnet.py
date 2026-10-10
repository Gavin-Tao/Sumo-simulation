"""arch menuq (2026-10-10, 用户批准): 菜单条件化 Q 网络。默认 arch frap 不受影响; menuq 对候选相位置换等变、读头部、读 is_current;
FRAPAgent(arch="menuq") 正确构造, 与 pressure/hold_bias/header_in 互斥。"""
import os, sys, numpy as np, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.menu_qnet import MenuQNet
from sumo_rl.agents.frap_agent import FRAPAgent, FRAPQNet

H, SD, K = 2, 17, 12

def _tables(seed=0):
    g = torch.Generator().manual_seed(seed)
    pm = (torch.rand(1, K, 12, generator=g) > 0.5).float(); pm[0, 0] = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1.])
    rel = torch.randint(-1, 4, (1, 12, 12), generator=g); rel[0].fill_diagonal_(0)
    return pm, rel, torch.ones(1, 12)

def test_shape_and_param_count():
    torch.manual_seed(0); net = MenuQNet(H, SD, k_max=K, hidden=128, n_layers=2)
    pm, rel, exist = _tables(); x = torch.rand(3, H + 12 * SD)
    q = net(x, pm.expand(3, -1, -1), rel.expand(3, -1, -1), exist.expand(3, -1))
    assert q.shape == (3, K) and torch.isfinite(q).all()
    n = sum(p.numel() for p in net.parameters()); assert 30000 < n < 60000   # 与 8STD MLP (4.2 万) 同量级

def test_permutation_equivariant_over_candidates():
    torch.manual_seed(0); net = MenuQNet(H, SD, k_max=K)
    pm, rel, exist = _tables(); x = torch.rand(1, H + 12 * SD)
    perm = torch.randperm(K, generator=torch.Generator().manual_seed(3))
    q = net(x, pm, rel, exist); qp = net(x, pm[:, perm], rel, exist)
    assert torch.allclose(q[:, perm], qp)                                   # 候选顺序无关, 只看说明书

def test_reads_header_and_is_current():
    torch.manual_seed(0); net = MenuQNet(H, SD, k_max=K)
    pm, rel, exist = _tables(); x1 = torch.rand(1, H + 12 * SD); x2 = x1.clone(); x2[0, :H] = torch.tensor([0.0, 0.9])
    assert not torch.allclose(net(x1, pm, rel, exist), net(x2, pm, rel, exist))   # 头部进入 Q
    x3 = x1.clone(); x3[0, H::SD] = pm[0, 0]                                       # is_green 位 = 候选 0 的组成 -> is_current_0 = 1
    d = net.descriptors(x3, pm, exist); assert d[0, 0, 12] == 1.0 and d[0, 1:, 12].sum() == 0
    assert torch.equal(d[0, :, :12], pm[0])

def test_agent_default_unchanged_and_menuq_plumbing():
    pmn = np.zeros((11, 12), np.float32); pmn[0, 0] = 1; pmn[1, 1] = 1
    rel = np.full((12, 12), -1, np.int64); rel[:2, :2] = 3; np.fill_diagonal(rel, 0)
    exist = np.zeros(12, np.float32); exist[:2] = 1; mask = np.array([True] * 2 + [False] * 9)
    def agent(**kw):
        return FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors={"J": dict(pm=pmn, rel=rel, exist=exist, mask=mask)},
                         lr=1e-3, gamma=0.95, epsilon=0.0, target_update=10**9, capacity=100, mini_size=10, batch_size=8,
                         eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", **kw)
    assert isinstance(agent().q_net, FRAPQNet)                                     # 默认不变
    m = agent(arch="menuq", menu_hidden=64, menu_layers=1)
    assert isinstance(m.q_net, MenuQNet) and m.q_net.mlp[0].in_features == H + 12 * SD + 13 and m.q_net.mlp[0].out_features == 64
    s = np.random.default_rng(0).random(H + 12 * SD).astype(np.float32)
    a = m.take_action(s, "J"); assert a in (0, 1)                                   # 掩码后只在有效候选里选
    for bad in (dict(arch="menuq", score_mode="pressure", layout=dict(queue_idx=[1, 4, 7, 10, 13], down_queue_idx=None)),
                dict(arch="menuq", hold_bias=True), dict(arch="menuq", header_in=True)):
        try:
            agent(**bad); assert False, f"应拒绝 {bad}"
        except ValueError:
            pass
