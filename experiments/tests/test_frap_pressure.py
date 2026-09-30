"""frap.score (2026-09-30): learned (默认) 逐位不变; pressure 模式下零需求方向分恰为 0, 出口堵满为负; pressure_only 的 Q = 相位内压力之和。"""
import os, sys, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.frap_agent import FRAPQNet

H, SD = 2, 27                       # perphase 头 2 + 每槽 27 = is_green(1) + φ 5 级×3 字段 + ψ 5 级×2 字段 + lane_occ(1)
FIELDS = ("count", "queue", "mean_awt"); DS = ("count", "queue")
LAYOUT = dict(queue_idx=[1 + (p - 1) * 3 + FIELDS.index("queue") for p in range(1, 6)],
              down_queue_idx=[1 + 15 + DS.index("queue") * 5 + (p - 1) for p in range(1, 6)],
              levels=[1, 2, 3, 4, 5])

def _tables(K=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    pm = (torch.rand(1, K, 12, generator=g) > 0.5).float(); pm[0, 0] = 0; pm[0, 0, 0] = 1; pm[0, 1] = 0; pm[0, 1, 1] = 1; pm[0, 2] = 0; pm[0, 2, 2] = 1
    rel = torch.randint(-1, 4, (1, 12, 12), generator=g); rel[0].fill_diagonal_(0)
    return pm, rel, torch.ones(1, 12)

def test_learned_default_identical_and_no_extra_state():
    torch.manual_seed(1); a = FRAPQNet(H, SD, k_max=12)
    torch.manual_seed(1); b = FRAPQNet(H, SD, k_max=12, score_mode="learned")
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())
    x = torch.rand(3, H + 12 * SD); pm, rel, exist = _tables()
    pm, rel, exist = pm.expand(3, -1, -1), rel.expand(3, -1, -1), exist.expand(3, -1)
    assert torch.equal(a(x, pm, rel, exist), b(x, pm, rel, exist))

def _x_with(slot_vals):
    """slot_vals: {slot: (queue per level list, down queue per level list)}; 其余槽位随机但需求为 0。"""
    x = torch.rand(1, H + 12 * SD)
    for s in range(12):
        base = H + s * SD
        for i in LAYOUT["queue_idx"] + LAYOUT["down_queue_idx"]: x[0, base + i] = 0.0
    for s, (qin, qout) in slot_vals.items():
        base = H + s * SD
        for p in range(5): x[0, base + LAYOUT["queue_idx"][p]] = qin[p]; x[0, base + LAYOUT["down_queue_idx"][p]] = qout[p]
    return x

def test_pressure_sign_rules():
    torch.manual_seed(2); net = FRAPQNet(H, SD, k_max=12, score_mode="pressure", layout=LAYOUT)
    x = _x_with({1: ([0.1, 0, 0, 0, 0.2], [0] * 5), 2: ([0.05, 0, 0, 0, 0], [0.9, 0, 0, 0, 0])})
    pm, rel, exist = _tables(); g = net.movement_scores(x, exist)[0]
    assert g[0].item() == 0.0                       # 零需求方向: 分恰为 0 (不受 MLP 影响)
    assert g[1].item() > 0.0                        # 有需求、出口空: 正
    assert g[2].item() < 0.0                        # 出口堵满 (0.9) > 进口 (0.05): 负
    p = net.pressure(x)[0]
    assert torch.isclose(p[1], torch.tensor(0.1 * 1 + 0.2 * 5))   # 等级加权: 5 级桶权重 5
    assert torch.allclose(g[1] / p[1], 1 + torch.tanh(net.g_head(net.encode(x))[0, 1, 0]))   # 倍率 = 1 + tanh(MLP)

def test_pressure_only_is_sum_of_pressures_without_duel():
    torch.manual_seed(3); net = FRAPQNet(H, SD, k_max=12, score_mode="pressure_only", layout=LAYOUT)
    x = _x_with({0: ([0.2, 0, 0, 0, 0], [0] * 5), 1: ([0, 0.1, 0, 0, 0], [0.05, 0, 0, 0, 0]), 2: ([0] * 5, [0.3, 0, 0, 0, 0])})
    pm, rel, exist = _tables(); q = net(x, pm, rel, exist)[0]; p = net.pressure(x)[0]
    for k in range(12): assert torch.isclose(q[k], (pm[0, k] * p).sum(), atol=1e-6)
    assert torch.isclose(q[0], torch.tensor(0.2)) and torch.isclose(q[1], torch.tensor(0.2 - 0.05)) and torch.isclose(q[2], torch.tensor(-0.3))
