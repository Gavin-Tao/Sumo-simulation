"""frap.header_in (2026-10-10, 用户批准): 默认关时 FRAPQNet / MTTQNet / FRAPAgent 逐位不变 (编码器输入仍为 slot_dim, 头部被丢弃);
开时头部广播拼进每个槽的编码器输入 (slot_dim + header_dim), 只改头部就能改变 Q。pressure 模式与 hold_bias 不受影响。"""
import os, sys, numpy as np, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.frap_agent import FRAPQNet, MTTQNet, FRAPAgent

H, SD, K = 2, 16, 12

def _tables(seed=0):
    g = torch.Generator().manual_seed(seed)
    pm = (torch.rand(1, K, 12, generator=g) > 0.5).float(); pm[0, 0] = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1.])
    rel = torch.randint(-1, 4, (1, 12, 12), generator=g); rel[0].fill_diagonal_(0)
    return pm, rel, torch.ones(1, 12)

def _x_pair():
    """两条观测只差头部 (前 H 维), 槽特征完全相同。"""
    torch.manual_seed(7); x1 = torch.rand(1, H + 12 * SD); x2 = x1.clone(); x2[0, :H] = torch.tensor([0.0, 0.73])
    assert not torch.equal(x1[0, :H], x2[0, :H]) and torch.equal(x1[0, H:], x2[0, H:])
    return x1, x2

def test_default_off_is_byte_identical():
    torch.manual_seed(1); a = FRAPQNet(H, SD, k_max=K)
    torch.manual_seed(1); b = FRAPQNet(H, SD, k_max=K, header_in=False)
    assert a.enc[0].in_features == SD and b.enc[0].in_features == SD and a.header_in is False
    assert [k for k, _ in a.named_parameters()] == [k for k, _ in b.named_parameters()]
    pm, rel, exist = _tables(); x1, x2 = _x_pair()
    assert torch.equal(a(x1, pm, rel, exist), b(x1, pm, rel, exist))
    assert torch.equal(a(x1, pm, rel, exist), a(x2, pm, rel, exist))   # 关: 头部被丢弃, 只改头部输出不变

def test_on_reads_header():
    torch.manual_seed(1); on = FRAPQNet(H, SD, k_max=K, header_in=True)
    assert on.enc[0].in_features == SD + H and on.header_in is True
    pm, rel, exist = _tables(); x1, x2 = _x_pair()
    q1, q2 = on(x1, pm, rel, exist), on(x2, pm, rel, exist)
    assert q1.shape == (1, K) and not torch.allclose(q1, q2)         # 开: 只改头部, Q 变
    assert torch.isfinite(q1).all()

def test_on_pressure_mode_still_uses_raw_slots():
    layout = dict(queue_idx=[1 + k * 3 + 1 for k in range(5)], down_queue_idx=None, levels=[1, 2, 3, 4, 5])
    torch.manual_seed(1); on = FRAPQNet(H, SD, k_max=K, score_mode="pressure", layout=layout, header_in=True)
    pm, rel, exist = _tables(); x1, x2 = _x_pair()
    assert torch.equal(on.pressure(x1), on.pressure(x2))              # 压力只看槽特征, 与头部无关
    assert not torch.allclose(on(x1, pm, rel, exist), on(x2, pm, rel, exist))

def test_mtt_inherits_switch():
    torch.manual_seed(1); off = MTTQNet(H, SD, k_max=K, n_heads=2, n_layers=1)
    torch.manual_seed(1); on = MTTQNet(H, SD, k_max=K, n_heads=2, n_layers=1, header_in=True)
    assert off.enc[0].in_features == SD and on.enc[0].in_features == SD + H
    pm, rel, exist = _tables(); x1, x2 = _x_pair()
    off.eval(); on.eval()
    assert torch.equal(off(x1, pm, rel, exist), off(x2, pm, rel, exist))
    assert not torch.allclose(on(x1, pm, rel, exist), on(x2, pm, rel, exist))

def _agent(**kw):
    pm = np.zeros((11, 12), np.float32); pm[0, 0] = 1; pm[1, 1] = 1
    rel = np.full((12, 12), -1, np.int64); rel[:2, :2] = 3; np.fill_diagonal(rel, 0)
    exist = np.zeros(12, np.float32); exist[:2] = 1; mask = np.array([True] * 2 + [False] * 9)
    return FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors={"J": dict(pm=pm, rel=rel, exist=exist, mask=mask)},
                     lr=1e-3, gamma=0.95, epsilon=0.0, target_update=10**9, capacity=100, mini_size=10, batch_size=8,
                     eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", **kw)

def test_agent_plumbing_default_and_on():
    assert _agent().q_net.enc[0].in_features == SD                     # 默认关
    a = _agent(header_in=True); assert a.q_net.enc[0].in_features == SD + H and a.target_q_net.enc[0].in_features == SD + H
    m = _agent(arch="mtt", header_in=True); assert m.q_net.enc[0].in_features == SD + H
    try:
        _agent(arch="mtt_pure", header_in=True); assert False, "mtt_pure 应拒绝 header_in"
    except ValueError:
        pass
