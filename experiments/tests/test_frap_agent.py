"""Tests for FRAPQNet / FRAPAgent (FRAP_ENUM_PLAN Tasks 4-5). Pure torch,
no SUMO. Run: python -m pytest experiments/tests/test_frap_agent.py -q"""
import sys, os, math

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import torch  # noqa: E402
from sumo_rl.agents.frap_agent import FRAPQNet  # noqa: E402


def _toy():
    """3 slots exist (0,1,2); rel: 0-1 crossing(3), 0-2 compatible(1),
    1-2 merge(2). Menu: p0={0,2}, p1={1} (both maximal)."""
    rel = torch.full((12, 12), -1, dtype=torch.long)
    for i in range(3):
        rel[i, i] = 0
    rel[0, 1] = rel[1, 0] = 3
    rel[0, 2] = rel[2, 0] = 1
    rel[1, 2] = rel[2, 1] = 2
    pm = torch.zeros(2, 12)
    pm[0, 0] = pm[0, 2] = 1
    pm[1, 1] = 1
    exist = torch.zeros(12)
    exist[:3] = 1
    return rel, pm, exist


def test_forward_shape_and_finite():
    net = FRAPQNet(header_dim=2, slot_dim=7, k_max=4)
    rel, pm, exist = _toy()
    pm4 = torch.zeros(4, 12)
    pm4[:2] = pm
    x = torch.randn(5, 2 + 12 * 7)
    q = net(x, pm4.unsqueeze(0).expand(5, -1, -1),
            rel.unsqueeze(0).expand(5, -1, -1), exist.unsqueeze(0).expand(5, -1))
    assert q.shape == (5, 4) and torch.isfinite(q[:, :2]).all()


def test_q_decomposition_exact():
    """Q(p) == sum g(members) + sum over suppressed n of max member duel."""
    torch.manual_seed(1)
    net = FRAPQNet(header_dim=2, slot_dim=7, k_max=2)
    rel, pm, exist = _toy()
    x = torch.randn(1, 2 + 12 * 7)
    q = net(x, pm.unsqueeze(0), rel.unsqueeze(0), exist.unsqueeze(0))
    d = net.encode(x)
    g = net.g_head(d).squeeze(-1)
    s = net.duel_scores(d, rel.unsqueeze(0))
    # p0={0,2}: suppressed n=1 (conflicts 0 via crossing, 2 via merge)
    q0 = g[0, 0] + g[0, 2] + torch.max(s[0, 0, 1], s[0, 2, 1])
    # p1={1}: suppressed 0 (crossing) and 2 (merge)
    q1 = g[0, 1] + s[0, 1, 0] + s[0, 1, 2]
    assert torch.allclose(q[0, 0], q0, atol=1e-5)
    assert torch.allclose(q[0, 1], q1, atol=1e-5)


def test_gradients_flow():
    net = FRAPQNet(header_dim=2, slot_dim=7, k_max=2)
    rel, pm, exist = _toy()
    x = torch.randn(3, 2 + 12 * 7, requires_grad=True)
    q = net(x, pm.unsqueeze(0).expand(3, -1, -1),
            rel.unsqueeze(0).expand(3, -1, -1), exist.unsqueeze(0).expand(3, -1))
    q[:, :2].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
