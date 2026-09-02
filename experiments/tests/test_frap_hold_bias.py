"""hold_bias (2026-09-02): 默认关时 FRAPQNet 逐位不变; 开时只在当前相位行加 b."""
import os, sys, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.frap_agent import FRAPQNet

def _tables(K=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    pm = (torch.rand(1, K, 12, generator=g) > 0.5).float()
    pm[0, 0] = torch.tensor([1,0,1,0,0,1,0,0,1,1,1,1.])   # 保证行 0 唯一且非零
    rel = torch.randint(-1, 4, (1, 12, 12), generator=g); rel[0].fill_diagonal_(0)
    exist = torch.ones(1, 12)
    return pm, rel, exist

def test_off_is_identical_and_has_no_extra_param():
    torch.manual_seed(1); a = FRAPQNet(2, 16, k_max=12)
    torch.manual_seed(1); b = FRAPQNet(2, 16, k_max=12, hold_bias=False)
    assert "hold_bias" not in dict(a.named_parameters()) and a.hold_bias is None
    x = torch.rand(3, 2 + 12 * 16); pm, rel, exist = _tables()
    pm, rel, exist = pm.expand(3, -1, -1), rel.expand(3, -1, -1), exist.expand(3, -1)
    assert torch.equal(a(x, pm, rel, exist), b(x, pm, rel, exist))

def test_on_with_zero_bias_equals_off_and_matches_only_current_phase():
    torch.manual_seed(1); off = FRAPQNet(2, 16, k_max=12)
    torch.manual_seed(1); on = FRAPQNet(2, 16, k_max=12, hold_bias=True)
    on.load_state_dict(off.state_dict(), strict=False)
    pm, rel, exist = _tables()
    x = torch.rand(1, 2 + 12 * 16)
    x[0, 2::16] = pm[0, 0]                      # is_green 位 = 相位 0 的组成 -> 当前相位 = 0
    q_off = off(x, pm, rel, exist); q_on0 = on(x, pm, rel, exist)
    assert torch.allclose(q_off, q_on0)         # b=0 时逐位一致
    with torch.no_grad(): on.hold_bias.fill_(2.5)
    d = on(x, pm, rel, exist) - q_off
    assert torch.isclose(d[0, 0], torch.tensor(2.5)) and torch.allclose(d[0, 1:], torch.zeros(11))
    x[0, 2::16] = 0.0                            # 全红 (无当前相位) -> 不加偏置
    assert torch.allclose(on(x, pm, rel, exist), off(x, pm, rel, exist))
