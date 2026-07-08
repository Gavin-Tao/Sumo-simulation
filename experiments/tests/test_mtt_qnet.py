"""MTTQNet (arch: mtt) — forward contract parity with FRAPQNet + start-state
sanity. Pure torch, no SUMO. FRAPQNet must remain unaffected."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import torch
from sumo_rl.agents.frap_agent import FRAPQNet, MTTQNet


def _toy(B=4):
    rel = torch.full((B, 12, 12), -1, dtype=torch.long)
    for i in range(3):
        rel[:, i, i] = 0
    rel[:, 0, 1] = rel[:, 1, 0] = 3
    rel[:, 0, 2] = rel[:, 2, 0] = 2
    pm = torch.zeros(B, 2, 12); pm[:, 0, 0] = pm[:, 0, 2] = 1; pm[:, 1, 1] = 1
    exist = torch.zeros(B, 12); exist[:, :3] = 1
    x = torch.randn(B, 2 + 12 * 7)
    return x, pm, rel, exist


def test_mtt_forward_shape_and_finite():
    net = MTTQNet(header_dim=2, slot_dim=7, embed_dim=16, pair_dim=16, k_max=2)
    x, pm, rel, exist = _toy()
    q = net(x, pm, rel, exist)
    assert q.shape == (4, 2)
    assert torch.isfinite(q).all(), "MTT produced non-finite Q (mask/NaN leak)"


def test_mtt_signature_matches_frap():
    # both accept the identical call and return identical shape -> drop-in
    x, pm, rel, exist = _toy()
    qf = FRAPQNet(2, 7, 16, 16, 2)(x, pm, rel, exist)
    qm = MTTQNet(2, 7, 16, 16, 2)(x, pm, rel, exist)
    assert qf.shape == qm.shape == (4, 2)


def test_mtt_extra_params_only():
    # MTT = FRAP params + attention block; FRAP submodule names unchanged
    f = set(dict(FRAPQNet(2, 7, 16, 16, 2).named_parameters()))
    m = set(dict(MTTQNet(2, 7, 16, 16, 2).named_parameters()))
    assert f.issubset(m), "MTT dropped/renamed a FRAP parameter"
    assert any(k.startswith("layers.") or k.startswith("rel_bias") for k in m - f)
