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


# ── MTTCoLightQNet + agent (arch: mtt_colight) ──────────────────────────────

def _toy_cl(B=3, own_dim=2 + 12 * 7):
    x, pm, rel, exist = _toy(B)
    nb = torch.randn(B, 4, own_dim)
    return x, nb, pm, rel, exist


def test_mttcolight_forward_finite_and_shape():
    from sumo_rl.agents.frap_agent import MTTCoLightQNet
    net = MTTCoLightQNet(2, 7, 16, 16, k_max=2, n_neighbors=4)
    x, nb, pm, rel, exist = _toy_cl()
    q = net(x, nb, pm, rel, exist)
    assert q.shape == (3, 2) and torch.isfinite(q).all()


def test_mttcolight_all_missing_neighbors_no_nan():
    from sumo_rl.agents.frap_agent import MTTCoLightQNet
    net = MTTCoLightQNet(2, 7, 16, 16, k_max=2, n_neighbors=4)
    x, _, pm, rel, exist = _toy_cl(B=1)
    zeros = torch.zeros(1, 4, 2 + 12 * 7)          # all neighbours absent
    q = net(x, zeros, pm, rel, exist)
    assert torch.isfinite(q).all(), "all-missing neighbours -> NaN"


def test_mttcolight_coordination_active():
    from sumo_rl.agents.frap_agent import MTTCoLightQNet
    net = MTTCoLightQNet(2, 7, 16, 16, k_max=2, n_neighbors=4)
    x, _, pm, rel, exist = _toy_cl(B=1)
    zeros = torch.zeros(1, 4, 2 + 12 * 7)
    real = torch.randn(1, 4, 2 + 12 * 7)
    assert (net(x, zeros, pm, rel, exist)
            - net(x, real, pm, rel, exist)).abs().max() > 1e-3, "neighbours ignored"


def test_mttcolight_layers0_is_pure_frap_plus_gat():
    # exp234 semantics: n_layers=0 -> refine is identity (no attention layers),
    # net degrades to hand-crafted FRAP duel + neighbour GAT. Lock it.
    from sumo_rl.agents.frap_agent import MTTCoLightQNet
    net = MTTCoLightQNet(2, 7, 16, 16, k_max=2, n_neighbors=4, n_layers=0)
    assert len(net.layers) == 0, "n_layers=0 must build zero attention layers"
    x, nb, pm, rel, exist = _toy_cl(B=2)
    q = net(x, nb, pm, rel, exist)
    assert q.shape == (2, 2) and torch.isfinite(q).all()
    # FRAP core params all present (superset check, same names)
    f = set(dict(FRAPQNet(2, 7, 16, 16, 2).named_parameters()))
    m = set(dict(net.named_parameters()))
    assert f.issubset(m)


def test_mttcolight_agent_learn_gradients():
    import numpy as np
    torch.manual_seed(0); np.random.seed(0)   # deterministic: no cross-test RNG leak
    from sumo_rl.agents.frap_agent import MTTCoLightAgent
    rel = np.full((12, 12), -1, np.int64)
    for i in range(4):
        rel[i, i] = 0
    rel[0, 1] = rel[1, 0] = 3
    pm = np.zeros((2, 12), np.float32); pm[0, 0] = pm[0, 2] = 1; pm[1, 1] = 1
    exist = (rel.diagonal() >= 0).astype(np.float32)
    tt = {"A": {"pm": pm, "rel": rel, "exist": exist, "mask": np.array([True, True])}}
    od = 2 + 12 * 7
    ag = MTTCoLightAgent(od, 2, 7, tt, 1e-2, 0.95, 0.0, 5, 200, 8, 8,
                         0, 0, 1, "cpu", embed_dim=16, pair_dim=16, k_max=2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        ag.replay_buffer.add(rng.standard_normal(od).astype(np.float32), 0, -1.0,
                             rng.standard_normal(od).astype(np.float32), False, "A",
                             rng.standard_normal((4, od)).astype(np.float32),
                             rng.standard_normal((4, od)).astype(np.float32))
    # accumulate "ever received a nonzero gradient" across steps (robust to a
    # single step's batch happening to zero one param's grad)
    ever = {}
    for _ in range(5):
        ag.learn_step()
        for n, p in ag.q_net.named_parameters():
            if ("nb_encoder" in n or "attn_score" in n) and p.grad is not None:
                ever[n] = ever.get(n, False) or (p.grad.abs().sum().item() > 0)
    assert ag.loss is not None and np.isfinite(ag.loss)
    assert ever and all(ever.values()), f"coordination path not trained: {ever}"
