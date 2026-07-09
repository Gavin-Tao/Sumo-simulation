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


# ── MTTPureQNet (arch: mtt_pure, replace-variant) ───────────────────────────

def test_mtt_pure_q_formula_is_sum_of_member_g():
    # Lock the replace semantics: Q(k) == sum of member g on refined tokens,
    # byte-for-byte (no hidden suppression term).
    from sumo_rl.agents.frap_agent import MTTPureQNet
    torch.manual_seed(0)
    net = MTTPureQNet(2, 7, 16, 16, k_max=2)
    x, pm, rel, exist = _toy()
    q = net(x, pm, rel, exist)
    with torch.no_grad():
        d = net.refine(net.encode(x), rel, exist)
        g = net.g_head(d).squeeze(-1)
        q_manual = (g.unsqueeze(1) * pm).sum(-1)
    assert torch.allclose(q, q_manual, atol=0, rtol=0), "Q != sum of member g"


def test_mtt_pure_no_duel_dependence_but_attention_active():
    # Duel params (pair_fc/rel_emb/s_head) must be OUT of the graph: perturbing
    # them cannot change Q. Attention must be IN: a NON-uniform rel_bias
    # perturbation (softmax is shift-invariant — uniform shifts are a no-op)
    # must change Q.
    from sumo_rl.agents.frap_agent import MTTPureQNet
    torch.manual_seed(0)
    net = MTTPureQNet(2, 7, 16, 16, k_max=2)
    x, pm, rel, exist = _toy()
    q0 = net(x, pm, rel, exist)
    with torch.no_grad():
        net.pair_fc.weight.add_(100.0)
        net.rel_emb.weight.add_(100.0)
        net.s_head.weight.add_(100.0)
    assert torch.allclose(net(x, pm, rel, exist), q0), "duel params leaked into Q"
    with torch.no_grad():
        net.rel_bias.weight[3, 0] += 3.0     # crossing rel, head 0 only
    assert (net(x, pm, rel, exist) - q0).abs().max() > 1e-4, \
        "rel-biased attention inactive"


def test_mtt_pure_gradient_routing():
    # Backward must reach enc/g_head/attention and must NOT reach duel params.
    from sumo_rl.agents.frap_agent import MTTPureQNet
    torch.manual_seed(0)
    net = MTTPureQNet(2, 7, 16, 16, k_max=2)
    x, pm, rel, exist = _toy()
    net(x, pm, rel, exist).sum().backward()
    grads = {n: p.grad for n, p in net.named_parameters()}
    for n in ("enc.0.weight", "g_head.weight"):
        assert grads[n] is not None and grads[n].abs().sum() > 0, f"no grad: {n}"
    assert any(k.startswith("layers.") and grads[k] is not None
               and grads[k].abs().sum() > 0 for k in grads), "attention untrained"
    for n in ("pair_fc.weight", "rel_emb.weight", "s_head.weight"):
        assert grads[n] is None or grads[n].abs().sum() == 0, f"duel grad leak: {n}"


def test_mtt_pure_via_frap_agent_factory():
    # arch="mtt_pure" must build through FRAPAgent unchanged surface and learn.
    import numpy as np
    torch.manual_seed(0); np.random.seed(0)
    from sumo_rl.agents.frap_agent import FRAPAgent, MTTPureQNet
    rel = np.full((12, 12), -1, np.int64)
    for i in range(4):
        rel[i, i] = 0
    rel[0, 1] = rel[1, 0] = 3
    pm = np.zeros((2, 12), np.float32); pm[0, 0] = pm[0, 2] = 1; pm[1, 1] = 1
    exist = (rel.diagonal() >= 0).astype(np.float32)
    tt = {"A": {"pm": pm, "rel": rel, "exist": exist, "mask": np.array([True, True])}}
    od = 2 + 12 * 7
    ag = FRAPAgent(od, 2, 7, tt, 1e-2, 0.95, 0.0, 5, 200, 8, 8,
                   0, 0, 1, "cpu", embed_dim=16, pair_dim=16, k_max=2,
                   arch="mtt_pure")
    assert isinstance(ag.q_net, MTTPureQNet)
    rng = np.random.default_rng(0)
    for _ in range(20):
        ag.replay_buffer.add(rng.standard_normal(od).astype(np.float32), 0, -1.0,
                             rng.standard_normal(od).astype(np.float32), False, "A")
    a = ag.take_action(rng.standard_normal(od).astype(np.float32), "A")
    assert a in (0, 1)
    for _ in range(3):
        ag.learn_step()
    assert ag.loss is not None and np.isfinite(ag.loss)


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
