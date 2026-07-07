"""Tests for the exp227 fine-tune plumbing (train.py):
  * resolve_obs_priority_table — obs bucket table decoupled from reward table
  * init_checkpoint contract — save_checkpoint round-trips policy weights into
    a FRESH agent of each family (DQN / FRAPAgent / DQNMultiHead), target
    synced to policy, optimizer untouched.
Pure torch, no SUMO. Run: python -m pytest experiments/tests/test_warmstart_obs_split.py -q"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
import torch  # noqa: E402

from train import resolve_obs_priority_table, save_checkpoint  # noqa: E402


# ── obs/reward table split ──────────────────────────────────────────────────

def test_obs_table_legacy_single_key():
    t = {"ambulance": 5, "bus": 3, "car": 1}
    assert resolve_obs_priority_table({"priority_source": t}) == t


def test_obs_table_split_key_wins():
    reward_t = {"ambulance": 4, "bus": 2, "car": 1}
    obs_t = {"ambulance": 5, "bus": 4, "truck": 2, "car": 1}
    cfg = {"priority_source": reward_t, "obs_priority_source": obs_t}
    assert resolve_obs_priority_table(cfg) == obs_t          # obs keeps wiring
    assert cfg["priority_source"] == reward_t                # reward untouched


def test_obs_table_absent_is_none():
    assert resolve_obs_priority_table({}) is None            # class default


# ── init_checkpoint contract (policy loads, target syncs, per family) ───────

def _roundtrip(make_agent):
    a = make_agent()
    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(a, episode=7, model_dir=d)
        b = make_agent()
        # any weight actually differs before load (fresh init)
        pa = torch.cat([p.flatten() for p in a.q_net.state_dict().values()])
        pb = torch.cat([p.flatten() for p in b.q_net.state_dict().values()])
        assert not torch.equal(pa, pb)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["episode"] == 7
        b.q_net.load_state_dict(ck["policy_state_dict"])      # train.py block
        b.target_q_net.load_state_dict(b.q_net.state_dict())
        for k, v in a.q_net.state_dict().items():
            assert torch.equal(v, b.q_net.state_dict()[k])
            assert torch.equal(v, b.target_q_net.state_dict()[k])


def test_warmstart_roundtrip_dqn():
    from sumo_rl.agents.dqn_agent_txw import DQN
    _roundtrip(lambda: DQN(starting_state=tuple([0.0] * 8), state_space=8,
        hidden_dim=16, action_space=4, learning_rate=1e-3, gamma=0.99,
        epsilon=0.1, target_update=10, capacity=100, mini_size=10,
        batch_size=4, eps_start=0.5, eps_end=0.01, eps_decay=100,
        device="cpu"))


def test_warmstart_roundtrip_multihead():
    from sumo_rl.agents.dqn_multihead_agent import DQNMultiHead
    _roundtrip(lambda: DQNMultiHead(starting_state=tuple([0.0] * 8),
        state_space=8, hidden_dim=16, action_space=4, learning_rate=1e-3,
        gamma=0.99, epsilon=0.1, target_update=10, capacity=100,
        mini_size=10, batch_size=4, eps_start=0.5, eps_end=0.01,
        eps_decay=100, device="cpu"))


def test_warmstart_roundtrip_frap():
    from sumo_rl.agents.frap_agent import FRAPAgent
    import numpy as np
    rel = np.full((12, 12), -1, dtype=np.int64)
    for i in range(3):
        rel[i, i] = 0
    rel[0, 1] = rel[1, 0] = 3
    rel[0, 2] = rel[2, 0] = 1
    rel[1, 2] = rel[2, 1] = 2
    pm = np.zeros((2, 12), dtype=np.float32)
    pm[0, 0] = pm[0, 2] = 1.0
    pm[1, 1] = 1.0
    exist = (rel.diagonal() >= 0).astype(np.float32)
    tls_tensors = {"t0": {"pm": pm, "rel": rel, "exist": exist,
                          "mask": np.array([True, True])}}
    def mk():
        return FRAPAgent(obs_dim=2 + 12 * 7, header_dim=2, slot_dim=7,
                         tls_tensors=tls_tensors, lr=1e-3, gamma=0.99,
                         epsilon=0.1, target_update=10, capacity=100,
                         mini_size=10, batch_size=4, eps_start=0.5,
                         eps_end=0.01, eps_decay=100, device="cpu",
                         embed_dim=8, pair_dim=8, k_max=2)
    _roundtrip(mk)
