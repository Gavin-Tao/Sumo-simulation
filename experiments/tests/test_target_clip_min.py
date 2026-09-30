"""target_clip_min (2026-09-30): TD 目标下限开关, 默认 None 逐位不变; 设定后 FRAPAgent 的 loss 等于对 clamp 后目标的 Huber loss。"""
import os, sys, numpy as np, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.frap_agent import FRAPAgent
from sumo_rl.agents.dqn_agent_txw import DQN

H, SD, K = 2, 27, 4
def _tables():
    pm = np.zeros((K, 12), np.float32); pm[0, 0] = 1; pm[1, 1] = 1; pm[2, 2] = 1; pm[3, 3] = 1
    rel = np.full((12, 12), -1, np.int64); rel[:4, :4] = 3; np.fill_diagonal(rel, 0)
    exist = np.zeros(12, np.float32); exist[:4] = 1
    mask = np.array([True] * K + [False] * (11 - K))
    pm11 = np.zeros((11, 12), np.float32); pm11[:K] = pm
    return {"J": dict(pm=pm11, rel=rel, exist=exist, mask=mask)}

def _agent(clip_min):
    return FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors=_tables(), lr=1e-3, gamma=0.95,
                     epsilon=0.0, target_update=10**9, capacity=1000, mini_size=10, batch_size=16, eps_start=0.0,
                     eps_end=0.0, eps_decay=1, device="cpu", target_clip_max=0.0, target_clip_min=clip_min)

def _fill(agent, reward):
    rng = np.random.default_rng(0)
    for _ in range(40):
        s = rng.random(H + 12 * SD).astype(np.float32); ns = rng.random(H + 12 * SD).astype(np.float32)
        agent.replay_buffer.add(s, int(rng.integers(K)), reward, ns, False, "J")

def test_default_none_attribute_and_dqn_accepts_kwarg():
    a = _agent(None); assert a.target_clip_min is None and a.target_clip_max == 0.0
    d = DQN(starting_state=(0.0,) * 8, state_space=8, hidden_dim=16, action_space=4, learning_rate=1e-3, gamma=0.95,
            epsilon=0.1, target_update=10, capacity=100, mini_size=5, batch_size=4, eps_start=0.5, eps_end=0.01,
            eps_decay=100, device="cpu", target_clip_min=-50.0)
    assert d.target_clip_min == -50.0

def test_clamp_bounds_huber_loss_under_catastrophic_rewards():
    torch.manual_seed(0); a = _agent(None); _fill(a, reward=-500.0); a.learn_step(); loss_free = a.loss
    torch.manual_seed(0); b = _agent(-50.0); b.q_net.load_state_dict(a.q_net.state_dict()); b.target_q_net.load_state_dict(a.target_q_net.state_dict())
    _fill(b, reward=-500.0); b.learn_step(); loss_clamped = b.loss
    # 目标被截到 −50, 而 Q≈0: Huber (线性段) ≈ |tgt| − 0.5 → 约 49.5; 不截时 ≈ 500 量级
    assert 45.0 < loss_clamped < 55.0, loss_clamped
    assert loss_free > 400.0, loss_free
