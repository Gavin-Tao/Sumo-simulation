"""F1 动态最短绿掩码 (2026-10-12, 用户批准): FRAPAgent 侧。
默认 (不给 mask / next_mask) 与旧版逐位相同; 给了 mask 时选动作只在交集内; 回放 7 元组时 TD 目标的 max 只在 s' 可执行相位上取。"""
import os, sys, random, numpy as np, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
import torch.nn.functional as F
from sumo_rl.agents.frap_agent import FRAPAgent, FRAPReplayBuffer, NEG

H, SD, K = 2, 27, 4


def _tables():
    pm = np.zeros((11, 12), np.float32)
    for k in range(K):
        pm[k, k] = 1
    rel = np.full((12, 12), -1, np.int64); rel[:4, :4] = 3; np.fill_diagonal(rel, 0)
    exist = np.zeros(12, np.float32); exist[:4] = 1
    mask = np.array([True] * K + [False] * (11 - K))
    return {"J": dict(pm=pm, rel=rel, exist=exist, mask=mask)}


def _agent(eps=0.0, gamma=0.95):
    return FRAPAgent(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors=_tables(), lr=1e-3, gamma=gamma,
                     epsilon=eps, target_update=10**9, capacity=1000, mini_size=10, batch_size=16, eps_start=eps,
                     eps_end=eps, eps_decay=1, device="cpu")


def test_buffer_default_six_tuple_and_seven_with_mask():
    b = FRAPReplayBuffer(10)
    s = np.zeros(3, np.float32)
    b.add(s, 1, -1.0, s, False, "J")
    assert len(b.buffer[0]) == 6 and len(b.sample(1)) == 6
    c = FRAPReplayBuffer(10)
    c.add(s, 1, -1.0, s, False, "J", next_mask=[True, False, False])
    out = c.sample(1)
    assert len(out) == 7 and out[6].dtype == bool and out[6].tolist() == [[True, False, False]]


def test_take_action_default_unchanged_and_mask_restricts():
    torch.manual_seed(0); a = _agent(eps=0.0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.random(H + 12 * SD).astype(np.float32)
        with torch.no_grad():
            pm, rel, exist, m = a._tensors(torch.tensor([0]))
            q = a.q_net(torch.tensor(x).unsqueeze(0), pm, rel, exist).masked_fill(~m, NEG)
        assert a.take_action(x, "J") == int(q.argmax())               # 默认: 与旧的固定掩码 argmax 相同
        only = int(rng.integers(K)); mk = np.zeros(11, bool); mk[only] = True
        assert a.take_action(x, "J", mask=mk) == only                  # 贪心: 只剩当前相位
    b = _agent(eps=1.0)
    mk = np.zeros(11, bool); mk[2] = True
    assert all(b.take_action(np.zeros(H + 12 * SD, np.float32), "J", mask=mk) == 2 for _ in range(50))   # ε 探索也受限
    seen = {b.take_action(np.zeros(H + 12 * SD, np.float32), "J") for _ in range(200)}
    assert seen == set(range(K))                                        # 默认 ε 仍覆盖全部存在相位


def _fill_same(agent, s, a, r, ns, next_mask):
    for _ in range(40):
        if next_mask is None:
            agent.replay_buffer.add(s, a, r, ns, False, "J")
        else:
            agent.replay_buffer.add(s, a, r, ns, False, "J", next_mask=next_mask)


def _manual_loss(agent, s, a, r, ns, allowed):
    with torch.no_grad():
        pm, rel, exist, m = agent._tensors(torch.tensor([0]))
        q = agent.q_net(torch.tensor(s).unsqueeze(0), pm, rel, exist)[0, a]
        valid = m[0].clone(); valid &= torch.tensor(allowed)
        if agent.use_double:
            nq = agent.q_net(torch.tensor(ns).unsqueeze(0), pm, rel, exist)[0].masked_fill(~valid, NEG)
            mnq = agent.target_q_net(torch.tensor(ns).unsqueeze(0), pm, rel, exist)[0, int(nq.argmax())]
        else:
            mnq = agent.target_q_net(torch.tensor(ns).unsqueeze(0), pm, rel, exist)[0].masked_fill(~valid, NEG).max()
        tgt = r + agent.gamma * mnq
        if agent.target_clip_max is not None:
            tgt = tgt.clamp(max=agent.target_clip_max)
        return float(F.smooth_l1_loss(q, tgt) if agent.loss_fn == "huber" else F.mse_loss(q, tgt))


def test_learn_step_target_uses_next_mask():
    rng = np.random.default_rng(2)
    s = rng.random(H + 12 * SD).astype(np.float32); ns = rng.random(H + 12 * SD).astype(np.float32)
    torch.manual_seed(0); a = _agent()
    # 让目标网各相位 Q 明显不同, 掩码是否生效才看得出
    with torch.no_grad():
        for p in a.target_q_net.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    allowed_all = np.ones(11, bool)
    for only in range(K):
        allowed = np.zeros(11, bool); allowed[only] = True
        b = _agent(); b.q_net.load_state_dict(a.q_net.state_dict()); b.target_q_net.load_state_dict(a.target_q_net.state_dict())
        expect = _manual_loss(b, s, 1, -3.0, ns, allowed)
        _fill_same(b, s, 1, -3.0, ns, allowed); b.learn_step()
        assert abs(b.loss - expect) < 1e-5, (only, b.loss, expect)
    # 全 1 动态掩码 == 不给掩码 (旧路径)
    c = _agent(); c.q_net.load_state_dict(a.q_net.state_dict()); c.target_q_net.load_state_dict(a.target_q_net.state_dict())
    d = _agent(); d.q_net.load_state_dict(a.q_net.state_dict()); d.target_q_net.load_state_dict(a.target_q_net.state_dict())
    _fill_same(c, s, 1, -3.0, ns, None); _fill_same(d, s, 1, -3.0, ns, allowed_all)
    random.seed(0); c.learn_step(); random.seed(0); d.learn_step()
    assert abs(c.loss - d.loss) < 1e-7
    assert abs(c.loss - _manual_loss(a, s, 1, -3.0, ns, allowed_all)) < 1e-5
