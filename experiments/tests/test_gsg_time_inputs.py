"""GS-G time_inputs (2026-10-13): "none" = 代价项只读运动编码 e_i, 不要头部与逐运动放行时长, 编码器读完整的槽。
检验: 构造不报错且代价网络输入维 = 嵌入维; 代价关掉时等于读完整槽的 FRAP 核心; 当前绿灯集合对应的相位无代价; 半切换比整体换相便宜;
代价可求梯度; FRAPAgent 无头部时可用 arch gsg; 非法取值报错。默认 "full" 的行为由 test_gsg_qnet.py 覆盖。"""
import os, sys, numpy as np, torch, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from sumo_rl.agents.gsg_qnet import GSGQNet
from sumo_rl.agents.frap_agent import FRAPAgent
H, SD, K = 0, 17, 6        # 无头部; 槽 = is_green + φ15 + 槽内 min_green_ok


def _tables():
    pm = torch.zeros(1, K, 12)
    pm[0, 0, [0, 1, 6, 7]] = 1; pm[0, 1, [0, 2, 6]] = 1; pm[0, 2, [3, 4, 9, 10]] = 1; pm[0, 3, [3, 5]] = 1
    rel = torch.randint(0, 4, (1, 12, 12), generator=torch.Generator().manual_seed(0)); rel[0].fill_diagonal_(0)
    return pm.expand(2, -1, -1), rel.expand(2, -1, -1), torch.ones(2, 12)


def _x(cur_row):
    torch.manual_seed(3); x = torch.rand(2, H + 12 * SD); x[:, H::SD] = cur_row
    return x


def _net(**kw):
    torch.manual_seed(0); return GSGQNet(H, SD, k_max=K, gt_in_slot=False, time_inputs="none", **kw)


def test_builds_without_header_and_cost_inputs_are_embedding_only():
    net = _net()
    assert net.kappa_cut[0].in_features == 16 and net.kappa_new[0].in_features == 16 and net.core_slot == SD


def test_costs_off_equals_core_on_full_slot():
    net = _net(); P, R, E = _tables(); pm0 = P[0, 0]; x = _x(pm0)
    for m in (net.kappa_cut, net.kappa_new): torch.nn.init.constant_(m[2].bias, -40.0); torch.nn.init.zeros_(m[2].weight)
    assert torch.allclose(net(x, P, R, E), net.core(x, P, R, E), atol=1e-6)


def test_current_phase_free_and_partial_switch_cheaper():
    net = _net(); P, R, E = _tables(); x = _x(P[0, 0])
    d = net.core(x, P, R, E) - net(x, P, R, E)
    assert torch.allclose(d[:, 0], torch.zeros(2), atol=1e-7)
    assert (d[:, 1] < d[:, 2]).all()


def test_grad_reaches_costs_and_agent_plumbing():
    net = _net(); P, R, E = _tables(); net(_x(P[0, 1]), P, R, E).sum().backward()
    assert net.kappa_cut[0].weight.grad is not None and net.kappa_new[0].weight.grad is not None
    pmn = np.zeros((K, 12), np.float32); pmn[0, 0] = 1; pmn[1, 1] = 1
    reln = np.full((12, 12), -1, np.int64); reln[:2, :2] = 3; np.fill_diagonal(reln, 0)
    exn = np.zeros(12, np.float32); exn[:2] = 1; mask = np.array([True] * 2 + [False] * (K - 2))
    kw = dict(obs_dim=H + 12 * SD, header_dim=H, slot_dim=SD, tls_tensors={"J": dict(pm=pmn, rel=reln, exist=exn, mask=mask)}, lr=1e-3, gamma=0.95,
              epsilon=0.0, target_update=10, capacity=100, mini_size=10, batch_size=8, eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu", k_max=K)
    a = FRAPAgent(**kw, arch="gsg", gsg_kw=dict(gt_in_slot=False, time_inputs="none"))
    assert isinstance(a.q_net, GSGQNet) and a.q_net.time_inputs == "none"
    assert a.take_action(np.random.default_rng(0).random(H + 12 * SD).astype(np.float32), "J") in (0, 1)


def test_invalid_time_inputs_rejected():
    with pytest.raises(ValueError):
        GSGQNet(H, SD, gt_in_slot=False, time_inputs="partial")
