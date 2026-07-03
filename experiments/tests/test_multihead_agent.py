"""Hand-computed unit tests for the multi-head B-line agent + reward glue.
Run: pytest experiments/tests/test_multihead_agent.py"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from sumo_rl.agents.dqn_multihead_agent import DQNMultiHead, N_LEVELS


def _agent(action_space=3, use_double=True, target_clip_max=None):
    return DQNMultiHead(
        starting_state=(0.0,) * 4, state_space=4, hidden_dim=8,
        action_space=action_space, learning_rate=1e-3, gamma=0.5,
        epsilon=0.0, target_update=10, capacity=100, mini_size=1,
        batch_size=4, eps_start=0, eps_end=0, eps_decay=1, device="cpu",
        use_double=use_double, loss_fn="huber",
        target_clip_max=target_clip_max)


class _Fake(torch.nn.Module):
    """Returns a fixed (B, 5, A) tensor regardless of input."""
    def __init__(self, q):
        super().__init__()
        self.q = torch.tensor(q, dtype=torch.float)

    def forward(self, x):
        return self.q.expand(x.shape[0], -1, -1)


def test_take_action_weighted_argmax_and_mask():
    ag = _agent(action_space=3)
    # heads (5,3): action scores = w·Q per column, w = [1..5]
    q = np.zeros((1, 5, 3))
    q[0, 0] = [10, 0, 0]     # l1 loves a0
    q[0, 4] = [0, 3, 0]      # l5 prefers a1
    # composite: a0 = 1*10 = 10; a1 = 5*3 = 15; a2 = 0 -> a1 wins
    ag.q_net = _Fake(q)
    assert ag.take_action([0, 0, 0, 0]) == 1
    # mask out a1 -> falls to a0
    assert ag.take_action([0, 0, 0, 0], mask=np.array([True, False, True])) == 0


def test_targets_joint_policy_hand_math():
    """Double DQN: a* from ONLINE weighted score (masked); every head is
    evaluated by the TARGET net at that same a*."""
    ag = _agent(action_space=2, use_double=True)   # gamma = 0.5
    online = np.zeros((1, 5, 2))
    online[0, 4] = [0.0, 2.0]      # online composite: a0=0, a1=10 -> a*=1
    target = np.zeros((1, 5, 2))
    target[0, 0] = [7.0, -1.0]     # head1 value at a1 = -1
    target[0, 4] = [9.0, -2.0]     # head5 value at a1 = -2
    ag.q_net, ag.target_q_net = _Fake(online), _Fake(target)
    rewards = torch.zeros((1, 5)); rewards[0, 0] = -0.4
    y = ag._targets(rewards, torch.zeros((1, 4)), torch.zeros((1, 1)))
    # y_l = r_l + 0.5 * target_l[a*=1]
    exp = np.zeros((1, 5)); exp[0, 0] = -0.4 + 0.5 * -1.0; exp[0, 4] = 0.5 * -2.0
    assert np.allclose(y.numpy(), exp), y
    # mask a1 out -> a*=0; head values switch to column 0 (7.0 / 9.0)
    y2 = ag._targets(rewards, torch.zeros((1, 4)), torch.zeros((1, 1)),
                     next_mask_t=torch.tensor([[True, False]]))
    exp2 = np.zeros((1, 5)); exp2[0, 0] = -0.4 + 0.5 * 7.0; exp2[0, 4] = 0.5 * 9.0
    assert np.allclose(y2.numpy(), exp2), y2
    # done kills bootstrap; clip caps per head
    y3 = ag._targets(rewards, torch.zeros((1, 4)), torch.ones((1, 1)))
    assert np.allclose(y3.numpy(), rewards.numpy())
    ag.target_clip_max = 0.0
    y4 = ag._targets(torch.ones((1, 5)), torch.zeros((1, 4)),
                     torch.ones((1, 1)))
    assert np.allclose(y4.numpy(), np.zeros((1, 5)))


def test_vanilla_targets_use_target_scores():
    ag = _agent(action_space=2, use_double=False)
    # target net picks a* by ITS OWN weighted score; online net irrelevant
    target = np.zeros((1, 5, 2))
    target[0, 0] = [1.0, 0.0]      # composite: a0=1 > a1=0 -> a*=0
    ag.q_net = _Fake(np.zeros((1, 5, 2)))
    ag.target_q_net = _Fake(target)
    y = ag._targets(torch.zeros((1, 5)), torch.zeros((1, 4)),
                    torch.zeros((1, 1)))
    exp = np.zeros((1, 5)); exp[0, 0] = 0.5 * 1.0
    assert np.allclose(y.numpy(), exp)


def test_update_end_to_end_with_vector_rewards():
    ag = _agent(action_space=3)
    rng = np.random.RandomState(0)
    for _ in range(8):
        ag.replay_buffer.add(rng.rand(4), int(rng.randint(3)),
                             -rng.rand(N_LEVELS), rng.rand(4), False,
                             next_mask=np.array([True, True, False]))
    b_s, b_a, b_r, b_ns, b_d, b_m = ag.replay_buffer.sample(4)
    ag.update({"states": b_s, "actions": b_a, "rewards": b_r,
               "next_states": b_ns, "dones": b_d, "next_masks": b_m})
    assert ag.loss is not None and np.isfinite(ag.loss)
    assert ag.count == 1 and ag.q_mean is not None
    assert isinstance(ag.take_action([0.1] * 4,
                                     mask=np.array([True, False, True])), int)


def test_reward_vec_exact_decomposition():
    """scalar == Σ_l l·vec[l-1], hand-computed on a mocked junction."""
    from multihead_glue import make_priority_avg_waiting_reward_vec
    fn, cache = make_priority_avg_waiting_reward_vec(
        {"ambulance": 5, "bus": 3, "car": 1})
    ts = MagicMock()
    ts.id = "T"
    ts.lanes = ["L0", "L1"]
    ts.env.vehicles = {}
    sumo = ts.sumo
    sumo.lane.getLastStepVehicleIDs.side_effect = \
        lambda lane: ["c1", "c2"] if lane == "L0" else ["a1"]
    sumo.vehicle.getLaneID.side_effect = lambda v: "L0" if v.startswith("c") else "L1"
    sumo.vehicle.getAccumulatedWaitingTime.side_effect = \
        lambda v: {"c1": 20.0, "c2": 40.0, "a1": 10.0}[v]
    sumo.vehicle.getTypeID.side_effect = \
        lambda v: "ambulance" if v == "a1" else "car"
    r = fn(ts)
    # car avg = 30 -> vec[0] = -0.30 ; amb avg = 10 -> vec[4] = -0.10
    vec = cache["T"]
    assert abs(vec[0] + 0.30) < 1e-12 and abs(vec[4] + 0.10) < 1e-12
    assert abs(vec[1]) + abs(vec[2]) + abs(vec[3]) == 0.0
    assert abs(r - (-(1 * 30 + 5 * 10) / 100.0)) < 1e-12
    # the identity the whole design rests on:
    assert abs(r - sum((l + 1) * vec[l] for l in range(5))) < 1e-12
