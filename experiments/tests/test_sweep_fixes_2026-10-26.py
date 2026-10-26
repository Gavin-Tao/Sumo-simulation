"""2026-10-26 复查补修 (用户令 "重新检查, 确保没有遗漏"):
(1) DQN 可选初始化同步目标网络 sync_target_init (默认 False 逐位不变; FRAP/多头智能体本来就同步, 主线 DQN 没有);
(2) exp_params.is_sumo_error: 训练循环的 train_strict_errors 开关只把 SUMO/TraCI 异常当作 "重置继续", 其它异常重新抛出;
(3) 环境 info["step"]: reset 清零, 推进后再写入 (原来 reset 不清零且晚一个决策步)。
运行: python -m pytest experiments/tests/test_sweep_fixes_2026-10-26.py -q"""
import os, sys, functools, yaml, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")


def _dqn(**kw):
    import torch
    from sumo_rl.agents.dqn_agent_txw import DQN
    torch.manual_seed(0)
    return DQN(starting_state=tuple([0.0] * 6), state_space=6, hidden_dim=8, action_space=4, learning_rate=1e-3, gamma=0.95, epsilon=0.1,
               target_update=10, capacity=100, mini_size=10, batch_size=4, eps_start=0.9, eps_end=0.05, eps_decay=100, device="cpu", **kw)


def _same_params(a, b):
    import torch
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def test_dqn_target_sync_at_init_is_optional_and_default_off():
    agent = _dqn()
    assert not _same_params(agent.q_net, agent.target_q_net)          # 旧行为: 两套随机初始化
    agent = _dqn(sync_target_init=True)
    assert _same_params(agent.q_net, agent.target_q_net)              # 新行为: 初始化即同步


def test_is_sumo_error_distinguishes_traci_errors():
    from exp_params import is_sumo_error
    libsumo = pytest.importorskip("libsumo")
    assert is_sumo_error(libsumo.TraCIException("x")) is True
    assert is_sumo_error(libsumo.FatalTraCIError("x")) is True
    assert is_sumo_error(ValueError("x")) is False
    assert is_sumo_error(KeyError("x")) is False
    assert is_sumo_error(RuntimeError("x")) is False


@pytest.mark.skipif(not os.path.exists(CFG274), reason="1x1 配置缺失")
def test_info_step_resets_per_episode_and_counts_elapsed_seconds():
    pytest.importorskip("libsumo")
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from exp_params import obs_kwargs_from_cfg
    cfg = yaml.safe_load(open(CFG274)); os.chdir(os.path.join(_REPO, "experiments"))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                          num_seconds=100, min_green=5, single_agent=False, yellow_time=2, delta_time=5, reward_fn="avg-waiting-time",
                          observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **obs_kwargs_from_cfg(cfg)),
                          sumo_seed=0, sumo_warnings=False)
    env.reset(0); t = env.ts_ids[0]
    _, _, _, info = env.step(action={t: 0}); assert info["step"] == 5
    for _ in range(2):
        _, _, _, info = env.step(action={t: 0})
    assert info["step"] == 15
    env.reset(0)
    _, _, _, info = env.step(action={t: 0}); assert info["step"] == 5
    env.close()
