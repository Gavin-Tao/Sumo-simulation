"""TrafficSignal.out_lanes 全连接 (2026-10-26, 任务 7): 出口车道列表原来只取每个信号索引第一条连接的 to-lane (与观测建表同一遗漏)。
环境参数 all_connections (默认 False 逐位不变) 透传给 TrafficSignal; True 时收齐全部连接的出口车道并去重。
只被 get_out_lanes_density 与旧观测类使用, PriorityMovement 不用。需要 SUMO。运行: python -m pytest experiments/tests/test_out_lanes_all_connections.py -q"""
import os, sys, functools, yaml, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
CFG457 = os.path.join(_REPO, "experiments/configs/exp457_dublin18h_reroute_v2_amb11h_541_8std_slot12_f1.yaml")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")
pytestmark = pytest.mark.skipif(not (os.path.exists(CFG457) and os.path.exists(CFG274)), reason="配置缺失")


def _out_lanes(cfg_path, **env_kw):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from exp_params import obs_kwargs_from_cfg
    cfg = yaml.safe_load(open(cfg_path)); os.chdir(os.path.join(_REPO, "experiments"))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                          num_seconds=20, min_green=5, single_agent=False, yellow_time=2, delta_time=5, reward_fn="avg-waiting-time",
                          observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **obs_kwargs_from_cfg(cfg)),
                          sumo_seed=0, sumo_warnings=False, **env_kw)
    env.reset(0); out = {t: list(env.traffic_signals[t].out_lanes) for t in env.ts_ids}; env.close(); return out


def test_default_keeps_first_connection_only():
    out = _out_lanes(CFG457)
    assert out["389332"] == ["4919462#0_0", "-3791745#5_0", "304634278#0_0", "4919462#0_1"]


def test_all_connections_adds_missing_exit_lanes_without_duplicates():
    out = _out_lanes(CFG457, all_connections=True)
    assert set(out["389332"]) == {"4919462#0_0", "-3791745#5_0", "-3791745#5_1", "304634278#0_0", "304634278#0_1", "4919462#0_1"}
    assert len(out["389332"]) == len(set(out["389332"]))
    assert "77616869#0_1" in out["133307283"] and len(out["133307283"]) == len(set(out["133307283"]))
    old = _out_lanes(CFG457)
    for t in old:
        if t not in ("389332", "133307283"):
            assert old[t] == out[t], t


def test_grid_1x1_identical_both_ways():
    assert _out_lanes(CFG274) == _out_lanes(CFG274, all_connections=True)
