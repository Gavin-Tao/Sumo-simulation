"""all_connections (2026-10-26, 任务 2): PriorityMovement 建表遍历每个信号索引下的全部连接。
默认 False 逐位不变 (只取第一条连接, 旧行为); True 时都柏林 389332 的 E/W 直行槽容量加倍、出口与供车车道各两条,
133307283 的 N 直行出口两条; 其余 16 个路口两种取值下表相同; 1x1 网两种取值全等。需要 SUMO。
运行: python -m pytest experiments/tests/test_obs_all_connections.py -q"""
import os, sys, json, functools, yaml, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
CFG457 = os.path.join(_REPO, "experiments/configs/exp457_dublin18h_reroute_v2_amb11h_541_8std_slot12_f1.yaml")
META = os.path.join(_REPO, "nets/dublin/dublin_8std_meta.json")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")
pytestmark = pytest.mark.skipif(not os.path.exists(CFG457), reason="配置缺失")
E_T, W_T, N_T = 1 * 3 + 1, 3 * 3 + 1, 0 * 3 + 1          # 槽号 = 方向 (N,E,S,W) × 3 + 转向 (L,T,R)


def _tables(cfg_path, all_connections):
    """构造环境, 按 meta 重绑运动槽, 返回 {路口: (容量, 出口集合, 供车车道集合)} 与 sumo 句柄的车道长度表。"""
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from exp_params import obs_kwargs_from_cfg, turnmap_from_meta
    cfg = yaml.safe_load(open(cfg_path)); os.chdir(os.path.join(_REPO, "experiments"))
    kw = obs_kwargs_from_cfg(cfg); kw["all_connections"] = all_connections
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                          num_seconds=20, min_green=5, single_agent=False, yellow_time=2, delta_time=5, reward_fn="avg-waiting-time",
                          observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw), sumo_seed=0, sumo_warnings=False)
    env.reset(0)
    meta = json.load(open(os.path.join(_REPO, "experiments", cfg["action_meta_file"])))["tls"]
    out, lanelen = {}, {}
    for tid in env.ts_ids:
        of = env.traffic_signals[tid].observation_fn; of.rebind_movements(turnmap_from_meta(meta[tid]))
        out[tid] = (dict(of._slot_cap), {s: set(v) for s, v in of._slot_out.items()}, {s: set(v) for s, v in of._slot_in_lanes.items()})
        for l in env.traffic_signals[tid].lanes: lanelen[l] = env.sumo.lane.getLength(l)
    env.close(); return out, lanelen


@pytest.fixture(scope="module")
def dublin():
    return _tables(CFG457, False), _tables(CFG457, True)


def test_default_keeps_first_connection_only(dublin):
    (old, ll), _ = dublin
    cap, outs, ins = old["389332"]
    assert cap[E_T] == pytest.approx(ll["-304634278#0_0"] / 7.5)
    assert outs[E_T] == {"-3791745#5_0"} and ins[E_T] == {"-304634278#0_0"}
    assert old["133307283"][1][N_T] == {"77616869#0_0"}


def test_all_connections_registers_second_lane_at_389332(dublin):
    (_, ll), (new, _) = dublin
    cap, outs, ins = new["389332"]
    assert cap[E_T] == pytest.approx((ll["-304634278#0_0"] + ll["-304634278#0_1"]) / 7.5)
    assert cap[W_T] == pytest.approx((ll["3791745#4_0"] + ll["3791745#4_1"]) / 7.5)
    assert outs[E_T] == {"-3791745#5_0", "-3791745#5_1"} and outs[W_T] == {"304634278#0_0", "304634278#0_1"}
    assert ins[E_T] == {"-304634278#0_0", "-304634278#0_1"} and ins[W_T] == {"3791745#4_0", "3791745#4_1"}


def test_all_connections_registers_second_exit_lane_at_133307283(dublin):
    (old, _), (new, _) = dublin
    assert new["133307283"][1][N_T] == {"77616869#0_0", "77616869#0_1"}
    assert new["133307283"][0] == old["133307283"][0]          # 进口容量不变 (两条连接同一进口车道)
    assert new["133307283"][2] == old["133307283"][2]


def test_other_16_junctions_identical(dublin):
    (old, _), (new, _) = dublin
    for tid in old:
        if tid in ("389332", "133307283"):
            continue
        assert old[tid] == new[tid], tid


@pytest.mark.skipif(not os.path.exists(CFG274), reason="1x1 配置缺失")
def test_grid_1x1_identical_both_ways():
    old, _ = _tables(CFG274, False); new, _ = _tables(CFG274, True)
    assert old == new
