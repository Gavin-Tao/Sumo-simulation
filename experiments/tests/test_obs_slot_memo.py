"""slot_memo (2026-10-26, 任务 3): PriorityMovement 的车辆→槽缓存可关闭。默认 True 逐位不变 (按车辆 ID 整集缓存);
False 时每次按 (进口边, 路线下一条边) 重查。用 SUMO 的 changeTarget 让一辆在进口道上的车换去向: True 时槽不变 (旧缓存),
False 时槽随新路线变。无路线变化时两种取值的观测流逐位相同。需要 SUMO。
运行: python -m pytest experiments/tests/test_obs_slot_memo.py -q"""
import os, sys, json, functools, yaml, numpy as np, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")
pytestmark = pytest.mark.skipif(not os.path.exists(CFG274), reason="1x1 配置缺失")


def _env(slot_memo, num_seconds=400):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from exp_params import obs_kwargs_from_cfg, turnmap_from_meta
    cfg = yaml.safe_load(open(CFG274)); os.chdir(os.path.join(_REPO, "experiments"))
    kw = obs_kwargs_from_cfg(cfg); kw["slot_memo"] = slot_memo
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                          num_seconds=num_seconds, min_green=5, single_agent=False, yellow_time=2, delta_time=5, reward_fn="avg-waiting-time",
                          observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw), sumo_seed=0, sumo_warnings=False)
    env.reset(0)
    meta = json.load(open(os.path.join(_REPO, "experiments", cfg["action_meta_file"])))["tls"]
    t = env.ts_ids[0]; of = env.traffic_signals[t].observation_fn; of.rebind_movements(turnmap_from_meta(meta[t]))
    return env, t, of


def _pick_vehicle(env, t, of):
    """找一辆在进口道上、还有下一条边、且本路口还有另一条可达出口边的车; 返回 (车辆, 进口边, 原下一条边, 新目标边)。"""
    sumo = env.sumo
    for lane in env.traffic_signals[t].lanes:
        e = of._edge(lane)
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            r = sumo.vehicle.getRoute(vid); k = sumo.vehicle.getRouteIndex(vid)
            if k + 1 >= len(r):
                continue
            nxt = r[k + 1]
            others = sorted({to for (fr, to), s in of._movement_by_edges.items() if fr == e and to != nxt})
            if others:
                return vid, e, nxt, others[0]
    return None


def _run_until_pick(slot_memo):
    env, t, of = _env(slot_memo); pick = None
    for i in range(60):
        env.step(action={t: (i // 3) % 4})
        pick = _pick_vehicle(env, t, of)
        if pick:
            break
    assert pick, "400 秒内没找到合适的车"
    return env, t, of, pick


def test_memo_keeps_stale_slot_after_route_change():
    env, t, of, (vid, e, nxt, new_to) = _run_until_pick(True)
    before = of._vehicle_slot(vid, e); assert before == of._movement_by_edges[(e, nxt)]
    env.sumo.vehicle.changeTarget(vid, new_to)
    assert env.sumo.vehicle.getRoute(vid)[env.sumo.vehicle.getRouteIndex(vid) + 1] == new_to
    assert of._vehicle_slot(vid, e) == before                       # 旧行为: 缓存不更新
    env.close()


def test_no_memo_follows_current_route():
    env, t, of, (vid, e, nxt, new_to) = _run_until_pick(False)
    before = of._vehicle_slot(vid, e)
    env.sumo.vehicle.changeTarget(vid, new_to)
    after = of._vehicle_slot(vid, e)
    assert after == of._movement_by_edges[(e, new_to)] and after != before
    assert of._vid_slot == {}                                        # 关闭缓存时不写字典
    env.close()


def test_streams_identical_without_route_changes():
    seqs = []
    for memo in (True, False):
        env, t, of = _env(memo, num_seconds=300); seq = []
        for i in range(50):
            s, _, _, _ = env.step(action={t: (i // 3) % 4}); seq.append(np.asarray(s[t], dtype=np.float32))
        env.close(); seqs.append(np.stack(seq))
    assert np.array_equal(seqs[0], seqs[1])
