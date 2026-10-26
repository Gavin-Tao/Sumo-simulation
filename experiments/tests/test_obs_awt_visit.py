"""awt_basis=visit (2026-10-26, 任务 5): PriorityMovement 的 mean_awt 改读逐秒访问统计器。
visit 口径下每槽每等级 mean_awt == 按统计器独立聚合的值; local 口径不碰统计器 (有无统计器输出相同);
visit 与 local 的逐车差只来自进入到第一次采样之间的等待 (0..5 秒); visit 配下游 mean_awt 报错; visit 下 _awt_entry 恒空。需要 SUMO。
运行: python -m pytest experiments/tests/test_obs_awt_visit.py -q"""
import os, sys, json, math, functools, yaml, numpy as np, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.environ.setdefault("SUMO_RL_LIBSUMO", "1"); pytest.importorskip("libsumo")
CFG274 = os.path.join(_REPO, "experiments/configs/exp274_1x1_std_8std_full_g099_perphase.yaml")
pytestmark = pytest.mark.skipif(not os.path.exists(CFG274), reason="1x1 配置缺失")
TABLE = {"ambulance": 5, "bus": 4, "car": 1}; LOG = math.log1p(3600.0)


def _env(awt_basis, wait_tracker, awt_log=True, downstream_fields=("count", "queue"), include_downstream=False, num_seconds=400):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from exp_params import turnmap_from_meta
    cfg = yaml.safe_load(open(CFG274)); os.chdir(os.path.join(_REPO, "experiments"))
    kw = dict(fields=("count", "queue", "mean_awt"), phase_state="perphase", priority_source=TABLE, awt_basis=awt_basis, awt_log=awt_log,
              include_downstream=include_downstream, downstream_fields=downstream_fields)
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                          num_seconds=num_seconds, min_green=5, single_agent=False, yellow_time=2, delta_time=5, reward_fn="avg-waiting-time",
                          observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw), sumo_seed=0, sumo_warnings=False,
                          waiting_time_memory=100000, wait_tracker=wait_tracker)
    s = env.reset(0); t = env.ts_ids[0]; of = env.traffic_signals[t].observation_fn
    meta = json.load(open(os.path.join(_REPO, "experiments", cfg["action_meta_file"])))["tls"]; of.rebind_movements(turnmap_from_meta(meta[t]))
    return env, t, of


def _mean_awt_pos(slot, level):
    """布局: 头部 2 + 每槽 16 = is_green + 5 等级 × (count, queue, mean_awt)。"""
    return 2 + 16 * slot + 1 + 3 * (level - 1) + 2


def test_visit_mean_awt_equals_independent_aggregation():
    env, t, of = _env("visit", True); sumo = env.sumo; tr = env.visit_tracker; checked = 0
    for i in range(60):
        s, _, _, _ = env.step(action={t: (i // 3) % 4}); obs = np.asarray(s[t], dtype=np.float64)
        sums, cnts = {}, {}
        for lane in env.traffic_signals[t].lanes:
            e = of._edge(lane)
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                slot = of._vehicle_slot(vid, e)
                if slot < 0:
                    continue
                k = (slot, TABLE[sumo.vehicle.getTypeID(vid)]); w = tr.get(t, vid, lane)
                sums[k] = sums.get(k, 0.0) + w; cnts[k] = cnts.get(k, 0) + 1
        for slot in range(12):
            for p in (1, 2, 3, 4, 5):
                mean = sums.get((slot, p), 0.0) / cnts[(slot, p)] if (slot, p) in cnts else 0.0
                assert abs(obs[_mean_awt_pos(slot, p)] - math.log1p(mean) / LOG) < 1e-6; checked += 1
    env.close(); assert checked == 60 * 60


def test_local_basis_ignores_tracker():
    seqs = []
    for wt in (False, True):
        env, t, of = _env("local", wt, num_seconds=300); seq = []
        for i in range(50):
            s, _, _, _ = env.step(action={t: (i // 3) % 4}); seq.append(np.asarray(s[t], dtype=np.float32))
        if wt:
            assert env.visit_tracker is not None and env.visit_tracker.missing == 0
        env.close(); seqs.append(np.stack(seq))
    assert np.array_equal(seqs[0], seqs[1])


def test_visit_minus_local_is_only_the_entry_window():
    env, t, of = _env("local", True); sumo = env.sumo; tr = env.visit_tracker; diffs = []
    for i in range(60):
        env.step(action={t: (i // 3) % 4})
        for lane in env.traffic_signals[t].lanes:
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                acc = sumo.vehicle.getAccumulatedWaitingTime(vid)
                local = acc - of._awt_entry[(vid, lane)]          # 旧口径: 第一次被观测到时的累计等待作起点
                diffs.append(tr.get(t, vid, lane) - local)
    env.close(); d = np.asarray(diffs)
    assert len(d) > 2000 and d.min() >= 0.0 and d.max() <= 5.0 and (d == 0).mean() > 0.5


def test_visit_with_downstream_mean_awt_raises():
    with pytest.raises(ValueError):
        _env("visit", True, include_downstream=True, downstream_fields=("count", "queue", "mean_awt"))


def test_visit_keeps_awt_entry_empty():
    env, t, of = _env("visit", True)
    for i in range(30):
        env.step(action={t: (i // 3) % 4})
    assert of._awt_entry == {} and env.visit_tracker.missing == 0; env.close()
