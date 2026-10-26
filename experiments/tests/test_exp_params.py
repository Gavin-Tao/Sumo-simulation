"""exp_params (2026-10-26, 状态/奖励修正计划任务 1): train.py / ckpt_env.py / eval_unified.py 共用的配置→参数构造。
不需要 SUMO。运行: python -m pytest experiments/tests/test_exp_params.py -q"""
import os, sys, json, copy, yaml, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
CFG457 = os.path.join(_REPO, "experiments/configs/exp457_dublin18h_reroute_v2_amb11h_541_8std_slot12_f1.yaml")
META = os.path.join(_REPO, "nets/dublin/dublin_8std_meta.json")


def _cfg457():
    return yaml.safe_load(open(CFG457))


def test_obs_kwargs_exp457_includes_keys_eval_unified_used_to_drop():
    from exp_params import obs_kwargs_from_cfg
    kw = obs_kwargs_from_cfg(_cfg457())
    assert kw == {
        "fields": ("count", "queue", "mean_awt"),
        "phase_state": "perphase",
        "priority_source": {"ambulance": 5, "bus": 4, "car": 1},
        "include_downstream": True,
        "downstream_fields": ("count", "queue"),
        "include_lane_occ": True,
        "awt_cap": 500.0,
        "awt_basis": "local",
        "awt_log": True,            # eval_unified.py 原来漏传的键之一 (共漏 10 个: since_green/awt_log/green_elapsed*/header/slot_min_green/pad_tokens, 另 obs_priority_source 优先规则)
        "include_header": False,
        "include_slot_min_green": True,
    }


def test_obs_kwargs_absent_keys_do_not_appear():
    from exp_params import obs_kwargs_from_cfg
    assert obs_kwargs_from_cfg({}) == {}
    kw = obs_kwargs_from_cfg({"obs_all_connections": True, "obs_slot_memo": False})
    assert kw == {"all_connections": True, "slot_memo": False}


def test_env_kwargs_defaults_and_switches():
    from exp_params import env_kwargs_from_cfg
    assert env_kwargs_from_cfg({}) == {"wait_tracker": False, "all_connections": False}
    assert env_kwargs_from_cfg({"obs_awt_basis": "visit"})["wait_tracker"] is True
    assert env_kwargs_from_cfg({"reward_wait_basis": "visit", "reward_fn": "avg-waiting-time"})["wait_tracker"] is True
    assert env_kwargs_from_cfg({"obs_awt_basis": "local"})["wait_tracker"] is False
    assert env_kwargs_from_cfg({"all_connections": True})["all_connections"] is True


def test_reward_wait_basis_values():
    from exp_params import reward_wait_basis
    assert reward_wait_basis({}) == "legacy"
    assert reward_wait_basis({"reward_wait_basis": "visit", "reward_fn": "priority-avg-waiting"}) == "visit"
    with pytest.raises(ValueError):
        reward_wait_basis({"reward_wait_basis": "bogus"})


def test_turnmap_from_meta_matches_old_expression_for_all_18_junctions():
    from exp_params import turnmap_from_meta
    meta = json.load(open(META))["tls"]
    assert len(meta) == 18
    for tid, t in meta.items():
        old = {int(i): (c[0]["approach"], c[0]["turn"]) for i, c in t["links"].items()}
        assert turnmap_from_meta(t) == old


def test_turnmap_from_meta_rejects_inconsistent_labels_within_one_index():
    from exp_params import turnmap_from_meta
    t = copy.deepcopy(json.load(open(META))["tls"]["389332"])
    t["links"]["1"][1]["turn"] = "L"           # 索引 1 的第二条连接标签改坏
    with pytest.raises(ValueError):
        turnmap_from_meta(t)


def test_reward_wait_basis_visit_rejects_unsupported_reward_fn():
    """审查 Important 1: visit 口径只接了 priority-avg-waiting / avg-waiting-time 两个奖励, 其它奖励名会静默走 legacy → 必须报错。"""
    from exp_params import reward_wait_basis
    assert reward_wait_basis({"reward_wait_basis": "visit", "reward_fn": "priority-avg-waiting"}) == "visit"
    assert reward_wait_basis({"reward_wait_basis": "visit", "reward_fn": "avg-waiting-time"}) == "visit"
    with pytest.raises(ValueError):
        reward_wait_basis({"reward_wait_basis": "visit", "reward_fn": "5-4-1-avg-waiting-time"})
    with pytest.raises(ValueError):
        reward_wait_basis({"reward_wait_basis": "visit"})
    assert reward_wait_basis({"reward_fn": "5-4-1-avg-waiting-time"}) == "legacy"      # legacy 不限制奖励名


def test_assert_tracker_clean():
    """审查 Minor 4 (提级): 训练里每集结束检查统计器从未缺条目, 缺了就响亮失败。"""
    from exp_params import assert_tracker_clean
    class NoTracker:
        visit_tracker = None
    assert_tracker_clean(NoTracker())
    class Tr:
        missing = 0
    class WithTracker:
        visit_tracker = Tr()
    assert_tracker_clean(WithTracker())
    Tr.missing = 3
    with pytest.raises(RuntimeError):
        assert_tracker_clean(WithTracker())
