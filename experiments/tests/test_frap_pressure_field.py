"""frap.pressure_field (2026-10-12): 默认 queue 时 layout 与旧实现逐位相同; count 时取 count 字段 (进口与下游)。只测 layout 计算, 不需要 SUMO。"""
import os, sys, types
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
import frap_glue

def _layout(cfg):
    captured = {}
    import sumo_rl.agents.frap_agent as fa
    class Dummy:
        def __init__(self, **kw): captured.update(kw)
    orig = fa.FRAPAgent; fa.FRAPAgent = Dummy
    try:
        env = types.SimpleNamespace(observation_space=types.SimpleNamespace(shape=(2 + 12 * 27,)))
        tables = {"k_max": 11, "tls": {}, "turnmap": {}}
        frap_glue.build_frap_agent(cfg, tables, env, "cpu")
    finally:
        fa.FRAPAgent = orig
    return captured["layout"]

BASE = dict(obs_phase_state="perphase", obs_fields=["count", "queue", "mean_awt"], obs_downstream=True, obs_downstream_fields=["count", "queue"], obs_lane_occ=True)

def test_default_is_queue_and_unchanged():
    L = _layout(dict(BASE, frap={"score": "pressure"}))
    assert L["queue_idx"] == [1 + k * 3 + 1 for k in range(5)]
    assert L["down_queue_idx"] == [1 + 15 + 1 * 5 + k for k in range(5)]
    assert L == _layout(dict(BASE, frap={"score": "pressure", "pressure_field": "queue"}))

def test_count_field():
    L = _layout(dict(BASE, frap={"score": "pressure", "pressure_field": "count"}))
    assert L["queue_idx"] == [1 + k * 3 + 0 for k in range(5)]
    assert L["down_queue_idx"] == [1 + 15 + 0 * 5 + k for k in range(5)]
