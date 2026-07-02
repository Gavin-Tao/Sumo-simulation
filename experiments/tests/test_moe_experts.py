"""Behavioral unit test for MoE experts (hand-computed reference scores).
Scenario: slot0 = 3 stopped cars (queue), slot1 = 1 ambulance approaching at
10 m/s, 20 m out (ETA 2 s < Δt). Expected stance divergence:
expert-5 switches to serve the amb BEFORE it stops (ETA term);
expert-1 keeps the car phase. Run: pytest experiments/tests/test_moe_experts.py"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from unittest.mock import MagicMock
from sumo_rl.agents.moe_experts import MoEExperts


def _sim():
    tab = {'J': {'phase_slots': [frozenset({0}), frozenset({1})],
                 'slot_lanes': {0: ['eA_0'], 1: ['eB_0']},
                 'lanes': {'eA_0': 'eA', 'eB_0': 'eB'},
                 'movement_by_edges': {('eA', 'out1'): 0, ('eB', 'out2'): 1}}}
    ex = MoEExperts(tab, delta_time=5, yellow_time=2,
                    prio_of_type={'amb': 5, 'car': 1}, default_level=1)
    sumo = MagicMock()
    sumo.lane.getLength.return_value = 100.0
    sumo.lane.getLastStepVehicleIDs.side_effect = \
        lambda lane: ['c1', 'c2', 'c3'] if lane == 'eA_0' else ['a1']
    sumo.vehicle.getTypeID.side_effect = lambda v: 'amb' if v == 'a1' else 'car'
    sumo.vehicle.getSpeed.side_effect = lambda v: 0.0 if v.startswith('c') else 10.0
    sumo.vehicle.getLanePosition.side_effect = \
        lambda v: {'c1': 95, 'c2': 88, 'c3': 81, 'a1': 80}[v]
    sumo.vehicle.getRoute.side_effect = \
        lambda v: ('eA', 'out1') if v.startswith('c') else ('eB', 'out2')
    sumo.vehicle.getRouteIndex.return_value = 0
    return ex, sumo


def test_expert_stance_divergence():
    ex, sumo = _sim()
    props, levels = ex.propose('J', sumo, current_phase=0)
    assert props[5] == 1, "expert-5 must pre-open for the approaching amb (ETA term)"
    assert props[1] == 0, "expert-1 must keep serving the car queue"
    assert levels == {1, 5}


def test_intent_exact_no_double_count():
    ex, sumo = _sim()
    ex.propose('J', sumo, current_phase=0)
    # 4 vehicles total -> memo has exactly 4 entries (each counted once)
    assert len(ex._vid_slot) == 4
