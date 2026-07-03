"""Behavioral unit test for MoE experts (hand-computed reference scores).
Scenario: slot0 = 3 stopped cars (queue), slot1 = 1 ambulance approaching at
10 m/s, 20 m out (ETA 2 s < Δt). Expected stance divergence:
expert-5 switches to serve the amb BEFORE it stops (ETA term);
expert-1 keeps the car phase. Run: pytest experiments/tests/test_moe_experts.py"""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from unittest.mock import MagicMock
import numpy as np
from sumo_rl.agents.moe_experts import MoEExperts


def _sim(design=2):
    tab = {'J': {'phase_slots': [frozenset({0}), frozenset({1})],
                 'slot_lanes': {0: ['eA_0'], 1: ['eB_0']},
                 'lanes': {'eA_0': 'eA', 'eB_0': 'eB'},
                 'movement_by_edges': {('eA', 'out1'): 0, ('eB', 'out2'): 1}}}
    ex = MoEExperts(tab, delta_time=5, yellow_time=2,
                    prio_of_type={'amb': 5, 'car': 1}, default_level=1,
                    design=design)
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
    """Holds under BOTH designs: amb expert pre-opens, car expert keeps."""
    for design in (1, 2):
        ex, sumo = _sim(design)
        props, levels = ex.propose('J', sumo, current_phase=0)
        assert props[5] == 1, f"v{design}: expert-5 must pre-open for the amb"
        assert props[1] == 0, f"v{design}: expert-1 must keep the car queue"
        assert levels == {1, 5}


def test_intent_exact_no_double_count():
    ex, sumo = _sim()
    ex.propose('J', sumo, current_phase=0)
    # 4 vehicles total -> memo has exactly 4 entries (each counted once)
    assert len(ex._vid_slot) == 4


def _sim_two_queues(qa, qb, current=0):
    """All-car junction: qa stopped cars on slot0 (served by phase0), qb on
    slot1 (phase1). Single lane each, lane length 100, cars packed from the
    stop line (position = 100 - 7.5*j)."""
    tab = {'J': {'phase_slots': [frozenset({0}), frozenset({1})],
                 'slot_lanes': {0: ['eA_0'], 1: ['eB_0']},
                 'lanes': {'eA_0': 'eA', 'eB_0': 'eB'},
                 'movement_by_edges': {('eA', 'out1'): 0, ('eB', 'out2'): 1}}}
    ex = MoEExperts(tab, delta_time=5, yellow_time=2,
                    prio_of_type={'amb': 5, 'car': 1}, default_level=1,
                    design=2, max_green=50)
    sumo = MagicMock()
    A = [f'a{j}' for j in range(qa)]
    B = [f'b{j}' for j in range(qb)]
    sumo.lane.getLength.return_value = 100.0
    sumo.lane.getLastStepVehicleIDs.side_effect = \
        lambda lane: A if lane == 'eA_0' else B
    sumo.vehicle.getTypeID.return_value = 'car'
    sumo.vehicle.getSpeed.return_value = 0.0
    sumo.vehicle.getLanePosition.side_effect = \
        lambda v: 100.0 - 7.5 * int(v[1:])
    sumo.vehicle.getRoute.side_effect = \
        lambda v: ('eA', 'out1') if v.startswith('a') else ('eB', 'out2')
    sumo.vehicle.getRouteIndex.return_value = 0
    return ex, sumo


def test_v2_hand_computed_plan_cost():
    """qa=3 vs qb=2, cur=0. Keep phase0: served 2+4+6=12 (T_hold=6), B waits
    2*6=12 -> 24. Switch phase1: delay 2, T_clear=2+4=6, T_hold=6; B served
    min(4,6)+min(6,6)=10; A waits 3*6=18 -> 28. Expert-0 keeps phase0."""
    ex, sumo = _sim_two_queues(3, 2, current=0)
    tab = ex.tables['J']
    queued, arriving, n_c = ex._scan(tab, sumo)
    mass = ex._mass_v2(tab, queued, arriving, 0)
    assert abs(mass[0, 0] - 24.0) < 1e-9, mass[:, 0]
    assert abs(mass[0, 1] - 28.0) < 1e-9, mass[:, 1]
    props, _ = ex.propose('J', sumo, current_phase=0)
    assert props[0] == 0


def test_v2_hysteresis_no_thrash():
    """Anti-M-B: mid-discharge of a big queue, a slightly larger rival must
    NOT trigger a switch (v1 flip-flopped here). 7 remaining on served slot0
    vs 8 waiting on slot1 -> keep. Only when the rival dominates (2 vs 12)
    does the expert release the phase."""
    ex, sumo = _sim_two_queues(7, 8, current=0)
    props, _ = ex.propose('J', sumo, current_phase=0)
    assert props[0] == 0, "must finish the discharging queue"
    ex2, sumo2 = _sim_two_queues(2, 12, current=0)
    props2, _ = ex2.propose('J', sumo2, current_phase=0)
    assert props2[0] == 1, "must release once the rival clearly dominates"


def test_presence_mask():
    """Structural validity mask: absent-level experts unselectable;
    expert-0 always valid; lexicographic still dominates when stacked."""
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    from moe_glue import gate_mask
    m = gate_mask({1, 3}, lexicographic=False, presence=True)
    assert m.tolist() == [True, True, False, True, False, False]
    m = gate_mask(set(), lexicographic=False, presence=True)
    assert m.tolist() == [True, False, False, False, False, False]
    m = gate_mask({1, 5}, lexicographic=True, presence=True)
    assert m.tolist() == [False, False, False, False, False, True]
    m = gate_mask({1, 3}, lexicographic=False, presence=False)
    assert m.all()


def test_v2_focal_divergence_and_abstain():
    """Anti-M-A: car queue on slot0, ONE amb queued on slot1, cur=1.
    Expert-1 (cars) wants phase0, expert-5 (amb) keeps phase1, experts of
    absent levels (2,3,4) abstain to current."""
    ex, sumo = _sim(design=2)
    # make the amb queued instead of arriving (speed 0)
    sumo.vehicle.getSpeed.side_effect = lambda v: 0.0
    props, levels = ex.propose('J', sumo, current_phase=1)
    assert props[1] == 0, "car expert must claim the green for the car queue"
    assert props[5] == 1, "amb expert must keep serving the queued amb"
    assert props[2] == 1 and props[3] == 1 and props[4] == 1, \
        "absent-level experts must abstain (keep current)"
    assert set(np.unique(props[[1, 5]])) == {0, 1}, "proposals must diverge"
