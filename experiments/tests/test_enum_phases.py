"""Tests for enum_phases (FRAP_ENUM_PLAN Tasks 1-2). Run from repo root:
    python -m pytest experiments/tests/test_enum_phases.py -q
Uses the real Dublin net + 8std meta (read-only)."""
import sys, os, json

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "tools", "dublin"))
import common  # noqa: E402
import enum_phases as EP  # noqa: E402

NET = common.load_net(os.path.join(common.OUT_DIR, "dublin_8action_demoted.net.xml"))
META8 = json.load(open(common.META_8STD))["tls"]


def test_slot_order_matches_obs():
    from sumo_rl.environment import observations as O
    cls = O.PriorityMovementObservationFunction
    apps = getattr(cls, "_APPROACHES", None) or getattr(O, "_APPROACHES")
    turns = getattr(cls, "_TURNS", None) or getattr(O, "_TURNS")
    assert EP.SLOTS == [(a, t) for a in apps for t in turns]


def test_slot_tables_cover_all_links():
    for tid in META8:
        mov = common.tls_movements(NET, tid)
        st = EP.slot_tables(mov)
        assert sorted(i for v in st.values() for i in v) == sorted(mov["links"])


def test_rel_matrix_props():
    for tid in META8:
        mov = common.tls_movements(NET, tid)
        st = EP.slot_tables(mov)
        rel = EP.movement_rel(mov, mov["nodes"], st)
        for i in range(12):
            assert rel[i][i] in (-1, 0)
            for j in range(12):
                assert rel[i][j] == rel[j][i], (tid, i, j)      # symmetric
                if rel[i][j] == -1 and i != j:
                    assert EP.SLOTS[i] not in st or EP.SLOTS[j] not in st


def test_no_intra_slot_conflict():
    """Spec §1: intra-slot foe assertion (measured 0 on this net)."""
    for tid in META8:
        mov = common.tls_movements(NET, tid)
        assert EP.intra_slot_conflicts(mov, mov["nodes"], EP.slot_tables(mov)) == []
