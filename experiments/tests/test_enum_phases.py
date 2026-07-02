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


# ---- Task 2: menu enumeration + phase strings ----

def _menu(tid):
    mov = common.tls_movements(NET, tid)
    st = EP.slot_tables(mov)
    rel = EP.movement_rel(mov, mov["nodes"], st)
    return mov, st, rel, EP.enumerate_menu(rel, st)


def test_menu_maximality_and_conflictfree():
    for tid in META8:
        mov, st, rel, menu = _menu(tid)
        exist = [EP.SLOT_IDX[s] for s in st]
        assert len(menu) >= 1, tid
        for p in menu:
            for m in p:
                for n in p:
                    assert rel[m][n] < 2, (tid, m, n)          # zero conflict inside
            for n in exist:                                     # maximality
                if n not in p:
                    assert any(rel[m][n] >= 2 for m in p), (tid, n)


def test_every_movement_served():
    for tid in META8:
        _, st, _, menu = _menu(tid)
        for s in st:
            assert any(EP.SLOT_IDX[s] in p for p in menu), (tid, s)


def test_phase_state_protected_only():
    for tid in META8:
        mov, st, rel, menu = _menu(tid)
        for p in menu:
            state = EP.phase_state(mov, st, p, mov["n_links"])
            assert "g" not in state and "G" in state
            EP.verify_phase(mov, mov["nodes"], state)           # raises on violation


def test_written_artifacts_row_correspondence():
    """The load-bearing contract: net tlLogic green order == meta
    phase_movements row order, per link, per phase (action k -> phase k ->
    Q row k must be the same phase). Verifies the WRITTEN files."""
    import xml.etree.ElementTree as ET
    meta_p = os.path.join(common.OUT_DIR, "dublin_enum_meta.json")
    net_p = os.path.join(common.OUT_DIR, "dublin_enum.net.xml")
    if not os.path.exists(meta_p):
        import pytest
        pytest.skip("enum artifacts not generated")
    meta = json.load(open(meta_p))
    root = ET.parse(net_p).getroot()
    for tl in root.findall("tlLogic"):
        tid = tl.get("id")
        t = meta["tls"][tid]
        greens = [p.get("state") for p in tl.findall("phase")
                  if "y" not in p.get("state")]
        assert len(greens) == t["n_phases"], tid
        link_slot = {int(i): EP.SLOT_IDX[(c[0]["approach"], c[0]["turn"])]
                     for i, c in t["links"].items()}
        for k, state in enumerate(greens):
            assert "g" not in state
            for i, s in link_slot.items():
                assert (state[i] == "G") == (t["phase_movements"][k][s] == 1), \
                    (tid, k, i)
