# FRAP-Enum Implementation Plan (spec: FRAP_ENUM_DESIGN_2026-07-02.txt)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable `action_scheme: enum_frap` mode — offline-enumerated protected-only phases + movement-duel FRAP network — with bit-identical behavior for all existing configs.

**Architecture:** New offline tool `enum_phases.py` writes `dublin_enum.net.xml` + `dublin_enum_meta.json`; new `frap_agent.py` provides `FRAPQNet`/`FRAPAgent` with the same attribute surface as `DQN`; `train.py` gets additive `elif` branches keyed on `cfg["action_scheme"]` at the four existing `action_meta_file` branch points.

**Tech Stack:** Python, PyTorch, sumolib, libsumo (SUMO_RL_LIBSUMO=1), pytest.

## Global Constraints

- Protected-only phases: every phase all-'G'; conflict = areFoes ∪ same-(to_edge,to_lane) merge; NO 'g' anywhere (spec §1d; user hard rule 2026-07-02).
- Zero modification to: `sumo_rl/environment/{traffic_signal,env,observations,rewards}.py`, `sumo_rl/agents/dqn_agent_txw.py`, all existing configs/nets/metas. Only `experiments/train.py` may be modified, additively.
- Old configs must be bit-identical: verified by the Task-0 baseline digests re-run after every train.py change.
- NEVER launch real training or anything that writes to wandb. Smoke/dry runs: `logging_mode: none`, ≤1 episode, short num_seconds.
- Slot order everywhere: `[(a,t) for a in ("N","E","S","W") for t in ("L","T","R")]` (index = 3*approach+turn), asserted against the obs class constants.
- Relation codes in meta: -1 nonexistent, 0 same-arm, 1 compatible, 2 merge, 3 crossing. Network consumes only {2,3}.
- Q aggregation: `Q(p) = Σ_{m∈p} g(d_m) + Σ_{n∉p} max_{m∈p, rel(m,n)≥2} s_mn` (each suppressed n counted once).
- Commit after each passing task on branch `dev`; commit messages `feat(frap-enum): ...` / `test(frap-enum): ...`.

---

### Task 0: Regression baseline harness (BEFORE any code change)

**Files:**
- Create: `experiments/tools/frap/regress_dryrun.py`
- Create: `experiments/tools/frap/regress_baseline.json` (generated)

**Interfaces:**
- Produces: CLI `python experiments/tools/frap/regress_dryrun.py <cfg.yaml> [--steps 30]` printing `DIGEST <sha256>`; baseline JSON `{cfg_path: digest}` consumed by Task 6.

- [ ] **Step 1: Write the harness** (stack-level determinism probe: env + obs + meta tables + greedy DQN with fixed init, exactly mirroring train.py's act path)

```python
"""Regression dry-run: N decision steps of a config's env/obs/action stack.
Digest covers (per step): sorted action dict, obs dims, rounded rewards.
Run from repo root. Deterministic on CPU for a fixed config."""
import os, sys, json, hashlib, functools, argparse
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np, torch, yaml, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg"); ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg))
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.agents.dqn_agent_txw import DQN
    obs_class = getattr(obsmod, {  # mirror OBS_REGISTRY names used by dublin/1x3 cfgs
        "PriorityMovement": "PriorityMovementObservationFunction",
        "PriorityBCA": "PriorityBCAObservationFunction",
    }[cfg["observation_class"]])
    obs_kwargs = {}
    for src, dst in [("obs_fields","fields"),("obs_phase_state","phase_state"),
                     ("priority_source","priority_source"),("obs_downstream","include_downstream"),
                     ("obs_downstream_fields","downstream_fields"),("obs_lane_occ","include_lane_occ"),
                     ("obs_awt_cap","awt_cap"),("obs_awt_basis","awt_basis"),("obs_slot_stats","slot_stats")]:
        if src in cfg:
            v = cfg[src]
            obs_kwargs[dst] = tuple(v) if isinstance(v, list) else v
    if obs_kwargs:
        obs_class = functools.partial(obs_class, **obs_kwargs)
    reward_fn = cfg["reward_fn"]
    if reward_fn == "priority-avg-waiting":
        from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
        from sumo_rl.environment.priority_map import load_priority_table
        reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
        num_seconds=cfg.get("num_seconds", 1000), min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50), use_max_green=cfg.get("use_max_green", False),
        single_agent=False, yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5), reward_fn=reward_fn,
        observation_class=obs_class, sumo_seed=cfg.get("seed", 0), sumo_warnings=False)
    ts_mask, std2green, green2std, ts_turnmap = {}, {}, {}, {}
    meta_file = cfg.get("action_meta_file")
    if meta_file:
        meta = json.load(open(meta_file))["tls"]
        for tid, t in meta.items():
            ts_mask[tid] = np.array(t["mask"], dtype=bool)
            ts_turnmap[tid] = {int(i): (c[0]["approach"], c[0]["turn"]) for i, c in t["links"].items()}
            s2g = np.full(8, -1, dtype=int)
            for k, gi in t["std_to_green_index"].items():
                s2g[int(k)] = gi
            std2green[tid] = s2g
            g2s = np.full(int(s2g.max()) + 1, -1, dtype=int)
            for k in range(8):
                if s2g[k] >= 0:
                    g2s[s2g[k]] = k
            green2std[tid] = g2s
    states = env.reset(int(cfg.get("seed", 0)))
    if meta_file:
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.std_action_map = green2std[tid]
            if hasattr(ts.observation_fn, "rebind_movements"):
                ts.observation_fn.rebind_movements(ts_turnmap[tid])
            ts.observation_space = ts.observation_fn.observation_space()
        states = {tid: env.traffic_signals[tid].observation_fn() for tid in env.ts_ids}
    od = len(next(iter(states.values())))
    agent = DQN(starting_state=tuple([0.0] * od), state_space=od, hidden_dim=cfg.get("hidden_dim", 64),
        action_space=(8 if meta_file else env.action_space.n), learning_rate=1e-3,
        gamma=0.95, epsilon=0.0, target_update=10, capacity=100, mini_size=10**9,
        batch_size=1, eps_start=0, eps_end=0, eps_decay=1, device="cpu")
    h = hashlib.sha256()
    done = {"__all__": False}
    for step in range(a.steps):
        if done["__all__"]:
            break
        if meta_file:
            acts = {ts: int(std2green[ts][agent.take_action(states[ts], mask=ts_mask[ts])])
                    for ts in env.ts_ids}
        else:
            acts = {ts: agent.take_action(states[ts]) for ts in env.ts_ids}
        states, r, done, _ = env.step(action=acts)
        h.update(json.dumps([sorted(acts.items()),
                             sorted((k, len(v)) for k, v in states.items()),
                             sorted((k, round(float(v), 6)) for k, v in r.items())]).encode())
    env.close()
    print(f"DIGEST {h.hexdigest()}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Capture baselines (pre-change) and save**

```bash
cd /home/xiaowen/sumo-rl
D208=$(python experiments/tools/frap/regress_dryrun.py experiments/configs/exp208_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095.yaml | grep DIGEST)
D136=$(python experiments/tools/frap/regress_dryrun.py experiments/configs/exp136_1x3_531_avg_waiting_NS20bus_1amb_U_tuned.yaml | grep DIGEST)
python - <<EOF
import json
json.dump({"exp208": "$D208".split()[1], "exp136": "$D136".split()[1]},
          open("experiments/tools/frap/regress_baseline.json", "w"), indent=1)
EOF
```
Expected: two `DIGEST <64-hex>` lines; JSON written. Run each twice to confirm digest is stable (determinism sanity).

- [ ] **Step 3: Commit**

```bash
git add experiments/tools/frap/
git commit -m "test(frap-enum): regression dry-run harness + pre-change baselines"
```

---

### Task 1: enum tool — slot tables, conflict matrix, relation classification

**Files:**
- Create: `experiments/tools/dublin/enum_phases.py` (functions only this task)
- Test: `experiments/tests/test_enum_phases.py`

**Interfaces:**
- Produces: `SLOTS: list[tuple[str,str]]` (12, fixed order); `slot_tables(mov) -> dict[(app,turn), list[int]]` (slot -> link indices); `movement_rel(mov, nodes, slot_links) -> list[list[int]]` (12×12, codes -1/0/1/2/3); reuses `reindex_8std.are_foes/same_arm`, `common.tls_movements/load_net`.

- [ ] **Step 1: Write failing tests**

```python
# experiments/tests/test_enum_phases.py
import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "tools", "dublin"))
import common
import enum_phases as EP

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
                assert rel[i][j] == rel[j][i]        # symmetric
                if rel[i][j] == -1:
                    si, sj = EP.SLOTS[i], EP.SLOTS[j]
                    assert si not in st or sj not in st

def test_no_intra_slot_conflict():          # spec §1 槽内 foe 断言 (measured 0)
    for tid in META8:
        mov = common.tls_movements(NET, tid)
        assert EP.intra_slot_conflicts(mov, mov["nodes"], EP.slot_tables(mov)) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/xiaowen/sumo-rl && python -m pytest experiments/tests/test_enum_phases.py -x -q`
Expected: FAIL `ModuleNotFoundError: enum_phases` (install pytest first if missing: `pip install pytest`).

- [ ] **Step 3: Implement**

```python
# experiments/tools/dublin/enum_phases.py
"""Enumerate ALL maximal protected (zero-conflict incl. merging) movement
phases per TLS -> dublin_enum.net.xml + dublin_enum_meta.json.
Spec: experiments/analysis/FRAP_ENUM_DESIGN_2026-07-02.txt §1.
Conflict(slot a, slot b) = (not same-arm AND any link-pair areFoes)
                           OR any link-pair shares (to_edge, to_lane).
Relation codes: -1 nonexistent, 0 same-arm, 1 compatible, 2 merge, 3 crossing."""
from __future__ import annotations
import json, os, subprocess, sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from reindex_8std import are_foes, same_arm, yellow_between  # noqa: E402

SLOTS = [(a, t) for a in ("N", "E", "S", "W") for t in ("L", "T", "R")]
SLOT_IDX = {s: k for k, s in enumerate(SLOTS)}
GREEN_DUR, YELLOW_DUR = "30", "3"
NET_ENUM = os.path.join(common.OUT_DIR, "dublin_enum.net.xml")
META_ENUM = os.path.join(common.OUT_DIR, "dublin_enum_meta.json")


def slot_tables(mov):
    st = {}
    for i, conns in mov["links"].items():
        st.setdefault((conns[0]["approach"], conns[0]["turn"]), []).append(i)
    return st


def _lane_share(mov, i, j):
    ti = {(c["to_edge"], c["to_lane"]) for c in mov["links"][i]}
    tj = {(c["to_edge"], c["to_lane"]) for c in mov["links"][j]}
    return bool(ti & tj)


def _edge_share(mov, i, j):
    ti = {c["to_edge"] for c in mov["links"][i]}
    tj = {c["to_edge"] for c in mov["links"][j]}
    return bool(ti & tj)


def intra_slot_conflicts(mov, nodes, st):
    bad = []
    for slot, idxs in st.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if (not same_arm(mov, i, j) and are_foes(mov, nodes, i, j)) \
                        or _lane_share(mov, i, j):
                    bad.append((slot, i, j))
    return bad


def movement_rel(mov, nodes, st):
    rel = [[-1] * 12 for _ in range(12)]
    for sa, ia in st.items():
        ka = SLOT_IDX[sa]
        for sb, ib in st.items():
            kb = SLOT_IDX[sb]
            if ka == kb:
                rel[ka][kb] = 0
                continue
            if sa[0] == sb[0]:                       # same approach arm
                rel[ka][kb] = 0
                continue
            foe = any(are_foes(mov, nodes, i, j) for i in ia for j in ib)
            merge_lane = any(_lane_share(mov, i, j) for i in ia for j in ib)
            if not foe and not merge_lane:
                rel[ka][kb] = 1
            else:
                # merge vs crossing: ALL conflict evidence is same-to_edge -> merge
                all_merge = all(
                    _edge_share(mov, i, j)
                    for i in ia for j in ib
                    if are_foes(mov, nodes, i, j) or _lane_share(mov, i, j))
                rel[ka][kb] = 2 if all_merge else 3
    return rel
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest experiments/tests/test_enum_phases.py -x -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/tools/dublin/enum_phases.py experiments/tests/test_enum_phases.py
git commit -m "feat(frap-enum): slot tables + 4-class movement relation matrix"
```

---

### Task 2: enum tool — maximal-set enumeration + phase strings + hard assertions

**Files:**
- Modify: `experiments/tools/dublin/enum_phases.py` (append)
- Test: `experiments/tests/test_enum_phases.py` (append)

**Interfaces:**
- Produces: `enumerate_menu(rel, st) -> list[frozenset[int]]` (maximal slot-index sets, sorted by 12-bit multi-hot tuple); `phase_state(mov, st, members, n_state) -> str`; `verify_phase(mov, nodes, state)` raising on any conflict or 'g'.

- [ ] **Step 1: Failing tests (append)**

```python
def _menu(tid):
    mov = common.tls_movements(NET, tid)
    st = EP.slot_tables(mov)
    rel = EP.movement_rel(mov, mov["nodes"], st)
    return mov, st, EP.enumerate_menu(rel, st)

def test_menu_maximality_and_conflictfree():
    for tid in META8:
        mov, st, menu = _menu(tid)
        rel = EP.movement_rel(mov, mov["nodes"], st)
        exist = [EP.SLOT_IDX[s] for s in st]
        assert 2 <= len(menu) <= 11
        for p in menu:
            for m in p:
                for n in p:
                    assert rel[m][n] < 2                      # zero conflict inside
            for n in exist:                                    # maximal
                if n not in p:
                    assert any(rel[m][n] >= 2 for m in p)

def test_every_movement_served():
    for tid in META8:
        _, st, menu = _menu(tid)
        for s in st:
            assert any(EP.SLOT_IDX[s] in p for p in menu)

def test_phase_state_protected_only():
    for tid in META8:
        mov, st, menu = _menu(tid)
        for p in menu:
            state = EP.phase_state(mov, st, p, mov["n_links"])
            assert "g" not in state and "G" in state
            EP.verify_phase(mov, mov["nodes"], state)          # raises on violation
```

- [ ] **Step 2: Run — expect FAIL (functions missing)**

Run: `python -m pytest experiments/tests/test_enum_phases.py -x -q` → `AttributeError: enumerate_menu`

- [ ] **Step 3: Implement (append to enum_phases.py)**

```python
def enumerate_menu(rel, st):
    exist = sorted(SLOT_IDX[s] for s in st)
    conflict = lambda a, b: rel[a][b] >= 2
    res = []
    def grow(cur, cand):
        ext = [c for c in cand if all(not conflict(c, x) for x in cur)]
        if not ext:
            if cur:
                res.append(frozenset(cur))
            return
        for k, c in enumerate(ext):
            grow(cur | {c}, ext[k + 1:])
    grow(set(), exist)
    maximal = [s for s in res if not any(s < r for r in res if r != s)]
    key = lambda p: tuple(1 if m in p else 0 for m in range(12))
    return sorted(set(maximal), key=key)


def phase_state(mov, st, members, n_state):
    state = ["r"] * n_state
    inv = {SLOT_IDX[s]: idxs for s, idxs in st.items()}
    for m in members:
        for i in inv[m]:
            state[i] = "G"
    return "".join(state)


def verify_phase(mov, nodes, state):
    if "g" in state:
        raise RuntimeError("protected-only violated: 'g' present")
    gset = [i for i in sorted(mov["links"]) if state[i] == "G"]
    for a in range(len(gset)):
        for b in range(a + 1, len(gset)):
            i, j = gset[a], gset[b]
            if not same_arm(mov, i, j) and are_foes(mov, nodes, i, j):
                raise RuntimeError(f"foe conflict {i},{j}")
            if _lane_share(mov, i, j):
                raise RuntimeError(f"merge conflict {i},{j}")
    seen = set()
    for i in gset:                                   # shared-index self merge
        here = set()
        for c in mov["links"][i]:
            key = (c["to_edge"], c["to_lane"])
            if key in here:
                raise RuntimeError(f"shared-index merge {i}->{key}")
            here.add(key)
    return True
```

- [ ] **Step 4: Run tests** → all pass (note: `test_menu_maximality_and_conflictfree` also revalidates menu sizes 2..11 with the stricter lane-merge rule; if any junction now exceeds/undershoots, record actual numbers in the tool report — spec expects mean ≤3.7).

- [ ] **Step 5: Commit** `git commit -am "feat(frap-enum): maximal menu enumeration + protected phase strings + verifiers"`

---

### Task 3: enum tool — net/meta writer, sumo load check, report; generate artifacts

**Files:**
- Modify: `experiments/tools/dublin/enum_phases.py` (append `main()`)
- Create (generated): `nets/dublin/dublin_enum.net.xml`, `nets/dublin/dublin_enum_meta.json`, `nets/dublin/weekday_{02,11,18}h/dublin_weekday_{02,11,18}h_enum.sumocfg`

**Interfaces:**
- Produces: meta JSON per spec §1 schema — top `{action_scheme:"enum_frap", n_actions:K_max, source_net, boundary, tls:{tid:{n_phases, phase_movements, movement_rel, links, mask}}}`. Consumed by Tasks 5/6.

- [ ] **Step 1: Implement main() (append)**

```python
def main():
    demoted = os.path.join(common.OUT_DIR, "dublin_8action_demoted.net.xml")
    src = sys.argv[1] if len(sys.argv) > 1 else demoted
    net = common.load_net(src)
    tree = ET.parse(src); root = tree.getroot()
    tls_ids = [tl.get("id") for tl in root.findall("tlLogic")]
    all_meta, programs, kmax = {}, {}, 0
    for tid in tls_ids:
        mov = common.tls_movements(net, tid)
        nodes = mov["nodes"]
        st = slot_tables(mov)
        bad = intra_slot_conflicts(mov, nodes, st)
        if bad:
            sys.exit(f"V1 FAIL {tid}: intra-slot conflicts {bad}")
        rel = movement_rel(mov, nodes, st)
        menu = enumerate_menu(rel, st)
        greens = []
        for p in menu:
            s = phase_state(mov, st, p, mov["n_links"])
            verify_phase(mov, nodes, s)
            greens.append(s)
        kmax = max(kmax, len(menu))
        phases = []
        for k, s in enumerate(greens):
            phases.append((GREEN_DUR, s))
            y = yellow_between(s, greens[(k + 1) % len(greens)])
            if "y" in y:
                phases.append((YELLOW_DUR, y))
        programs[tid] = phases
        all_meta[tid] = {
            "n_phases": len(menu),
            "phase_movements": [[1 if m in p else 0 for m in range(12)] for p in menu],
            "movement_rel": rel,
            "links": {str(i): mov["links"][i] for i in sorted(mov["links"])},
        }
    for tid, m in all_meta.items():                 # pad masks to global K_max
        m["mask"] = [k < m["n_phases"] for k in range(kmax)]
    for tl in root.findall("tlLogic"):
        tid = tl.get("id")
        tl.set("programID", "enum"); tl.set("offset", "0"); tl.set("type", "static")
        for p in list(tl):
            tl.remove(p)
        for dur, s in programs[tid]:
            ET.SubElement(tl, "phase", duration=dur, state=s)
    tree.write(NET_ENUM, encoding="UTF-8", xml_declaration=True)
    entries, exits = common.boundary_edges(net)
    json.dump({"action_scheme": "enum_frap", "n_actions": kmax,
               "source_net": os.path.relpath(src, common.REPO),
               "boundary": {"entries": sorted(e.getID() for e in entries),
                            "exits": sorted(e.getID() for e in exits)},
               "tls": all_meta}, open(META_ENUM, "w"), indent=1)
    sizes = sorted(m["n_phases"] for m in all_meta.values())
    print(f"TLS: {len(all_meta)}  K_max={kmax}  menu sizes={sizes}  "
          f"mean={sum(sizes)/len(sizes):.1f}")
    res = subprocess.run(["sumo", "-n", NET_ENUM, "--no-step-log", "-e", "0"],
                         capture_output=True, text=True)
    if res.returncode != 0 or "Error" in (res.stderr or ""):
        print(res.stderr); sys.exit("V1 FAIL: sumo cannot load enum net")
    print(f"sumo load: OK\nwrote {NET_ENUM}\nwrote {META_ENUM}")
    for h in ("02", "11", "18"):                    # sumocfg copies
        src_cfg = os.path.join(common.OUT_DIR, f"weekday_{h}h", f"dublin_weekday_{h}h.sumocfg")
        dst_cfg = src_cfg.replace(".sumocfg", "_enum.sumocfg")
        txt = open(src_cfg).read().replace("dublin_8std.net.xml", "dublin_enum.net.xml")
        open(dst_cfg, "w").write(txt)
        print(f"wrote {dst_cfg}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the tool**

Run: `cd /home/xiaowen/sumo-rl && python experiments/tools/dublin/enum_phases.py`
Expected: menu sizes printed (mean ≈3.7 or lower with the lane-merge rule — record actual), `sumo load: OK`, 3 sumocfgs written. Then re-run pytest: `python -m pytest experiments/tests/test_enum_phases.py -q` → pass.

- [ ] **Step 3: netconvert round-trip connection check**

```bash
netconvert -s nets/dublin/dublin_enum.net.xml -o /tmp/claude-1000/enum_rt.net.xml 2>&1 | grep -i error
python - <<'EOF'
import xml.etree.ElementTree as ET
def conns(p):
    return sorted((c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane"))
                  for c in ET.parse(p).getroot().findall("connection") if not c.get("via", "").startswith(":"))
a = conns("nets/dublin/dublin_8action_demoted.net.xml"); b = conns("nets/dublin/dublin_enum.net.xml")
assert a == b, f"connection drift: {len(a)} vs {len(b)}"
print("connections identical:", len(a))
EOF
```
Expected: no errors; `connections identical: <N>`.

- [ ] **Step 4: Commit** `git add nets/dublin/ experiments/tools/dublin/enum_phases.py && git commit -m "feat(frap-enum): enum net + meta + sumocfgs (protected-only menus)"`

---

### Task 4: FRAPQNet (network only, pure torch — no SUMO)

**Files:**
- Create: `sumo_rl/agents/frap_agent.py`
- Test: `experiments/tests/test_frap_agent.py`

**Interfaces:**
- Produces: `FRAPQNet(header_dim:int, slot_dim:int, embed_dim=16, pair_dim=16, k_max:int)`; `forward(x:(B,obs), pm:(B,K,12)f, rel:(B,12,12)l, exist:(B,12)f) -> (B,K)` raw Q (padded rows garbage — caller masks). Consumed by Task 5.

- [ ] **Step 1: Failing tests**

```python
# experiments/tests/test_frap_agent.py
import torch, math
from sumo_rl.agents.frap_agent import FRAPQNet

def _toy():
    # 3 slots exist (0,1,2); rel: 0-1 crossing(3), 0-2 compatible(1), 1-2 merge(2)
    rel = torch.full((12, 12), -1, dtype=torch.long)
    for i in range(3): rel[i, i] = 0
    rel[0, 1] = rel[1, 0] = 3; rel[0, 2] = rel[2, 0] = 1; rel[1, 2] = rel[2, 1] = 2
    pm = torch.zeros(2, 12); pm[0, 0] = pm[0, 2] = 1; pm[1, 1] = 1   # menu: {0,2}, {1}
    exist = torch.zeros(12); exist[:3] = 1
    return rel, pm, exist

def test_forward_shape_and_padding_maskable():
    net = FRAPQNet(header_dim=2, slot_dim=7, k_max=4)
    rel, pm, exist = _toy()
    pm4 = torch.zeros(4, 12); pm4[:2] = pm
    x = torch.randn(5, 2 + 12 * 7)
    q = net(x, pm4.unsqueeze(0).expand(5, -1, -1), rel.unsqueeze(0).expand(5, -1, -1),
            exist.unsqueeze(0).expand(5, -1))
    assert q.shape == (5, 4) and torch.isfinite(q[:, :2]).all()

def test_q_decomposition_exact():
    """Q(p) must equal sum(g of members) + sum over suppressed n of max duel."""
    torch.manual_seed(1)
    net = FRAPQNet(header_dim=2, slot_dim=7, k_max=2)
    rel, pm, exist = _toy()
    x = torch.randn(1, 2 + 12 * 7)
    q = net(x, pm.unsqueeze(0), rel.unsqueeze(0), exist.unsqueeze(0))
    d = net.encode(x)                     # (1,12,E) helper exposed for tests
    g = net.g_head(d).squeeze(-1)         # (1,12)
    s = net.duel_scores(d, rel.unsqueeze(0))  # (1,12,12), -inf on non-conflict
    # phase 0 = {0,2}: suppressed n=1 (conflicts 0 via crossing, 2 via merge)
    q0 = g[0, 0] + g[0, 2] + torch.max(s[0, 0, 1], s[0, 2, 1])
    # phase 1 = {1}: suppressed 0 (crossing) and 2 (merge)
    q1 = g[0, 1] + s[0, 1, 0] + s[0, 1, 2]
    assert torch.allclose(q[0, 0], q0, atol=1e-5)
    assert torch.allclose(q[0, 1], q1, atol=1e-5)

def test_gradients_flow():
    net = FRAPQNet(header_dim=2, slot_dim=7, k_max=2)
    rel, pm, exist = _toy()
    x = torch.randn(3, 2 + 12 * 7, requires_grad=True)
    q = net(x, pm.unsqueeze(0).expand(3, -1, -1), rel.unsqueeze(0).expand(3, -1, -1),
            exist.unsqueeze(0).expand(3, -1))
    q[:, :2].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run — FAIL (module missing)**: `python -m pytest experiments/tests/test_frap_agent.py -x -q`

- [ ] **Step 3: Implement FRAPQNet**

```python
# sumo_rl/agents/frap_agent.py
"""FRAP-enum agent: movement-duel network over enumerated protected phases.
Spec: experiments/analysis/FRAP_ENUM_DESIGN_2026-07-02.txt §2/§2.5.
Q(p) = sum_{m in p} g(d_m) + sum_{n not in p} max_{m in p, rel(m,n)>=2} s_mn."""
import collections, random
import numpy as np
import torch
import torch.nn.functional as F

NEG = -1e9


class FRAPQNet(torch.nn.Module):
    def __init__(self, header_dim, slot_dim, embed_dim=16, pair_dim=16, k_max=11):
        super().__init__()
        self.header_dim, self.slot_dim, self.k_max = header_dim, slot_dim, k_max
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(slot_dim, embed_dim), torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim))
        self.g_head = torch.nn.Linear(embed_dim, 1)
        self.pair_fc = torch.nn.Linear(2 * embed_dim, pair_dim)
        self.rel_emb = torch.nn.Embedding(2, pair_dim)      # 0=merge, 1=crossing
        self.s_head = torch.nn.Linear(pair_dim, 1)

    def encode(self, x):
        B = x.shape[0]
        slots = x[:, self.header_dim:].reshape(B, 12, self.slot_dim)
        return self.enc(slots)                               # (B,12,E)

    def duel_scores(self, d, rel):
        """(B,12,E),(B,12,12) -> (B,12,12) duel scores; NEG on non-conflict."""
        B, _, E = d.shape
        di = d.unsqueeze(2).expand(B, 12, 12, E)
        dj = d.unsqueeze(1).expand(B, 12, 12, E)
        h = self.pair_fc(torch.cat([di, dj], dim=-1))        # (B,12,12,P)
        conflict = rel >= 2                                   # (B,12,12)
        ridx = (rel.clamp(min=2) - 2).long()                  # (B,12,12) in {0,1}
        h = h * self.rel_emb(ridx)                            # relation modulation
        s = self.s_head(h).squeeze(-1)                        # (B,12,12)
        return s.masked_fill(~conflict, NEG)

    def forward(self, x, pm, rel, exist):
        """x (B,obs); pm (B,K,12) float; rel (B,12,12) long; exist (B,12) float."""
        d = self.encode(x)
        g = self.g_head(d).squeeze(-1)                        # (B,12)
        s = self.duel_scores(d, rel)                          # (B,12,12)
        q_self = (g.unsqueeze(1) * pm).sum(-1)                # (B,K)
        cand = pm.unsqueeze(-1) * (rel >= 2).float().unsqueeze(1)   # (B,K,12m,12n)
        masked = s.unsqueeze(1).masked_fill(cand == 0, NEG)   # (B,K,12,12)
        duel_max = masked.max(dim=2).values                   # (B,K,12) over members m
        suppressed = (cand.max(dim=2).values > 0).float() \
            * (1.0 - pm) * exist.unsqueeze(1)                 # (B,K,12)
        q_sup = (torch.where(suppressed > 0, duel_max,
                             torch.zeros_like(duel_max)) * suppressed).sum(-1)
        return q_self + q_sup                                 # (B,K)
```

- [ ] **Step 4: Run tests** → 3 passed. **Step 5: Commit** `git add sumo_rl/agents/frap_agent.py experiments/tests/test_frap_agent.py && git commit -m "feat(frap-enum): FRAPQNet with exact decomposition test"`

---

### Task 5: FRAPAgent + buffer (DQN-compatible attribute surface)

**Files:**
- Modify: `sumo_rl/agents/frap_agent.py` (append)
- Test: `experiments/tests/test_frap_agent.py` (append)

**Interfaces:**
- Produces: `FRAPAgent(obs_dim, header_dim, slot_dim, tls_tensors, lr, gamma, epsilon, target_update, capacity, mini_size, batch_size, eps_start, eps_end, eps_decay, device, embed_dim, pair_dim, k_max, use_double, loss_fn, grad_clip)` where `tls_tensors = {tid: {"pm": np(K_max,12), "rel": np(12,12), "exist": np(12,), "mask": np(K_max,)bool}}`.
  Exposes (read by train.py unchanged): `q_net, target_q_net, optimizer, epsilon, eps_start, eps_end, eps_decay, count, loss, grad_norm, q_mean, q_abs_max, start_train, use_per=False, replay_buffer, mini_size, batch_size`.
  Methods: `take_action(state, tls_id) -> int` (phase index, always valid); `replay_buffer.add(s, a, r, ns, done, tls_id)`; `learn_step()` (sample+update; no-op safety if buffer small).

- [ ] **Step 1: Failing tests (append)**

```python
import numpy as np
from sumo_rl.agents.frap_agent import FRAPAgent

def _agent(eps=0.0):
    rel, pm, exist = _toy()
    tls = {"J": {"pm": np.vstack([pm.numpy(), np.zeros((2, 12))]),   # K_max=4, 2 valid
                 "rel": rel.numpy(), "exist": exist.numpy(),
                 "mask": np.array([True, True, False, False])}}
    return FRAPAgent(obs_dim=2 + 12 * 7, header_dim=2, slot_dim=7, tls_tensors=tls,
                     lr=1e-3, gamma=0.95, epsilon=eps, target_update=5, capacity=500,
                     mini_size=8, batch_size=8, eps_start=eps, eps_end=eps, eps_decay=1,
                     device="cpu", k_max=4), tls

def test_take_action_always_valid():
    ag, _ = _agent(eps=1.0)                       # pure exploration
    s = np.random.randn(2 + 12 * 7).astype(np.float32)
    for _ in range(50):
        assert ag.take_action(s, "J") in (0, 1)
    ag.epsilon = 0.0
    for _ in range(10):
        assert ag.take_action(s, "J") in (0, 1)   # greedy also masked

def test_update_decreases_loss_and_targets_masked():
    torch.manual_seed(0); np.random.seed(0)
    ag, _ = _agent()
    for k in range(64):
        s = np.random.randn(2 + 12 * 7).astype(np.float32)
        ns = np.random.randn(2 + 12 * 7).astype(np.float32)
        ag.replay_buffer.add(s, k % 2, -1.0, ns, False, "J")
    losses = []
    for _ in range(30):
        ag.learn_step(); losses.append(ag.loss)
    assert all(math.isfinite(l) for l in losses)
    assert ag.q_mean is not None and ag.count == 30
    assert np.mean(losses[-5:]) <= np.mean(losses[:5]) * 2 + 1.0   # not exploding

def test_checkpoint_surface():
    ag, _ = _agent()
    assert hasattr(ag, "q_net") and hasattr(ag, "target_q_net") and hasattr(ag, "optimizer")
    assert ag.use_per is False and ag.start_train in (False, True)
```

- [ ] **Step 2: Run — FAIL.** **Step 3: Implement (append):**

```python
class FRAPReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, tls_id):
        self.buffer.append((state, action, reward, next_state, done, tls_id))

    def sample(self, batch_size):
        s, a, r, ns, d, t = zip(*random.sample(self.buffer, batch_size))
        return np.array(s), a, r, np.array(ns), d, t

    def size(self):
        return len(self.buffer)


class FRAPAgent:
    def __init__(self, obs_dim, header_dim, slot_dim, tls_tensors, lr, gamma, epsilon,
                 target_update, capacity, mini_size, batch_size, eps_start, eps_end,
                 eps_decay, device, embed_dim=16, pair_dim=16, k_max=11,
                 use_double=True, loss_fn="huber", grad_clip=1.0):
        self.device = torch.device(device)
        self.q_net = FRAPQNet(header_dim, slot_dim, embed_dim, pair_dim, k_max).to(self.device)
        self.target_q_net = FRAPQNet(header_dim, slot_dim, embed_dim, pair_dim, k_max).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma, self.epsilon = gamma, epsilon
        self.target_update, self.count = target_update, 0
        self.mini_size, self.batch_size = mini_size, batch_size
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.use_double, self.loss_fn, self.grad_clip = use_double, loss_fn, grad_clip
        self.use_per, self.start_train = False, False
        self.loss = None; self.grad_norm = None; self.q_mean = None; self.q_abs_max = None
        self.replay_buffer = FRAPReplayBuffer(capacity)
        self._ids = sorted(tls_tensors)
        self._idx = {t: k for k, t in enumerate(self._ids)}
        self.PM = torch.tensor(np.stack([tls_tensors[t]["pm"] for t in self._ids]),
                               dtype=torch.float, device=self.device)      # (T,K,12)
        self.REL = torch.tensor(np.stack([tls_tensors[t]["rel"] for t in self._ids]),
                                dtype=torch.long, device=self.device)      # (T,12,12)
        self.EXIST = torch.tensor(np.stack([tls_tensors[t]["exist"] for t in self._ids]),
                                  dtype=torch.float, device=self.device)   # (T,12)
        self.MASK = torch.tensor(np.stack([tls_tensors[t]["mask"] for t in self._ids]),
                                 dtype=torch.bool, device=self.device)     # (T,K)

    def _tensors(self, idx):
        return self.PM[idx], self.REL[idx], self.EXIST[idx], self.MASK[idx]

    def take_action(self, state, tls_id):
        i = self._idx[tls_id]
        if np.random.random() < self.epsilon:
            return int(np.random.choice(np.flatnonzero(self.MASK[i].cpu().numpy())))
        x = torch.tensor(np.asarray(state, dtype=np.float32),
                         device=self.device).unsqueeze(0)
        pm, rel, exist, mask = self._tensors(torch.tensor([i], device=self.device))
        q = self.q_net(x, pm, rel, exist).masked_fill(~mask, NEG)
        return int(q.argmax().item())

    def learn_step(self):
        if self.replay_buffer.size() <= self.mini_size:
            return
        self.start_train = True
        s, a, r, ns, d, tids = self.replay_buffer.sample(self.batch_size)
        idx = torch.tensor([self._idx[t] for t in tids], device=self.device)
        pm, rel, exist, mask = self._tensors(idx)
        states = torch.tensor(s, dtype=torch.float, device=self.device)
        next_states = torch.tensor(ns, dtype=torch.float, device=self.device)
        actions = torch.tensor(a, device=self.device).view(-1, 1)
        rewards = torch.tensor(r, dtype=torch.float, device=self.device).view(-1, 1)
        dones = torch.tensor(d, dtype=torch.float, device=self.device).view(-1, 1)
        q = self.q_net(states, pm, rel, exist).gather(1, actions)
        with torch.no_grad():
            if self.use_double:
                nq = self.q_net(next_states, pm, rel, exist).masked_fill(~mask, NEG)
                na = nq.argmax(1, keepdim=True)
                mnq = self.target_q_net(next_states, pm, rel, exist).gather(1, na)
            else:
                tq = self.target_q_net(next_states, pm, rel, exist).masked_fill(~mask, NEG)
                mnq = tq.max(1)[0].view(-1, 1)
        tgt = rewards + self.gamma * mnq * (1 - dones)
        loss = F.smooth_l1_loss(q, tgt) if self.loss_fn == "huber" else F.mse_loss(q, tgt)
        self.loss = loss.item()
        with torch.no_grad():
            self.q_mean = q.mean().item(); self.q_abs_max = q.abs().max().item()
        self.optimizer.zero_grad(); loss.backward()
        if self.grad_clip is not None:
            self.grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip))
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
```

- [ ] **Step 4: Run all agent tests** → pass. **Step 5: Commit** `git commit -am "feat(frap-enum): FRAPAgent with DQN-compatible surface + masked Double-DQN"`

---

### Task 6: train.py additive branch + exp211 config + regression proof

**Files:**
- Modify: `experiments/train.py` (five additive insertions, listed below with anchors)
- Create: `experiments/configs/exp211_dublin11h_531_enumfrap.yaml`

**Interfaces:**
- Consumes: Task 3 meta, Task 5 `FRAPAgent`. New cfg keys: `action_scheme: enum_frap`, `enum_meta_file`, optional `frap: {embed_dim, pair_dim}`.

- [ ] **Step 1: Insertion 1 — guards + meta load (right after the existing `action_meta_file` block ends at the `_masked_reset` definition; anchor: line ~254, after `green2std[_tid] = _g2s` loop)**

```python
    # ── enum_frap (spec FRAP_ENUM_DESIGN_2026-07-02): enumerated protected
    # phases + movement-duel agent. Old configs lack action_scheme -> None ->
    # every line below is skipped; mutually exclusive with action_meta_file.
    action_scheme = cfg.get("action_scheme")
    enum_tables = None
    if action_scheme not in (None, "enum_frap"):
        sys.exit(f"unknown action_scheme: {action_scheme}")
    if action_scheme == "enum_frap":
        if action_meta_file:
            sys.exit("action_scheme=enum_frap is mutually exclusive with action_meta_file")
        if cfg.get("use_per"):
            sys.exit("enum_frap + use_per not supported (uniform replay by design)")
        if cfg.get("agent_arch", "mlp") != "mlp":
            sys.exit("enum_frap defines its own network; agent_arch must be absent/mlp")
        from frap_glue import load_enum_tables
        enum_tables = load_enum_tables(cfg["enum_meta_file"])
```

- [ ] **Step 2: Insertion 2 — `_masked_reset` enum branch (inside `_masked_reset`, after the `if action_meta_file:` block, same indent)**

```python
        if enum_tables is not None:
            for _tid in _env.ts_ids:
                _ts = _env.traffic_signals[_tid]
                if hasattr(_ts.observation_fn, "rebind_movements"):
                    _ts.observation_fn.rebind_movements(enum_tables["turnmap"][_tid])
                _ts.observation_space = _ts.observation_fn.observation_space()
            _states = {_tid: _env.traffic_signals[_tid].observation_fn()
                       for _tid in _env.ts_ids}
```

- [ ] **Step 3: Insertion 3 — agent construction (wrap the existing `agent = DQN(...)`: `if enum_tables is not None: agent = build_frap_agent(...) else: <existing DQN call unchanged>`)**

```python
        if enum_tables is not None:
            from frap_glue import build_frap_agent
            agent = build_frap_agent(cfg, enum_tables, env, device)
        else:
            agent = DQN(
                ...existing kwargs verbatim, unchanged...
            )
```

- [ ] **Step 4: Insertion 4 — act / store / update / eval-act branches**

Act (anchor line ~375): prepend a branch:
```python
                    if enum_tables is not None:
                        actions = {ts: agent.take_action(initial_states[ts], ts)
                                   for ts in env.ts_ids}
                    elif action_meta_file:
                        ...existing...
```
Store (anchor ~402): prepend:
```python
                        if enum_tables is not None:
                            agent.replay_buffer.add(initial_states[ts], actual_action,
                                ts_reward, ts_next_state, ts_done, ts)
                        elif action_meta_file:
                            ...existing...
```
Update (anchor ~426, inside `if agent.replay_buffer.size() > agent.mini_size:` before the PER branch):
```python
                        if enum_tables is not None:
                            agent.learn_step()
                        elif agent.use_per:
                            ...existing...
```
Eval act (anchor ~520): prepend the same enum branch as Act with `eval_obs`.
Note: `actual_action = env.traffic_signals[ts].last_executed_action` is already the dense green-phase index = enum action index (identity map) — no conversion line needed for enum.

- [ ] **Step 5: Create `experiments/frap_glue.py`**

```python
"""Glue between train.py and the enum_frap scheme. Import-only from the
enum branch — importing this module must have no side effects."""
import json
import numpy as np

SLOTS = [(a, t) for a in ("N", "E", "S", "W") for t in ("L", "T", "R")]


def load_enum_tables(meta_path):
    meta = json.load(open(meta_path))
    assert meta.get("action_scheme") == "enum_frap", meta_path
    k_max = int(meta["n_actions"])
    tls_tensors, turnmap = {}, {}
    for tid, t in meta["tls"].items():
        pm = np.zeros((k_max, 12), dtype=np.float32)
        pm[: t["n_phases"]] = np.array(t["phase_movements"], dtype=np.float32)
        rel = np.array(t["movement_rel"], dtype=np.int64)
        exist = (rel.diagonal() >= 0).astype(np.float32)
        tls_tensors[tid] = {"pm": pm, "rel": rel, "exist": exist,
                            "mask": np.array(t["mask"], dtype=bool)}
        turnmap[tid] = {int(i): (c[0]["approach"], c[0]["turn"])
                        for i, c in t["links"].items()}
    return {"k_max": k_max, "tls": tls_tensors, "turnmap": turnmap}


def build_frap_agent(cfg, tables, env, device):
    from sumo_rl.agents.frap_agent import FRAPAgent
    obs_dim = env.observation_space.shape[0]
    header_dim = 2                       # perphase: [min_green_ok, elapsed/100]
    assert cfg.get("obs_phase_state") == "perphase", \
        "enum_frap requires obs_phase_state: perphase (junction-independent obs dim)"
    slot_dim, rem = divmod(obs_dim - header_dim, 12)
    assert rem == 0, f"obs dim {obs_dim} not header+12*slot"
    fp = cfg.get("frap", {}) or {}
    return FRAPAgent(obs_dim=obs_dim, header_dim=header_dim, slot_dim=slot_dim,
        tls_tensors=tables["tls"], lr=cfg.get("lr", 1e-3), gamma=cfg.get("gamma", 0.95),
        epsilon=cfg.get("epsilon", 0.1), target_update=cfg.get("target_update", 10),
        capacity=cfg.get("capacity", 10000), mini_size=cfg.get("mini_size", 500),
        batch_size=cfg.get("batch_size", 256), eps_start=cfg.get("eps_start", 0.5),
        eps_end=cfg.get("eps_end", 0.01), eps_decay=cfg.get("eps_decay", 1000),
        device=device, embed_dim=int(fp.get("embed_dim", 16)),
        pair_dim=int(fp.get("pair_dim", 16)), k_max=tables["k_max"],
        use_double=cfg.get("use_double", True), loss_fn=cfg.get("loss_fn", "huber"),
        grad_clip=cfg.get("grad_clip", 1.0))
```

- [ ] **Step 6: exp211 config — copy exp208 yaml, then apply exactly these key changes** (all other keys verbatim from exp208):

```yaml
name: exp211_dublin11h_531_enumfrap
net_file: nets/dublin/dublin_enum.net.xml
cfg_file: nets/dublin/weekday_11h/dublin_weekday_11h_enum.sumocfg
action_scheme: enum_frap
enum_meta_file: nets/dublin/dublin_enum_meta.json
obs_phase_state: perphase
frap: {embed_dim: 16, pair_dim: 16}
# REMOVE the action_meta_file key
```

- [ ] **Step 7: Regression proof (the backward-compat gate)**

```bash
python experiments/tools/frap/regress_dryrun.py experiments/configs/exp208_...yaml
python experiments/tools/frap/regress_dryrun.py experiments/configs/exp136_...yaml
python - <<'EOF'
import json, subprocess, sys
base = json.load(open("experiments/tools/frap/regress_baseline.json"))
print("baselines:", base)  # compare manually printed digests above — must be EQUAL
EOF
grep -n "action_scheme\|enum_meta_file\|frap_glue" experiments/train.py
```
Expected: both digests EQUAL to baseline JSON; grep hits only inside the guarded enum branch lines. Also `python -c "import yaml; yaml.safe_load(open('experiments/configs/exp211_dublin11h_531_enumfrap.yaml'))"` parses.

- [ ] **Step 8: Commit** `git add experiments/train.py experiments/frap_glue.py experiments/configs/exp211_dublin11h_531_enumfrap.yaml && git commit -m "feat(frap-enum): additive enum_frap branch in train.py + exp211 config (old-config digests unchanged)"`

---

### Task 7: V3 smoke run (no wandb) + runtime assertions

**Files:**
- Create: `experiments/tools/frap/smoke_enum.py`

- [ ] **Step 1: Write smoke harness** — mirrors regress_dryrun but for exp211: builds env from exp211 cfg, `load_enum_tables` + `build_frap_agent`, runs 120 decision steps with epsilon=0.3, asserting each step:

```python
# key assertions inside the loop (full file mirrors regress_dryrun structure):
act = agent.take_action(states[ts], ts)
assert tables["tls"][ts]["mask"][act], f"invalid phase chosen at {ts}"
# after env.step: sumo red-yellow-green sanity via ts objects
ts_obj = env.traffic_signals[ts]
assert ts_obj.green_phase < tables["tls"][ts]["mask"].sum()
# feed buffer + learn every step once size > mini(=64 for smoke); record losses
# end of run:
assert all(np.isfinite(losses)) and len(losses) > 20
net_xml = open(cfg["net_file"]).read()
assert 'state="' in net_xml and "g" not in __import__("re").findall(r'state="([^"]+)"', net_xml)[0]
# stronger: assert no 'g' in ANY phase string of the enum net:
import re
assert all("g" not in s for s in re.findall(r'state="([^"]+)"', net_xml)), "protected-only violated"
print("SMOKE OK: steps=%d losses[first,last]=%.4f,%.4f" % (n, losses[0], losses[-1]))
```

- [ ] **Step 2: Run** `SUMO_RL_LIBSUMO=1 python experiments/tools/frap/smoke_enum.py` → `SMOKE OK`, no assertion failures. Also run once with `epsilon=0.0` to exercise the greedy path.

- [ ] **Step 3: Full test suite + final regression**

```bash
python -m pytest experiments/tests/ -q          # all green
python experiments/tools/frap/regress_dryrun.py experiments/configs/exp208_...yaml   # digest still == baseline
```

- [ ] **Step 4: Commit** `git commit -am "test(frap-enum): V3 smoke harness (masked actions, protected-only, finite loss)"`

---

### Task 8: Close out

- [ ] Append an "IMPLEMENTATION STATUS 2026-07-XX" block to `experiments/analysis/FRAP_ENUM_DESIGN_2026-07-02.txt` recording: actual menu sizes/K_max from the tool report, digests unchanged, smoke results, and the exact command for the user to launch exp211 training (`python experiments/train.py --config experiments/configs/exp211_dublin11h_531_enumfrap.yaml`) — training launch is the USER's action.
- [ ] Final commit.

## Self-Review Notes

- Spec coverage: §1 tool → Tasks 1–3; §2 network/agent → Tasks 4–5; §4 trainer branch → Task 6; §6 V1 → Task 2/3 asserts, V2 → Tasks 0+6, V3 → Task 7; §5 M4 arms 2/3 → exp211 is arm 3; arm 2 (flat-DQN on enum) and M5 frozen_tls are follow-ups, NOT in this plan (spec marks M5 as phase-2 design-only).
- Menu sizes may come out below 3.7 (stricter lane-merge rule) — Task 3 records actuals; spec front-matter updated in Task 8 if they differ.
- K_max is read from the generated meta, not hardcoded (spec's "=11" is an estimate).
