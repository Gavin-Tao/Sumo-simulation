"""reindex_8std.py --split-slots (2026-10-26, 任务 8): 8STD 重索引工具加槽粒度修正开关 (调用 enum_phases.split_merged_slots),
输出带 tag 的新 net/meta, 不覆盖旧文件。生成结果: 除 cluster_21620852 外 17 个路口的 tlLogic 相位与 meta 与现有文件相同;
该路口 link 2→(E,R)、link 6→(S,T), mask [T,T,T,T,F,F,T,T]; meta 顶层多一个 slot_split 键。需要 sumo 可执行文件。
运行: python -m pytest experiments/tests/test_reindex_split_slots.py -q"""
import os, sys, json, subprocess, shutil, xml.etree.ElementTree as ET, pytest
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(_REPO, "experiments/tools/dublin/reindex_8std.py")
OLD_NET = os.path.join(_REPO, "nets/dublin/dublin_8std.net.xml"); OLD_META = os.path.join(_REPO, "nets/dublin/dublin_8std_meta.json")
SRC = os.path.join(_REPO, "nets/dublin/dublin_8action_demoted.net.xml")
CLUSTER = "cluster_21620852_2383668110_2383668118_4009885977"
pytestmark = pytest.mark.skipif(not (os.path.exists(OLD_NET) and os.path.exists(SRC) and shutil.which("sumo")), reason="都柏林源网或 sumo 缺失")


def _phases(net_path):
    root = ET.parse(net_path).getroot()
    return {tl.get("id"): [(p.get("duration"), p.get("state")) for p in tl.findall("phase")] for tl in root.iter("tlLogic")}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("split")
    r = subprocess.run([sys.executable, TOOL, SRC, "--split-slots", "--tag", "splitslot", "--out-dir", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    net = out / "dublin_8std_splitslot.net.xml"; meta = out / "dublin_8std_splitslot_meta.json"
    assert net.exists() and meta.exists()
    return _phases(str(net)), json.load(open(meta))


def test_other_17_junctions_unchanged(generated):
    new_ph, new_meta = generated; old_ph, old_meta = _phases(OLD_NET), json.load(open(OLD_META))
    assert set(new_ph) == set(old_ph) and len(old_ph) == 18
    for tid in old_ph:
        if tid == CLUSTER:
            continue
        assert new_ph[tid] == old_ph[tid], tid
        assert new_meta["tls"][tid] == old_meta["tls"][tid], tid


def test_cluster_relabelled_and_actions_expanded(generated):
    _, meta = generated; t = meta["tls"][CLUSTER]
    assert {(c["approach"], c["turn"]) for c in t["links"]["2"]} == {("E", "R")}
    assert {(c["approach"], c["turn"]) for c in t["links"]["6"]} == {("S", "T")}
    assert t["mask"] == [True, True, True, True, False, False, True, True]
    assert meta["slot_split"] == {CLUSTER: [[2, ["E", "T"], ["E", "R"]], [6, ["S", "L"], ["S", "T"]]]}


def test_old_files_not_overwritten(generated):
    old_ph = _phases(OLD_NET); old_meta = json.load(open(OLD_META))
    assert [s for d, s in old_ph[CLUSTER] if "y" not in s] == ["GGGrrrrr", "rrrGrGrG", "rrrrrGGG", "rrrGGGrr"]
    assert {(c["approach"], c["turn"]) for c in old_meta["tls"][CLUSTER]["links"]["6"]} == {("S", "L")}
    assert "slot_split" not in old_meta
