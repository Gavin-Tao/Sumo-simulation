"""任务 4: 背景冻结。把 18h 定稿背景与车型文件逐位复制到新文件夹, 写默认 sumocfg (与 18h 相同的处理项)。
配对车辆另由 timetable50.py 写; 背景车一律不挪 (D7)。"""
import shutil
import common50 as P

SUMOCFG = """<?xml version='1.0' encoding='UTF-8'?>
<configuration>
    <input>
        <net-file value="../{net}" />
        <additional-files value="vtypes_same.add.xml" />
        <route-files value="{routes}" />
    </input>
    <time>
        <begin value="0" />
        <end value="3600" />
    </time>
    <processing>
        <time-to-teleport value="-1" />
        <ignore-junction-blocker value="-1" />
        <collision.check-junctions value="true" />
        <collision.action value="warn" />
    </processing>
    <report>
        <no-step-log value="true" />
        <duration-log.statistics value="true" />
    </report>
</configuration>
"""


def main():
    P.check_inputs()
    P.OUT.mkdir(parents=True, exist_ok=True); P.WORK.mkdir(parents=True, exist_ok=True)
    for dst, src in P.BG_COPY.items():
        shutil.copyfile(P.SRC18 / src, P.OUT / dst)
        assert P.sha256(P.OUT / dst) == P.BG_SHA[src], dst
    routes = ",".join(list(P.BG_DEFAULT) + ["pairs_tt1.rou.xml"])
    (P.OUT / "scenario.sumocfg").write_text(SUMOCFG.format(net=P.NET.name, routes=routes))
    print("背景已复制 (逐位相同):", ", ".join(P.BG_COPY))


if __name__ == "__main__":
    main()
