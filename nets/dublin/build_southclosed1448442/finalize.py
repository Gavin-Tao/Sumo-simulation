"""Collect completed checks and write the two copied scenarios' final READMEs."""
from pathlib import Path
import hashlib,json
from build import REPO,WORK,NETS,NEW_NET,NEW_META,digest
from normalize_injection import NAMES

report=dict(date='2026-10-28',net_sha256=digest(NEW_NET),meta_sha256=digest(NEW_META),scenarios={})
all_checks=0
for name in NAMES:
    folder=NETS/name;v=folder/'validation';static=json.loads((folder/'final_static_audit.json').read_text())
    rows=[];aggregate_state=0;aggregate_reward=0
    for policy in ('native','fixed','random'):
        for seed in (123,124,125):
            data=json.loads((v/f'{policy}_s{seed}_lanes.json').read_text());lanes=data['lanes']
            assert sum(x.get('planned',0) for x in lanes.values())==static['main_vehicles'],(name,policy,seed,'stale input')
            assert data['seconds']==3600 and data['collisions']==data['teleports']==0
            pending=sum(next((n for t,n in x.get('pending_samples',[]) if t==3300),0) for x in lanes.values())
            row=dict(policy=policy,seed=seed,pending_3300=pending,collisions=0,teleports=0,
                     max_source_pending=max(x.get('max_pending',0) for x in lanes.values()),
                     targets={l:{k:lanes[l][k] for k in ('arrivals','queue_seconds','tail_near_entry_seconds','longest_queue')}
                              for l in ('-77616869#2_0','317003251#0_0')})
            if policy=='native':
                assert data['drain']['cleared'] and data['drain']['remaining']==data['drain']['pending']==0
                row['drain']=data['drain']
            else:
                audit=json.loads((v/f'{policy}_s{seed}_audit.json').read_text())
                assert audit['tracker_missing']==0 and audit['decisions']==720
                assert audit['state_checks']==audit['reward_checks']==10800
                aggregate_state+=audit['state_checks'];aggregate_reward+=audit['reward_checks']
                outcome=json.loads((v/f'{policy}_s{seed}.json').read_text())
                row.update({k:outcome[k] for k in ('lock_visits','max_visit_wait','opp_by_type','running_growth_last2_vs_34','pairs_arrived')})
            rows.append(row);all_checks+=1
    files={str(p.relative_to(REPO)):digest(p) for p in folder.glob('*.rou.xml')}
    files.update({str(p.relative_to(REPO)):digest(p) for p in folder.glob('*.sumocfg')})
    scenario=dict(static=static,rows=rows,route_and_cfg_hashes=files,state_checks=aggregate_state,reward_checks=aggregate_reward)
    (folder/'validation/summary.json').write_text(json.dumps(scenario,indent=2)+'\n')
    report['scenarios'][name]=scenario
    is_pairs='pairs50' in name;counts=static['main_vehicles_by_type']
    table='| 控制器 | 种子 | 3300 s 未插入 | 等待 ≥150 s 访问 | 最长本次访问等待 |\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        table+=f"| {r['policy']} | {r['seed']} | {r['pending_3300']} | {r.get('lock_visits','未统计')} | {str(r['max_visit_wait'])+' s' if 'max_visit_wait' in r else '未统计'} |\n"
    drains=[r['drain']['time'] for r in rows if r['policy']=='native']
    cfg=next(folder.glob('*.sumocfg'))
    text=f'''# {name}

2026-10-28 定稿副本：包含 1448442 南侧改造、与 389281 双向直连、统一注入规则及背景流量标定。来源目录 `{name.removesuffix('_southclosed')}` 及其原共享路网均未修改。没有创建训练配置，没有启动训练。

## 打开与训练路径

GUI 打开本目录的 `{cfg.name}`。共用路网 `../{NEW_NET.name}`，配套动作 meta `../{NEW_META.name}`。训练/评估时，net_file、action_meta_file、cfg_file、route_file 必须一起指向本版本；旧检查点的性能不能直接视为本场景已训练效果。

## 道路及信号

- 封闭 135109542 的侧路 `14152304#2`、`-14152304#3`，移除该节点及仅有直行连接的 3506865299。
- 389281 → 1448442：`532427444#7`，2 条进口道，各 **110.24 m**。
- 1448442 → 389281：`-532427444#3`，1 条进口道，**110.04 m**。
- 两个路口独立控制；15 个 RL 信号保留，8STD 动作映射、掩码、转向语义均保持一致。边名变化由重索引工具更新到 net/meta。
- RL 读取合并后的整条进口道，state 为 336 维。加入注入缓冲段前后，全部 RL 进口的 SUMO 长度逐条相同；389332 东进口仍为 88.24 m。

## 统一注入规则

两套场景及所有备用路线统一：车辆起始边的下游节点必须为**非 RL 节点**，起始边长度至少 40 m。最终直接在 RL 进口道上发车的车辆数为 **0**。内部需求可以从现有非 RL 道路发车；不要求所有车辆从城市外围进入。

六处没有可用上游道路的边界入口使用 `source_` 前缀的接入缓冲段，长度约 140–150 m：`-4074527#1`、`-77616869#2`、`22962979`、`372226878#0`、`4074528#0`、`43321718#3`。这些是用于仿真注入的人工边界缓冲，不是实测 Dublin 新道路，也不计入 RL 进口容量。流量同时按实测瓶颈下调，避免仅把积压转移到缓冲区。

原有路线主体和最终目的边保留，向前补齐车道连续的上游路径；相应更换 departLane，移除不再适用的旧 departPos。车辆到达 RL 路口的时刻会因新增上游行程改变。三类 vType 物理参数文件原样保留。

## 最终需求

主场景共 **{static['main_vehicles']}** 辆：car **{counts.get('car',0)}**、bus **{counts.get('bus',0)}**、ambulance **{counts.get('ambulance',0)}**。背景小汽车比本次注入标定前减少 **{static['removed_background']}** 辆；公交、救护车及备用探针未删减。减流按原出发时刻确定性抽取，所有优先级对比臂应共用同一份最终场景文件。

{'pairs50 保留 50 辆公交 + 50 辆救护车、14 条共享主路线及 10 条备用未见路线；同一对车辆的完整路线相同。最初并路时延长了 290 辆背景车的终点，经过本轮减流后仍保留其中 281 辆，避免在新 RL 进口中途结束。' if is_pairs else '这是经过减流和注入调整的 18h 校准副本；不再代表原始 18h 需求总量。封闭侧路涉及的 17 辆主场景小汽车，以及备用探针，已保持原最终目的边改道。'}

最终实际逐转向需求见 `movement_demand.json`；逐车变更见 `modification_report.json` 和 `injection_changes.json`。`demand_targets.json` 是原生成目标表，实际最终需求以 movement_demand.json 为准。

## 整集验证

原生 GUI 配时（native）、固定轮转（fixed）、随机策略（random），各跑种子 123/124/125，每次 3600 s。本目录共 9 次完整仿真。所有运行碰撞数和 teleport 数均为 0。

fixed/random 使用现有 exp461 环境参数，只替换场景路径；按秒核对等待统计器，并在全部 15 个路口每 5 s 独立重算 state/reward。共 {aggregate_state} 次完整 state、{aggregate_reward} 次 reward 检查，浮点容差内一致，tracker.missing 恒为 0。独立等待参照为 100000 s 长窗口累计等待的逐步增量，不直接读取 tracker 作参照；该长窗口仅用于校验，不改变场景 sumocfg。

{table}
native 另在停止需求后继续仿真，三个种子均排空，最后一辆车离开时刻为 {min(drains):.0f}–{max(drains):.0f} s；排空期间也没有碰撞或 teleport。

“等待 ≥150 s”仅表示长等待访问，不能直接称作锁死。随机策略仍会因连续不给合适相位而出现长等待；这些检查不承诺任意 RL 策略都无排队。红灯排队和绿灯期间正常上游到达仍然存在。重点验证没有直接向 RL 进口插入车辆，没有持续增长的入口积压，且参考控制器下可排空。全网逐秒车道/源点记录见 `validation/*_lanes.json`，汇总见 `validation/summary.json`。

## 溯源

`SOURCE_README.md`、`source_calib/`、`source_*report.json`、`validation_before_injection/` 是以前版本的记录，不代表本版本结果。共享构建与核查脚本位于 `../build_southclosed1448442/`：build.py → normalize_injection.py → audit_final.py；原始输入哈希见 build_report.json，最终验证和文件哈希见 final_validation.json。修改之前的本次副本快照保存在该构建目录的 before_injection/。
'''
    (folder/'README.md').write_text(text,encoding='utf-8')
assert all_checks==18
report['one_hour_runs']=all_checks
(WORK/'final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Recorded',all_checks,'full-hour runs; READMEs updated')
