"""把主对比表汇总成一个 Excel (2026-09-30, 用户令)。
输入: experiments/analysis/figures/main_comparison_{1x1,1x3,02h,11h}_best10seeds_2026-09-30_en.json (best 检查点 × 10 评估种子, 与同名 png 同源)
      + Dublin 11h 尾 40 次评估块 (plot_main_table_en_2026-09-22.py 里的 S / EV 常量, 与 main_comparison_8std_vs_frap_2026-09-22_en.png 的 11h 块同源)。
输出: experiments/analysis/tables/main_comparison_tables_2026-09-30.xlsx
表内: 均值/标准差为测得数据 (来自 json); 相对差、保序门、换车型块的均值/标准差/计数全部用公式; 字体 Arial。"""
import json, os, sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIG = os.path.join(REPO, "experiments/analysis/figures"); OUT = os.path.join(REPO, "experiments/analysis/tables/main_comparison_tables_2026-09-30.xlsx")
F = "Arial"; B = Font(name=F, bold=True); N = Font(name=F); TITLE = Font(name=F, bold=True, size=12)
HFILL = PatternFill("solid", fgColor="C9D3DF"); TFILL = PatternFill("solid", fgColor="E3E8EF"); GREEN = PatternFill("solid", fgColor="CFE9CF")
thin = Side(style="thin", color="999999"); BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
LAB = {"Stopped time / visit (s) - car": "停车时间/visit (s) — 小汽车", "Stopped time / visit (s) - bus": "停车时间/visit (s) — 公交",
       "Stopped time / visit (s) - ambulance": "停车时间/visit (s) — 救护车", "Stopped time / visit (s) - all": "停车时间/visit (s) — 全体",
       "J(531) weighted cost (per visit)": "J(531) 加权停车时间 (per visit)",
       "Stop events / visit - car": "停车次数/visit — 小汽车", "Stop events / visit - bus": "停车次数/visit — 公交",
       "Stop events / visit - ambulance": "停车次数/visit — 救护车", "Stop events / visit - all": "停车次数/visit — 全体",
       "J(531) weighted stop events (per visit)": "J(531) 加权停车次数 (per visit)"}
HDR = ["指标", "8STD 均值", "8STD 标准差", "GS-ENUM 均值", "GS-ENUM 标准差", "相对差 (GS−8STD)/8STD", "GS 配对胜 / 总", "备注"]

def style_row(ws, r, ncol=8, bold=False, fill=None):
    for c in range(1, ncol + 1):
        cell = ws.cell(r, c); cell.font = B if bold else N; cell.border = BOX; cell.alignment = Alignment(vertical="center", wrap_text=(c in (1, 8)))
        if fill: cell.fill = fill

def write_block(ws, r, title, rows_t, rows_e, amb_na, note=""):
    """一个数据块: 标题行 + 表头 + 停车时间 5 行 + 门 + 停车次数 5 行 + 门。返回 (下一空行, 关键单元格字典)。"""
    ws.cell(r, 1, title).font = TITLE; ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8); ws.cell(r, 1).fill = TFILL; r += 1
    for c, h in enumerate(HDR, 1): ws.cell(r, c, h)
    style_row(ws, r, bold=True, fill=HFILL); r += 1
    key = {}
    def rows(block, tag):
        nonlocal r
        first = r
        for row in block:
            lab, am, asd, bm, bsd, _rel, wins = (row + [""] * 7)[:7]
            ws.cell(r, 1, LAB.get(lab, lab)); ws.cell(r, 2, float(am)); ws.cell(r, 3, float(asd)); ws.cell(r, 4, float(bm)); ws.cell(r, 5, float(bsd))
            ws.cell(r, 6, f'=IF(B{r}=0,"",D{r}/B{r}-1)'); ws.cell(r, 7, wins if "J(531)" in lab else "")
            for c in (2, 3, 4, 5): ws.cell(r, c).number_format = "0.00"
            ws.cell(r, 6).number_format = "+0%;-0%;0%"
            style_row(ws, r)
            # 两臂里更好的一方标绿 (越小越好)
            ws.cell(r, 2 if float(am) <= float(bm) else 4).fill = GREEN
            if "ambulance" in lab: ws.cell(r, 8, "Dublin 每次评估救护车很少 (02h 2 辆 / 11h 4 辆), 单看均值不可靠")
            key[(tag, lab.split(" - ")[-1] if " - " in lab else "J")] = r
            r += 1
        return first
    rows(rows_t, "t")
    car, bus, amb = key[("t", "car")], key[("t", "bus")], key[("t", "ambulance")]
    ws.cell(r, 1, "保序门 (停车时间): 救护车 ≤ 公交 ≤ 小汽车")
    for c, col in ((2, "B"), (4, "D")):
        ws.cell(r, c, f'=IF(AND({col}{amb}<={col}{bus},{col}{bus}<={col}{car}),"pass",IF(AND({col}{amb}-{col}{bus}<=5,{col}{bus}-{col}{car}<=5),"pass (tol. 5 s)","fail"))')
    ws.cell(r, 8, "tol. 5 s = 5 s 采样量子内的容差"); style_row(ws, r); r += 1
    rows(rows_e, "e")
    car, bus, amb = key[("e", "car")], key[("e", "bus")], key[("e", "ambulance")]
    ws.cell(r, 1, "保序门 (停车次数): 救护车 ≤ 公交 ≤ 小汽车" + (" (救护车 n/a, 只比公交 ≤ 小汽车)" if amb_na else ""))
    for c, col in ((2, "B"), (4, "D")):
        if amb_na: ws.cell(r, c, f'=IF({col}{bus}<={col}{car},"pass (amb n/a)","fail (amb n/a)")')
        else: ws.cell(r, c, f'=IF(AND({col}{amb}<={col}{bus},{col}{bus}<={col}{car}),"pass",IF(AND({col}{amb}-{col}{bus}<=0.5,{col}{bus}-{col}{car}<=0.5),"pass (tol.)","fail"))')
    if amb_na: ws.cell(r, 8, "Dublin 每次评估救护车 1-4 辆, 停车次数只能取 0/0.5/1, 不与公交几百辆的均值直接比")
    style_row(ws, r); r += 1
    if note: ws.cell(r, 1, note).font = Font(name=F, italic=True, color="555555"); ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8); r += 1
    return r + 1, key

def write_typeswap(ws, r, seeds, ts, title):
    """同路线同时刻换车型块: 逐种子原始值 + 公式均值/标准差/计数。"""
    ws.cell(r, 1, title).font = TITLE; ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8); ws.cell(r, 1).fill = TFILL; r += 1
    hdr = ["车辆 / 指标"] + [f"种子 {s}" for s in seeds] + ["均值", "标准差"]
    for c, h in enumerate(hdr, 1): ws.cell(r, c, h)
    style_row(ws, r, ncol=len(hdr), bold=True, fill=HFILL); r += 1
    n = len(seeds); last = get_column_letter(1 + n); mcol = get_column_letter(2 + n); scol = get_column_letter(3 + n)
    lines = []
    for arm in ("8STD", "GS"):
        for metric, mk in (("停车时间/visit (s)", "pv"), ("停车次数/visit", "ev")):
            for veh, idx in (("救护车 (探针路线)", 0), ("公交 (同路线同时刻)", 1)):
                ws.cell(r, 1, f"{arm} — {veh} — {metric}")
                for k, v in enumerate(ts[arm][mk][idx]): ws.cell(r, 2 + k, float(v)).number_format = "0.00"
                ws.cell(r, 2 + n, f"=AVERAGE(B{r}:{last}{r})").number_format = "0.00"; ws.cell(r, 3 + n, f"=STDEV(B{r}:{last}{r})").number_format = "0.00"
                style_row(ws, r, ncol=len(hdr)); lines.append((arm, mk, idx, r)); r += 1
    for arm in ("8STD", "GS"):
        for metric, mk in (("停车时间", "pv"), ("停车次数", "ev")):
            ra = [x[3] for x in lines if x[0] == arm and x[1] == mk and x[2] == 0][0]; rb = [x[3] for x in lines if x[0] == arm and x[1] == mk and x[2] == 1][0]
            ws.cell(r, 1, f"{arm} — {metric}: 救护车 < 公交 的评估次数 / 平局")
            ws.cell(r, 2, f'=SUMPRODUCT(--(B{ra}:{last}{ra}<B{rb}:{last}{rb}))&" / "&SUMPRODUCT(--(B{ra}:{last}{ra}=B{rb}:{last}{rb}))&" (共 {n})"')
            style_row(ws, r, ncol=2); r += 1
    return r + 1

def block_pairs(d):
    """json 里 '标题|指标' 键 → [(标题, 停车时间行, 停车次数行)] 按出现顺序。"""
    out = {}
    for k, v in d.items():
        if "|" in k and isinstance(v, list):
            t, m = k.split("|"); out.setdefault(t, {})[m] = v
    return [(t, b.get("avg_stopped_time_per_visit"), b.get("avg_stop_events_per_visit")) for t, b in out.items()]

def sheet_from_json(wb, name, path, amb_na, protocol, title_override=None, ws=None, r=None, typeswap=True):
    """title_override: 块标题替换文本 (2026-09-30: 用户选定的 11h 口径需要中文说明); ws/r 给出时在已有工作表上追加。"""
    d = json.load(open(path))
    if ws is None: ws = wb.create_sheet(name); r = 1
    ws.cell(r, 1, f"{name}: best 检查点 × {len(d['seeds'])} 评估种子 (种子 {', '.join(map(str, d['seeds']))}); {protocol}").font = TITLE; r += 1
    ws.cell(r, 1, f"数据源: {os.path.relpath(path, REPO)} (与同名 png 同源); 异常种子 (对称规则): {d.get('anomalous') or '无'}; Tukey 离群种子: {d.get('tukey_outliers') or '无'}").font = Font(name=F, italic=True); r += 2
    keys = []
    for t, rt, re_ in block_pairs(d):
        tt = title_override or t
        r, key = write_block(ws, r, tt, rt, re_, amb_na); keys.append((tt, key))
    if not typeswap: d.pop("typeswap", None)
    if "typeswap" in d: r = write_typeswap(ws, r, d["seeds"], d["typeswap"], f"{name}: 同路线同时刻换车型 (救护车 vs 公交探针), best 检查点, {len(d['seeds'])} 种子")
    if "per_seed" in d:
        ws.cell(r, 1, f"{name}: 逐种子原始值 (全车类), 用公式复算均值/标准差").font = TITLE; ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8); ws.cell(r, 1).fill = TFILL; r += 1
        seeds = d["seeds"]; n = len(seeds); last = get_column_letter(1 + n)
        hdr = ["臂 / 车类 / 指标"] + [f"种子 {s}" for s in seeds] + ["均值", "标准差"]
        for c, h in enumerate(hdr, 1): ws.cell(r, c, h)
        style_row(ws, r, ncol=len(hdr), bold=True, fill=HFILL); r += 1
        for arm in ("8STD", "GS"):
            for cls in ("car", "bus", "ambulance", "all"):
                for m, ml in (("avg_stopped_time_per_visit", "停车时间/visit"), ("avg_stop_events_per_visit", "停车次数/visit")):
                    vals = d["per_seed"][arm][cls][m]; ws.cell(r, 1, f"{arm} — {cls} — {ml}")
                    for k, v in enumerate(vals): ws.cell(r, 2 + k, float(v)).number_format = "0.00"
                    ws.cell(r, 2 + n, f"=AVERAGE(B{r}:{last}{r})").number_format = "0.00"; ws.cell(r, 3 + n, f"=STDEV(B{r}:{last}{r})").number_format = "0.00"
                    style_row(ws, r, ncol=len(hdr)); r += 1
    ws.column_dimensions["A"].width = 46
    for c in "BCDEFG": ws.column_dimensions[c].width = 16
    ws.column_dimensions["H"].width = 60; ws.freeze_panes = "B4"
    return ws, d, keys

def sheet_11h_tail40(wb, ws=None):
    """Dublin 11h 尾 40 次评估 (ep605-800 每 5 集, 评估种子 123): 常量来自 plot_main_table_en_2026-09-22.py 的 S/EV。"""
    S = {"car": ((4.95, 1.24), (2.44, 0.32)), "bus": ((1.26, 0.18), (1.06, 0.14)), "amb": ((0.90, 0.69), (0.92, 0.48)), "all": ((4.75, 1.17), (2.38, 0.31)), "J": ((4.89, 1.18), (2.48, 0.31))}
    EV = {"car": ((0.57, 0.09), (0.27, 0.03)), "bus": ((0.18, 0.02), (0.15, 0.02)), "amb": ((0.14, 0.1), (0.15, 0.07)), "all": ((0.55, 0.09), (0.27, 0.03)), "J": ((0.566, 0.091), (0.281, 0.027))}
    L = {"car": "Stopped time / visit (s) - car", "bus": "Stopped time / visit (s) - bus", "amb": "Stopped time / visit (s) - ambulance", "all": "Stopped time / visit (s) - all", "J": "J(531) weighted cost (per visit)"}
    LE = {"car": "Stop events / visit - car", "bus": "Stop events / visit - bus", "amb": "Stop events / visit - ambulance", "all": "Stop events / visit - all", "J": "J(531) weighted stop events (per visit)"}
    rows_t = [[L[k], S[k][0][0], S[k][0][1], S[k][1][0], S[k][1][1], "", "40/40" if k == "J" else ""] for k in ("car", "bus", "amb", "all", "J")]
    rows_e = [[LE[k], EV[k][0][0], EV[k][0][1], EV[k][1][0], EV[k][1][1], "", "40/40" if k == "J" else ""] for k in ("car", "bus", "amb", "all", "J")]
    return rows_t, rows_e

def main():
    wb = Workbook(); ws0 = wb.active; ws0.title = "说明"
    lines = ["主对比表汇总 (2026-09-30)", "",
             "工作表: 总览 | 1x1 | 1x3 | Dublin 02h | Dublin 11h。每块: 停车时间/visit 5 行 + 保序门 + 停车次数/visit 5 行 + 保序门; 均值/标准差是测得数据, 相对差与保序门是公式。",
             "两臂: 8STD = 标准相位模板 + DQN; GS-ENUM = 枚举相位集 + FRAP。绿色单元格 = 该行两臂里更好 (更小) 的一方。",
             "口径 A (best 检查点 × 10 评估种子): 每臂取训练中 best.pth (滑动 3 次评估均值最好), 在评估种子 123-132 上各跑一次; 收集器 5 s 采样, per-visit; 配对胜 = 同种子 GS 更好的次数。",
             "口径 B (尾 40 次评估): 训练过程中 ep605-800 每 5 集一次评估 (评估种子 123 固定), 40 个不同策略同一种子; 仅 Dublin 11h 块为此口径 (与 main_comparison_8std_vs_frap_2026-09-22_en.png 的 11h 块同源)。",
             "两种口径不能直接横比: A 度量固定策略对交通随机性的稳健性, B 度量训练末段策略的波动。",
             "异常剔除只用对称、写明的规则 (任一臂任一路口全车类停车 > 阈值 → 两臂同剔); 各块标题里写了是否命中。",
             "J(531) = 按优先级 5/3/1 加权的每类 per-visit 指标, 份额按各场景车流构成 (见脚本)。",
             "救护车样本: Dublin 02h 每次评估 2 辆, 11h 4 辆, 网格 1x1/1x3 较多; Dublin 的救护车行不要单独引用均值。",
             "Dublin 11h 主块按用户选定口径: 剔除评估种子 128、132 (两臂同剔), 原因是 8STD 的 amb_0 在这两个种子被 Aungier St 右转排队回溢卡在 389281 (45 s / 90 s), 属路线撞上非 RL 让行口的外部瓶颈; 全部 10 种子的结果作为参考块保留在同一工作表, 总览里两行都有。",
             "数据源文件与生成脚本: experiments/tools/figures/build_main_comparison_xlsx_2026-09-30.py; json 见各表第 2 行。"]
    for i, t in enumerate(lines, 1): ws0.cell(i, 1, t).font = TITLE if i == 1 else N
    ws0.column_dimensions["A"].width = 160
    sheets = {}
    sheets["1x1"] = sheet_from_json(wb, "1x1", f"{FIG}/main_comparison_1x1_best10seeds_2026-09-30_en.json", amb_na=False, protocol="口径 A")
    sheets["1x3"] = sheet_from_json(wb, "1x3", f"{FIG}/main_comparison_1x3_best10seeds_2026-09-30_en.json", amb_na=False, protocol="口径 A")
    sheets["Dublin 02h"] = sheet_from_json(wb, "Dublin 02h", f"{FIG}/main_comparison_02h_best10seeds_2026-09-30_en.json", amb_na=True, protocol="口径 A; 救护车路线 = 11h 的 amb_1 路线 (两辆)")
    # Dublin 11h: 口径 A (若 json 已生成) + 口径 B
    p11 = f"{FIG}/main_comparison_11h_best10seeds_2026-09-30_en.json"
    p11s = f"{FIG}/main_comparison_11h_best10seeds_SELECTED_excl128_132_2026-09-30_en.json"
    if os.path.exists(p11s) and os.path.exists(p11):
        # 用户选定口径 (2026-09-30 用户令 "把这个作为 11 hour"): 剔除种子 128/132 (两臂同剔; 8STD 的 amb_0 在这两个种子被 Aungier St 回溢卡在 389281 45/90 s), 8 种子
        ws, d, keys = sheet_from_json(wb, "Dublin 11h", p11s, amb_na=True, protocol="口径 A, 用户选定: 剔除种子 128、132 (两臂同剔; 8STD 的 amb_0 在这两个种子被 Aungier St 回溢卡在 389281)",
                                      title_override="Dublin 11:00, best 检查点 × 8 种子, 剔除种子 128、132 (8STD: exp208 ep240, GS-ENUM: exp211 ep215)  [用户选定口径]")
        r = ws.max_row + 2
        ws, d2, keys2 = sheet_from_json(wb, "Dublin 11h (参考: 全部 10 种子)", p11, amb_na=True, protocol="口径 A, 未剔除", ws=ws, r=r,
                                        title_override="Dublin 11:00, best 检查点 × 10 种子, 全部 (8STD: exp208 ep240, GS-ENUM: exp211 ep215)  [参考]", typeswap=False)
        keys += keys2; r = ws.max_row + 2
    elif os.path.exists(p11):
        ws, d, keys = sheet_from_json(wb, "Dublin 11h", p11, amb_na=True, protocol="口径 A (best 检查点 × 10 种子) + 口径 B (尾 40 次评估) 两块")
        r = ws.max_row + 2
    else:
        ws = wb.create_sheet("Dublin 11h"); ws.cell(1, 1, "Dublin 11h: 口径 B (尾 40 次评估); 口径 A 的 json 尚未生成").font = TITLE; r = 3; keys = []
        ws.column_dimensions["A"].width = 46
        for c in "BCDEFG": ws.column_dimensions[c].width = 16
        ws.column_dimensions["H"].width = 60
    rows_t, rows_e = sheet_11h_tail40(wb)
    r, key11 = write_block(ws, r, "Dublin 11:00, 尾 40 次评估 (ep605-800 每 5 集, 评估种子 123)  (8STD: exp208, GS-ENUM: exp211)  [口径 B]", rows_t, rows_e, amb_na=True,
                           note="常量来自 plot_main_table_en_2026-09-22.py (S/EV), 与 main_comparison_8std_vs_frap_2026-09-22_en.png 的 11h 块同源; GS 配对胜 40/40 指 40 次评估里 J 逐次更好。")
    keys.append(("Dublin 11:00 尾 40 [口径 B]", key11)); sheets["Dublin 11h"] = (ws, None, keys)
    # 总览
    ov = wb.create_sheet("总览", 1); r = 1
    ov.cell(r, 1, "总览: 各场景 全体停车时间/visit 与 J(531) (公式引用各工作表)").font = TITLE; r += 1
    hdr = ["场景 / 块", "口径", "8STD 全体停车/visit", "GS-ENUM 全体停车/visit", "相对差", "8STD J(531)", "GS-ENUM J(531)", "J 相对差", "GS 配对胜 (J)", "保序门 8STD (停车时间)", "保序门 GS (停车时间)"]
    for c, h in enumerate(hdr, 1): ov.cell(r, c, h)
    style_row(ov, r, ncol=len(hdr), bold=True, fill=HFILL); r += 1
    for name, (ws, d, keys) in sheets.items():
        for t, key in keys:
            ra, rj = key[("t", "all")], key[("t", "J")]; q = f"'{ws.title}'!"
            gate_row = rj + 1  # 停车时间门紧跟 J 行
            proto = "B" if "尾 40" in t else "A"
            vals = [f"{name}: {t[:80]}", proto, f"={q}B{ra}", f"={q}D{ra}", f'=IF(C{r}=0,"",D{r}/C{r}-1)', f"={q}B{rj}", f"={q}D{rj}", f'=IF(F{r}=0,"",G{r}/F{r}-1)', f"={q}G{rj}", f"={q}B{gate_row}", f"={q}D{gate_row}"]
            for c, v in enumerate(vals, 1): ov.cell(r, c, v)
            for c in (3, 4, 6, 7): ov.cell(r, c).number_format = "0.00"
            for c in (5, 8): ov.cell(r, c).number_format = "+0%;-0%;0%"
            style_row(ov, r, ncol=len(hdr)); r += 1
    ov.column_dimensions["A"].width = 80; ov.column_dimensions["B"].width = 6
    for c in "CDEFGHIJK": ov.column_dimensions[c].width = 20
    ov.freeze_panes = "C3"
    os.makedirs(os.path.dirname(OUT), exist_ok=True); wb.save(OUT); print("saved", OUT)

if __name__ == "__main__": main()
