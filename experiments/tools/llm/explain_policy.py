"""M5 — grounded policy explanation report (template version).

Renders a human-readable markdown report from verify_policy.py output
(+ optional KAN metrics when available). GROUNDING DISCIPLINE: every number
in the report is read from a real metrics key — nothing is generated.
An optional LLM pass may later polish the PROSE ONLY (numbers stay
template-injected); not needed for v1.

Usage: python explain_policy.py --report <verify.json> [--kan-metrics <json>]
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--kan-metrics", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = json.load(open(args.report))
    m, b = r["metrics"], r.get("baseline_metrics")
    lines = ["# 政策验证报告", "",
             f"- 场景配置: `{os.path.basename(r['config'])}`",
             f"- 政策 IR: `{os.path.basename(r['ir'])}`  控制器: {r['controller']}  种子: {r['seed']}",
             f"- 改型统计: 入网 {r['retype_stats']['seen']} 辆, 政策改型 "
             f"{r['retype_stats']['retyped']} 辆 "
             f"({r['retype_stats']['by_derived'] or '无'})", "",
             "## 分类别结果 (平均/P90 累积等待, 秒)", "",
             "| 类别 | 数量 | 均值 | P90 |" + (" Δ均值 vs 基线 |" if b else ""),
             "|---|---|---|---|" + ("---|" if b else "")]
    classes = sorted({k.rsplit("_", 2)[0] for k in m if k.endswith("_wait_mean")})
    for c in classes:
        row = (f"| {c} | {m.get(f'{c}_count', '?')} "
               f"| {m.get(f'{c}_wait_mean', float('nan')):.1f} "
               f"| {m.get(f'{c}_wait_p90', float('nan')):.1f} |")
        if b:
            base = b.get(f"{c}_wait_mean")
            if base:
                d = (m.get(f"{c}_wait_mean", base) - base) / base
                row += f" {d:+.1%} |"
            else:
                row += " n/a |"
        lines.append(row)
    lines += ["", "## KPI 断言", ""]
    for k in r["kpis"]:
        lines.append(f"- **{k['status']}** `{k['metric']} {k['op']} {k['value']}`"
                     + (f" (观测 {k.get('observed')})" if "observed" in k else ""))
    lines += ["", f"## 判决: **{r['verdict']}**"]
    if args.kan_metrics and os.path.exists(args.kan_metrics):
        km = json.load(open(args.kan_metrics))
        lines += ["", "## 策略内化的有效优先权重 (KAN shared-levels 读出)", ""]
        if "alpha_normalized_l1" in km:
            lines.append("等级 1..5 相对权重: "
                         + ", ".join(f"l{i+1}={a:.2f}" for i, a in
                                     enumerate(km["alpha_normalized_l1"]))
                         + f"  (拟合保真 R²={km.get('r2_test', float('nan')):.3f})")
    out = args.out or os.path.splitext(args.report)[0] + ".md"
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nreport ->", out)


if __name__ == "__main__":
    main()
