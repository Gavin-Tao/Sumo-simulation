"""NL -> IR compiler: LLM with a deterministic-validator retry loop and a
refuse-to-guess protocol (ambiguous input -> clarification object, never a
guessed policy).

Providers:
  --provider manual   print the prompt; read the IR from --ir-in (full
                      pipeline testable with zero LLM access)
  --provider openai   OpenAI-compatible HTTP (env OPENAI_API_KEY, --base-url
                      overridable — works for any compatible endpoint)
  --provider ollama   local ollama (--model, default llama3)

Output: validated IR YAML at --out, or a clarification request on stdout
(exit 3). Validator errors are fed back verbatim for up to --retries rounds.

Usage:
  python compile_policy.py --nl "救护车绝对优先; 公交高峰提到4级" \
      --provider ollama --out policy.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policy_dsl  # noqa: E402

FEW_SHOTS = [
    ("Ambulances always get absolute priority; everything else is ordinary "
     "traffic. Guarantee the ambulance priority strictly.",
     {"version": 1,
      "assignments": [
          {"match": {"type": "ambulance"}, "level": 5},
          {"match": {"type": "*"}, "level": 1}],
      "guarantees": {"lexicographic_min_level": 5}}),
    ("公交在早高峰(前30分钟)提到4级, 其余时间3级; 救护车5级; 其他1级。"
     "要求公交P90等待低于40秒。",
     {"version": 1,
      "assignments": [
          {"match": {"type": "ambulance"}, "level": 5},
          {"match": {"type": "bus", "time_window": [0, 1800]}, "level": 4},
          {"match": {"type": "bus"}, "level": 3},
          {"match": {"type": "*"}, "level": 1}],
      "kpis": [{"metric": "bus_wait_p90", "op": "<", "value": 40}]}),
    ("In the hospital zone give ambulances level 5 and demote cars to 1; "
     "elsewhere default: bus 3, car 1.",
     {"version": 1,
      "assignments": [
          {"match": {"type": "ambulance"}, "level": 5},
          {"match": {"type": "bus"}, "level": 3},
          {"match": {"type": "*"}, "level": 1}],
      "regions": {"hospital_zone": []}}),
]

SYSTEM = f"""You compile natural-language traffic priority policies into a
strict YAML/JSON IR. Output ONLY a JSON object, no prose.

DSL SPECIFICATION:
{policy_dsl.DSL_SPEC_FOR_PROMPT}

RULES:
1. Output exactly one JSON object: either a valid IR, or, if the request is
   ambiguous (unspecified thresholds, contradictions, unknown vehicle types),
   {{"clarification_needed": "<one concrete question>"}} — NEVER guess.
2. The last assignment MUST be the catch-all {{"match": {{"type": "*"}}, ...}}.
3. Levels are integers 1..5. Time windows are simulation seconds [start,end).
4. Only use match keys: type, time_window, region, share_lt.
   (sched_delay_min_gt exists but is not yet executable — avoid it.)
"""


def build_prompt(nl, feedback=None):
    shots = "\n\n".join(
        f"POLICY: {p}\nIR: {json.dumps(ir, ensure_ascii=False)}"
        for p, ir in FEW_SHOTS)
    fb = f"\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED BY THE VALIDATOR:\n{feedback}\n" \
        if feedback else ""
    return f"{shots}\n\nPOLICY: {nl}{fb}\nIR:"


def call_llm(provider, model, base_url, prompt):
    import requests
    if provider == "openai":
        url = (base_url or "https://api.openai.com/v1") + "/chat/completions"
        r = requests.post(url, timeout=120,
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={"model": model or "gpt-4o-mini",
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": prompt}],
                  "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if provider == "ollama":
        url = (base_url or "http://localhost:11434") + "/api/chat"
        r = requests.post(url, timeout=300,
            json={"model": model or "llama3", "stream": False,
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["message"]["content"]
    raise ValueError(provider)


def extract_json(text):
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON object found in LLM output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", required=True)
    ap.add_argument("--provider", choices=["manual", "openai", "ollama"],
                    default="manual")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--ir-in", default=None, help="manual mode: IR file to validate")
    ap.add_argument("--out", default="policy.yaml")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if args.provider == "manual":
        if not args.ir_in:
            print("=== SYSTEM PROMPT ===\n" + SYSTEM)
            print("=== USER PROMPT ===\n" + build_prompt(args.nl))
            print("\n(manual mode: paste the model's JSON into a file and "
                  "re-run with --ir-in <file>)")
            return
        ir = policy_dsl.load(args.ir_in)
        ok, errs, warns = policy_dsl.validate(ir)
        for w in warns:
            print("WARN:", w)
        if not ok:
            for e in errs:
                print("ERROR:", e)
            sys.exit(1)
        policy_dsl.dump(ir, args.out)
        print(f"VALID -> {args.out}")
        return

    feedback = None
    for attempt in range(1, args.retries + 1):
        raw = call_llm(args.provider, args.model, args.base_url,
                       build_prompt(args.nl, feedback))
        try:
            obj = extract_json(raw)
        except Exception as e:
            feedback = f"output was not parseable JSON: {e}"
            print(f"attempt {attempt}: {feedback}")
            continue
        if "clarification_needed" in obj:
            print("CLARIFICATION NEEDED:", obj["clarification_needed"])
            sys.exit(3)
        ok, errs, warns = policy_dsl.validate(obj)
        for w in warns:
            print("WARN:", w)
        if ok:
            policy_dsl.dump(obj, args.out)
            print(f"VALID (attempt {attempt}) -> {args.out}")
            return
        feedback = "\n".join(errs)
        print(f"attempt {attempt} rejected:\n{feedback}")
    sys.exit(f"compilation failed after {args.retries} attempts")


if __name__ == "__main__":
    main()
