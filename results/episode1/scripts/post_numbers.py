"""Numbers of the episode 1 post: one row per setup (2 models x 3 strategies, g5 tools, 5 domains, 42 answers).

Run from results/episode1/runs:
    python3 ../scripts/post_numbers.py > ../summary.csv
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_points import checks, final   # same keyword checks as hard_points.py

CELLS = [("12B", "ReAct", "react"), ("12B", "ReWOO", "rewoo"), ("12B", "ReWOO+repair", "repair"),
         ("26B", "ReAct", "react_26b"), ("26B", "ReWOO", "rewoo_26b"), ("26B", "ReWOO+repair", "repair_26b")]
CHECKED = {"summarize_results", "compare_groups", "describe_trends", "check_quality"}  # checked tools
rows = []
for model, strat, tag in CELLS:
    dirs = [d for d in sorted(glob.glob(f"*_g5_{tag}")) if os.path.isfile(d + "/runs.jsonl")]
    R, hard, nhard = [], 0, 0
    for d in dirs:
        rr = [json.loads(l) for l in open(d + "/runs.jsonl")]; R += rr
        dom = rr[0]["domain"] if rr else None
        for f in sorted(glob.glob(d + "/*_r*.txt")):
            q = os.path.basename(f)[0]; t = final(f)
            for lab, pats in checks.get((dom, q), []):
                hard += all(re.search(p, t, re.I) for p in pats); nhard += 1
    if not R: rows.append([model, strat, 0]); continue
    tc = {}
    for r in R:
        for k, v in r.get("tool_calls_per_tool", {}).items(): tc[k] = tc.get(k, 0) + v
    reads = sum(v for k, v in tc.items() if k in CHECKED or k == "query_results")
    tok = sum(v.get("tokens_written", 0) for r in R for v in r.get("per_node", {}).values())
    rows.append([model, strat, len(R), sum(r["ending"] == "ok" and not r.get("empty_answer") for r in R),
                 f"{hard}/{nhard}", round(sum(r["seconds"] for r in R) / 60), round(tok / 1000),
                 sum(r.get("steps_failed", 0) for r in R), sum(r.get("tool_errors", 0) for r in R),
                 tc.get("query_results", 0), reads, f"{100*tc.get('query_results',0)/max(reads,1):.0f}%",
                 tc.get("check_quality", 0), sum(r.get("numbers_untraceable", 0) for r in R)])
head = ["model", "strategy", "answers_run", "answers_nonempty", "hard_points", "minutes", "ktokens_written",
        "steps_failed", "tool_errors", "raw_sql_reads", "all_data_reads", "raw_sql_share", "check_quality", "numbers_untraceable"]
w = csv.writer(sys.stdout); w.writerow(head); [w.writerow(r) for r in rows]
print("NOTE: hard_points = automatic keyword count (raw); steps_failed only for ReWOO graphs; ReAct counts tool_errors.", file=sys.stderr)
