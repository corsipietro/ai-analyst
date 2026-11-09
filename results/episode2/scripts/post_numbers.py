"""Numbers of the episode 2 post, from the g5.1 bench only (6 setups: 12B/26B x react/rewoo/repair).

Reads runs.jsonl and the transcripts of results/episode2/runs/*_g51_<strategy>_<size>, plus the graded verdicts in
results/episode2/grading (through bench.classify_errors). Prints every number of the post with its source, and
writes the numbers the carousel draws to results/episode2/numbers.json. No model is run.

    python3 -B results/episode2/scripts/post_numbers.py      # from the repository root
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bench.classify_errors import classify_runs  # noqa: E402

CELLS = [f"g51_{s}_{m}" for m in ("12b", "26b") for s in ("react", "rewoo", "repair")]
PLAN_NODES = ("explore", "plan")  # "understanding the data and planning"
FREE_SQL = "query_results"        # the tool that takes free SQL: it bypasses the checked tools
JOIN = re.compile(r"\bJOIN\b", re.IGNORECASE)
PHASES = {"explore": "plan", "plan": "plan", "interpret": "answer", "analyze": "answer", "repair": "repair"}
EPISODE = ROOT / "results" / "episode2"
RUNS = EPISODE / "runs"
GRADING = EPISODE / "grading"
OUT = EPISODE / "numbers.json"

# Sources fixed by reading the final answers (4/11). classify_errors gives the source only from its
# automatic rule, even when the verdict was graded by hand.
#  - 4a: the AI names the duplicated station and keeps the inflated count -> source model (it could correct).
#  - 4a 26B repair r2: the automatic rule says "declared" (sum 5202), but the answer never mentions the duplicate.
SOURCE_FIX = {
    ("g51_rewoo_12b", "4a_r1.txt"): "model", ("g51_rewoo_12b", "4a_r2.txt"): "model",
    ("g51_repair_12b", "4a_r1.txt"): "model", ("g51_repair_12b", "4a_r2.txt"): "model",
    ("g51_repair_26b", "4a_r2.txt"): "data",
}


def run_dirs(cell: str) -> list[Path]:
    """The run folders of one cell (5 datasets), probes excluded."""
    return sorted(p for p in RUNS.glob(f"*_{cell}") if p.is_dir())


def records(dirs: list[Path]) -> list[dict]:
    """All runs.jsonl records of the given folders, with the folder kept in '_dir'."""
    out = []
    for d in dirs:
        for line in open(d / "runs.jsonl", encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                rec["_dir"] = d
                out.append(rec)
    return out


def node_seconds(rec: dict, nodes: tuple[str, ...]) -> float:
    """Seconds (reading + writing) spent in the given graph nodes."""
    per = rec.get("per_node", {})
    return sum(per[n]["seconds_reading"] + per[n]["seconds_writing"] for n in nodes if n in per)


def span(values: list[float], fmt: str = "{:.0f}") -> str:
    """'min-max' of a list of per-cell values."""
    return f"{fmt.format(min(values))}-{fmt.format(max(values))}"


def main() -> None:
    cells = {c: run_dirs(c) for c in CELLS}
    recs = {c: records(d) for c, d in cells.items()}

    print("== 1. Totals (source: runs.jsonl)")
    for c in CELLS:
        print(f"  {c}: {len(cells[c])} folders, {len(recs[c])} answers")
    print(f"  total answers: {sum(len(r) for r in recs.values())}")

    print("\n== 2. Where the work goes (per cell: sums over the answers)")
    writing, plan_share, minutes, written, read_tok = [], [], [], [], {}
    for c in CELLS:
        rs = recs[c]
        tot = sum(r["seconds"] for r in rs)
        writing.append(100 * sum(r["seconds_writing"] for r in rs) / tot)
        plan_share.append(100 * sum(node_seconds(r, PLAN_NODES) for r in rs) / tot)
        minutes.append(tot / len(rs) / 60)
        written.append(sum(r["tokens_written"] for r in rs) / len(rs) / 1000)
        read_tok[c] = sum(sum(n.get("tokens_read", 0) for n in r["per_node"].values()) for r in rs) / len(rs) / 1000
        print(f"  {c}: writing {writing[-1]:.0f}%, explore/plan {plan_share[-1]:.0f}%, "
              f"{minutes[-1]:.1f} min/answer, {written[-1]:.1f}k written, {read_tok[c]:.0f}k read per answer")
    print(f"  RANGE writing {span(writing)}%; explore/plan {span(plan_share)}%; "
          f"min/answer {span(minutes, '{:.1f}')}; k tokens written {span(written, '{:.1f}')}")

    print("\n== 3. Errors (source: classify_errors + graded verdicts in results/episode2/grading)")
    rows = []
    for c in CELLS:
        for r in classify_runs(cells[c], grading_dir=GRADING):
            r["cell"] = c
            if r["verdict"] == "wrong" and (c, r["file"]) in SOURCE_FIX and r["question"] == "4a":
                r["source"] = SOURCE_FIX[(c, r["file"])]
            rows.append(r)
    by_q = Counter((r["domain"], r["question"]) for r in rows)
    wrong = [r for r in rows if r["verdict"] == "wrong"]
    doubt = [r for r in rows if r["verdict"] in ("doubt", "read")]
    clean = [r for r in rows if not r["defect"]]
    clean_wrong = [r for r in clean if r["verdict"] == "wrong"]
    print(f"  answers {len(rows)}, wrong {len(wrong)}, doubt/to read {len(doubt)}")
    print(f"  questions: {len(by_q)}; with defect: "
          + ", ".join(sorted({f"{r['domain']} {r['question']} ({r['defect']})" for r in rows if r["defect"]})))
    print(f"  CLEAN questions: {len(clean_wrong)}/{len(clean)} wrong "
          f"({100 * len(clean_wrong) / len(clean):.1f}%, 1 in {len(clean) / max(len(clean_wrong), 1):.0f})")
    for dq in sorted({(r["domain"], r["question"]) for r in rows if r["defect"]}):
        g = [r for r in rows if (r["domain"], r["question"]) == dq]
        w = [r for r in g if r["verdict"] == "wrong"]
        size = Counter("26b" if "26b" in r["run"] else "12b" for r in w)
        print(f"  {dq[0]} {dq[1]}: {len(w)}/{len(g)} wrong (12B {size['12b']}, 26B {size['26b']}); "
              f"sources {dict(Counter(r['source'] for r in w))}; declared-but-not-corrected "
              f"{sum(r['declared'] for r in w)}; doubt {sum(r['verdict'] in ('doubt', 'read') for r in g)}")
    by_source = Counter(r["source"] for r in wrong)
    by_source_noq4 = Counter(r["source"] for r in wrong if r["question"] != "4a")
    print(f"  wrong by source, all: {dict(by_source)}; without 4a: {dict(by_source_noq4)}")

    print("\n== 4. Bypassed tools on defect questions (free SQL = query_results with a JOIN)")
    index = {(str(r["_dir"].name), f"{r['question']}_r{r['rep']}.txt"): r for c in CELLS for r in recs[c]}
    for q in ("3a", "4a"):
        g = [r for r in rows if r["domain"] == "gen_rentals" and r["question"] == q]
        for verdict in ("wrong", "right"):
            sub = [r for r in g if r["verdict"] == verdict]
            free = [r for r in sub if index[(r["run"], r["file"])]["tool_calls_per_tool"].get(FREE_SQL)]
            join = [r for r in free if any(JOIN.search(line) for line in
                    (RUNS / r["run"] / r["file"]).read_text(encoding="utf-8").splitlines()
                    if "sql" in line.lower())]
            print(f"  {q} {verdict}: {len(sub)}; with free SQL {len(free)}; free SQL with JOIN {len(join)}")

    print("\n== 5. Free SQL share of tool calls, and failed calls (per cell)")
    for c in CELLS:
        calls = sum(r["tool_calls"] for r in recs[c])
        free = sum(r["tool_calls_per_tool"].get(FREE_SQL, 0) for r in recs[c])
        errs = sum(r["tool_errors"] for r in recs[c])
        kinds = Counter()
        for r in recs[c]:
            kinds.update(r.get("tool_errors_per_kind", {}))
        print(f"  {c}: free SQL {100 * free / calls:.0f}% of {calls}; failed {100 * errs / calls:.0f}% {dict(kinds)}")

    print("\n== 3b. Wrong by source per cell (4a fixed by hand)")
    out = {"answers": len(rows), "per_cell": {}}
    for c in CELLS:
        w = Counter(r["source"] for r in rows if r["cell"] == c and r["verdict"] == "wrong")
        n = sum(r["cell"] == c for r in rows)
        print(f"  {c}: {dict(w)} of {n}")
        tok = {"plan": [0, 0], "answer": [0, 0], "repair": [0, 0]}
        for r in recs[c]:
            for node, v in r["per_node"].items():
                tok[PHASES[node]][0] += v.get("tokens_read", 0)
                tok[PHASES[node]][1] += v["tokens_written"]
        out["per_cell"][c] = {"answers": n, "wrong": dict(w),
                              "tokens_per_answer": {k: {"read": v[0] / len(recs[c]), "written": v[1] / len(recs[c])}
                                                    for k, v in tok.items()},
                              "empty": sum(r["empty_answer"] for r in recs[c])}
    q = lambda dq: [r for r in rows if r["domain"] == "gen_rentals" and r["question"] == dq]
    out["clean"] = {"wrong": len(clean_wrong), "n": len(clean)}
    out["same_flaw"] = {"untouched": {"wrong": sum(r["verdict"] == "wrong" for r in q("2a")), "n": len(q("2a"))},
                        "linked": {"wrong": sum(r["verdict"] == "wrong" for r in q("4a")), "n": len(q("4a"))},
                        "saw_it": sum(r["source"] == "model" for r in q("4a") if r["verdict"] == "wrong")}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  written {OUT}")

    print("\n== 6. Getting stuck (per cell)")
    for c in CELLS:
        rs = recs[c]
        print(f"  {c}: empty answers {sum(r['empty_answer'] for r in rs)}, "
              f"retries {sum(r.get('empty_retries', 0) for r in rs)}, "
              f"step limits {sum(r.get('length_stops', 0) for r in rs)}")


if __name__ == "__main__":
    main()
