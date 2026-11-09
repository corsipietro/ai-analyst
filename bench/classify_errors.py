"""Classify the wrong answers of bench runs by source (data, tools, model, grey), without running the model.

Reads runs.jsonl and the transcripts of each run folder, the domain's bench.yaml (auto_checks, defects) and,
when an answer was graded, bench/grading/<name>/sheet.csv with <name>_key.csv. Prints a report on stdout;
writes no file.

The labels (defects, trap values, the SQL signals of D3) are written by us on our bench data: the counts are
indicative, not a general measure of data quality.

    python -m bench.classify_errors bench/runs/<run folder> [<run folder> ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from bench.blind import MESSAGE_HEADER, TOLERANCE, family_of, final_answer_of, has_number
from sim_analyst.checker import extract_numbers
from sim_analyst.config import DOMAINS_DIR, ROOT, load_yaml

GRADING_DIR = ROOT / "bench" / "grading"
HEADER = ("Labels written by us on our bench data (defects, trap values, D3 signals): "
          "indicative, not a general measure of data quality.")
SOURCES = ("data", "tools", "model", "grey")
SQL_LINE = re.compile(r"^\s+sql: (.*)$", re.MULTILINE)
SIGNALS = [  # (label, pattern) of SQL that removes or changes rows. Checks are not signals: COUNT(DISTINCT ...),
    # IS NULL (counting missing values). SELECT DISTINCT is not a signal either (3/11): it mostly lists values.
    ("IS NOT NULL", re.compile(r"\bIS\s+NOT\s+NULL\b", re.IGNORECASE)),
    ("DELETE", re.compile(r"\bDELETE\b", re.IGNORECASE)),
    ("UPDATE", re.compile(r"\bUPDATE\b", re.IGNORECASE)),
    ("CREATE", re.compile(r"\bCREATE\b", re.IGNORECASE)),
]


def messages(transcript: str) -> list[tuple[str, str]]:
    """The messages of a transcript as (kind, body), kind being 'Human', 'Ai' or 'Tool'."""
    headers = list(MESSAGE_HEADER.finditer(transcript))
    ends = [h.start() for h in headers[1:]] + [len(transcript)]
    return [(h.group(1), transcript[h.end():end]) for h, end in zip(headers, ends)]


def tool_outputs(transcript: str) -> str:
    """All tool outputs of a transcript, joined: what the model saw from the tools."""
    return "\n".join(body for kind, body in messages(transcript) if kind == "Tool")


def sql_signals(transcript: str) -> list[tuple[str, str]]:
    """(signal, SQL) for every SQL argument of a model call that removes or changes rows.

    Only the first line of a multi-line SQL argument is read. One pair per SQL, with its first signal.
    """
    found = []
    for kind, body in messages(transcript):
        if kind != "Ai" or "Tool Calls:" not in body:
            continue
        for sql in SQL_LINE.findall(body):
            sql = sql.strip()
            label = next((name for name, pattern in SIGNALS if pattern.search(sql)), None)
            if label:
                found.append((label, sql))
    return found


def has_exact(text: str, target: float) -> bool:
    """True if the text writes exactly the target number."""
    return any(abs(value - target) <= 1e-9 * max(1.0, abs(target)) for _, value, _ in extract_numbers(text))


def matcher(trap: float | None, target: float):
    """Exact match if the trap value is within TOLERANCE of this number (they would be confused), else has_number."""
    if trap is not None and abs(trap - target) <= TOLERANCE * abs(target):
        return has_exact
    return has_number


def contains_trap(text: str, targets: list[float], trap: float | None) -> bool:
    """True if the text writes the trap value (exactly, when it is close to one of the targets)."""
    if trap is None:
        return False
    close = any(abs(trap - t) <= TOLERANCE * abs(t) for t in targets)
    return has_exact(text, trap) if close else has_number(text, trap)


def auto_verdict(answer: str, ending: str, targets: list[float], trap: float | None,
                 declare: float | list[float] | None = None,
                 cleanable: bool = False) -> tuple[str, list[float], bool, bool]:
    """Automatic verdict: ('right', 'wrong' or 'read', targets missing, trap value present, trap declared).

    Wrong if the run ended with a data stop, if the answer is empty or if it writes the trap value.
    A missing target alone does not make it wrong (the targets help the grader; the answer may say the same
    thing in words): the answer is 'read', to be read by hand.
    Declared trap: the answer writes the trap value and also one of the declare numbers (e.g. how many rows were
    left out, or their amount). If the defect cannot be corrected with the tools (cleanable false), the declaration
    is the best possible answer: the trap value stands in for the targets close to it. If it can (cleanable true),
    the answer stays wrong, marked declared.
    """
    if ending == "data_stop" or not answer.strip():
        return "wrong", list(targets), False, False
    missing = [t for t in targets if not matcher(trap, t)(answer, t)]
    trap_in = contains_trap(answer, targets, trap)
    declares = declare if isinstance(declare, list) else [] if declare is None else [declare]
    declared = trap_in and any(has_exact(answer, d) for d in declares)
    if declared and not cleanable:
        missing = [t for t in missing if abs(trap - t) > TOLERANCE * abs(t)]
        trap_in = False
    verdict = "wrong" if trap_in else "read" if missing else "right"
    return verdict, missing, trap_in, declared


def source_of(ending: str, answer: str, missing: list[float], trap_in: bool, declared: bool, seen: str,
              trap: float | None) -> str:
    """Source of a wrong answer: data (a data stop, or the trap value written without saying so), model (no
    answer, or a trap the model declared and could have corrected), tools (a missing number never in a tool
    output), model (every missing number was in the outputs), grey (no number missing: wrong for another reason)."""
    if ending == "data_stop":
        return "data"
    if not answer.strip():
        return "model"
    if trap_in:
        return "model" if declared else "data"
    if not missing:
        return "grey"
    if any(not matcher(trap, t)(seen, t) for t in missing):
        return "tools"
    return "model"


def grading_verdicts(grading_dir: Path = GRADING_DIR) -> dict[tuple[str, str], str]:
    """{(run folder name, file): 'right' | 'wrong' | 'doubt'} from every graded sheet.

    A flag 'unit1:*' = wrong, 'unit_doubt' = doubt, no flag = right.
    """
    verdicts = {}
    for key_path in sorted(Path(grading_dir).glob("*_key.csv")):
        sheet_path = key_path.with_name(key_path.name[:-len("_key.csv")]) / "sheet.csv"
        if not sheet_path.exists():
            continue
        with open(key_path, encoding="utf-8", newline="") as f:
            key = {r["code"]: (r["run"], r["file"]) for r in csv.DictReader(f)}
        with open(sheet_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["id"] not in key:
                    continue
                flags = row.get("flags", "") or ""
                verdict = "wrong" if "unit1:" in flags else "doubt" if "unit_doubt" in flags else "right"
                verdicts[key[row["id"]]] = verdict
    return verdicts


def classify_runs(run_dirs: list[Path], domains_dir: Path = DOMAINS_DIR,
                  grading_dir: Path = GRADING_DIR) -> list[dict]:
    """One row per answer of the given run folders: verdict, its origin, source, defect and SQL signals."""
    graded = grading_verdicts(grading_dir)
    benches = {}
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        with open(run_dir / "runs.jsonl", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        for rec in records:
            domain, question = rec["domain"], rec["question"]
            if domain not in benches:
                benches[domain] = load_yaml(Path(domains_dir) / domain / "bench.yaml")
            bench = benches[domain]
            targets = list(bench.get("auto_checks", {}).get(family_of(question), []))
            defect = (bench.get("defects") or {}).get(question) or {}
            trap = defect.get("trap_value")
            name = f"{question}_r{rec['rep']}.txt"
            text = (run_dir / name).read_text(encoding="utf-8")
            ending = rec.get("ending", "ok")
            answer = final_answer_of(text)
            cleanable = bool(defect.get("cleanable", False))
            auto, missing, trap_in, declared = auto_verdict(answer, ending, targets, trap,
                                                            defect.get("declare_value"), cleanable)
            verdict = graded.get((run_dir.name, name))
            verdict_by = "graded" if verdict else "automatic"
            verdict = verdict or auto
            likely = source_of(ending, answer, missing, trap_in, declared, tool_outputs(text), trap)
            source = likely if verdict == "wrong" else "-"
            rows.append({"run": run_dir.name, "file": name, "domain": domain, "question": question,
                         "defect": defect.get("defect", ""), "cleanable": cleanable,
                         "verdict": verdict, "verdict_by": verdict_by, "auto": auto, "source": source,
                         "likely": likely, "missing": missing, "declared": declared and trap_in,
                         "signals": sql_signals(text)})
    return rows


def report(rows: list[dict]) -> str:
    """The text report: header, D1 (per question and by source), D2, D3 and the disagreements."""
    wrong = [r for r in rows if r["verdict"] == "wrong"]
    n_graded = sum(r["verdict_by"] == "graded" for r in rows)
    lines = [HEADER,
             f"Answers: {len(rows)}; wrong {len(wrong)}, to read {sum(r['verdict'] == 'read' for r in rows)}, "
             f"doubt {sum(r['verdict'] == 'doubt' for r in rows)}; "
             f"verdicts: {n_graded} graded, {len(rows) - n_graded} automatic.",
             "D1 per question:"]
    questions = sorted({(r["domain"], r["question"]) for r in rows})
    for domain, question in questions:
        group = [r for r in rows if (r["domain"], r["question"]) == (domain, question)]
        count = Counter(r["source"] for r in group if r["verdict"] == "wrong")
        by_source = ", ".join(f"{s} {count[s]}" for s in SOURCES)
        lines.append(f"  {domain} {question} [{group[0]['defect'] or 'no defect'}]: {len(group)} answers, "
                     f"{sum(count.values())} wrong ({by_source}), "
                     f"{sum(r['verdict'] == 'read' for r in group)} to read")
    total = Counter(r["source"] for r in wrong)
    lines.append("D1 wrong by source: " + ", ".join(f"{s} {total[s]}" for s in SOURCES))
    cleanable = [r for r in wrong if r["source"] == "data" and r["cleanable"]]
    share = f"{100 * len(cleanable) / len(rows):.0f}%" if rows else "-"
    per_defect = Counter(r["defect"] for r in cleanable)
    tail = "".join(f"; {d} {n}" for d, n in sorted(per_defect.items()))
    lines.append(f"D2 data and cleanable: {len(cleanable)}/{len(rows)} answers ({share}){tail}")
    declared = [r for r in wrong if r["declared"]]
    lines.append(f"Declared but correctable (wrong, source model: it saw the defect and did not correct it): "
                 f"{len(declared)}")
    lines += [f"  {r['run']}/{r['file']}" for r in declared]
    flagged = [r for r in rows if r["signals"]]
    no_defect = sum(not r["defect"] for r in flagged)
    lines.append(f"D3 unrequested interventions: {len(flagged)} answers, {no_defect} on questions without a defect "
                 "(read each: declared or silent?)")
    for r in flagged:
        tag = "defect" if r["defect"] else "no defect"
        for label, sql in r["signals"]:
            lines.append(f"  {r['run']}/{r['file']} [{tag}] {label} | {sql}")
    to_read = [r for r in rows if r["verdict"] == "read" or (r["verdict"] == "wrong" and r["source"] in ("data", "grey"))]
    lines.append(f"To read by hand (numbers missing, or wrong with source data or grey): {len(to_read)}")
    for r in to_read:
        if r["verdict"] == "read":
            missing = ", ".join(str(t) for t in r["missing"])
            lines.append(f"  {r['run']}/{r['file']} missing {missing} (if wrong: {r['likely']})")
        else:
            lines.append(f"  {r['run']}/{r['file']} wrong, source {r['source']}")
    disagree = [r for r in rows if r["verdict_by"] == "graded" and r["verdict"] in ("right", "wrong")
                and r["auto"] in ("right", "wrong") and r["verdict"] != r["auto"]]
    lines.append(f"Disagreements, graded vs automatic: {len(disagree)}")
    for r in disagree:
        lines.append(f"  {r['run']}/{r['file']} graded {r['verdict']}, automatic {r['auto']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command line: classify the given run folders and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run folders (bench/runs/<name>)")
    args = parser.parse_args()
    print(report(classify_runs(args.runs)), end="")


if __name__ == "__main__":
    main()
