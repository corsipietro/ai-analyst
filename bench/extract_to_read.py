"""For each answer to read by hand (bench/classify_errors.py), print only the sentences that bear on it.

Same run folders and same list as classify_errors ("To read by hand"); reads files, writes nothing, runs no model.
For each missing number: the line of bench.yaml (targets or expected) that holds it, and at most 3 sentences of the
final answer that write a number within 10% of it or share words with that line. For a wrong answer with source
data: the sentences that write the trap value. The person reading writes the verdict.

    python -m bench.extract_to_read bench/runs/<run folder> [<run folder> ...]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from bench.blind import final_answer_of
from bench.classify_errors import classify_runs
from sim_analyst.checker import extract_numbers
from sim_analyst.config import DOMAINS_DIR, load_yaml

NEAR = 0.10          # relative distance for "a number close to the missing one"
MAX_SENTENCES = 3    # sentences shown per missing number
MAX_CHARS = 220      # a sentence is cut after this many characters
STOP = {"which", "their", "there", "these", "those", "about", "after", "every", "other", "where", "while"}


def target_lines(domain: str) -> list[str]:
    """Every line of bench.yaml that says what is true: targets (gen_*) or expected points (toy_shell)."""
    bench = load_yaml(DOMAINS_DIR / domain / "bench.yaml")
    lines = [line for family in (bench.get("targets") or {}).values() for line in family]
    lines += [point["text"] for point in (bench.get("expected") or {}).values()]
    return lines


def line_of(number: float, lines: list[str]) -> str:
    """The first target line that writes the number (as in bench.yaml), or '' if none does."""
    for line in lines:
        if any(abs(value - number) <= 1e-9 * max(1.0, abs(number)) for _, value, _ in extract_numbers(line)):
            return line
    return ""


def words(text: str) -> set[str]:
    """Lower-case words of at least 5 letters, without a few common ones."""
    return {w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in STOP}


def sentences(answer: str) -> list[str]:
    """The answer cut at line ends and at the end of each sentence."""
    parts = re.split(r"\n+|(?<=[.!?])\s+", answer)
    return [p.strip() for p in parts if p.strip()]


def relevant(answer: str, number: float, line: str) -> list[str]:
    """At most MAX_SENTENCES sentences: first those with a number within NEAR of it, then those sharing words."""
    keys = words(line)
    scored = []
    for s in sentences(answer):
        near = any(abs(value - number) <= NEAR * abs(number) for _, value, _ in extract_numbers(s))
        shared = len(keys & words(s))
        if near or shared >= 2:
            scored.append((near, shared, s))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s[:MAX_CHARS] for _, _, s in scored[:MAX_SENTENCES]]


def main() -> None:
    """Print one block per answer to read."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run folders (bench/runs/<name>)")
    run_dirs = parser.parse_args().runs
    rows = classify_runs(run_dirs)
    folders = {Path(d).name: Path(d) for d in run_dirs}
    lines_cache = {}
    for r in rows:
        data_or_grey = r["verdict"] == "wrong" and r["source"] in ("data", "grey")
        if r["verdict"] != "read" and not data_or_grey:
            continue
        answer = final_answer_of((folders[r["run"]] / r["file"]).read_text(encoding="utf-8"))
        lines = lines_cache.setdefault(r["domain"], target_lines(r["domain"]))
        state = f"wrong, source {r['source']}" if data_or_grey else "missing " + ", ".join(f"{t:g}" for t in r["missing"])
        print(f"\n## {r['run']}/{r['file']} | {r['domain']} {r['question']} | {state} | answer {len(answer)} chars")
        if data_or_grey:
            defect = (load_yaml(DOMAINS_DIR / r["domain"] / "bench.yaml").get("defects") or {}).get(r["question"], {})
            trap = defect.get("trap_value")
            if trap is not None:
                print(f"  trap {trap:g} ({defect.get('defect', '')}); declared: {r['declared']}")
                for s in relevant(answer, float(trap), ""):
                    print(f"  > {s}")
            continue
        if not answer:
            print("  (empty answer)")
            continue
        by_line = {}   # one block per target line: numbers that share it are shown together
        for number in r["missing"]:
            by_line.setdefault(line_of(number, lines), []).append(number)
        for line, numbers in by_line.items():
            print(f"  [{', '.join(f'{n:g}' for n in numbers)}] expected: {line[:MAX_CHARS] or '(not found in bench.yaml)'}")
            found = []
            for number in numbers:
                found += [s for s in relevant(answer, number, line) if s not in found]
            for s in found[:MAX_SENTENCES] or ["(nothing extracted: read the whole answer)"]:
                print(f"  > {s}")

if __name__ == "__main__":
    main()
