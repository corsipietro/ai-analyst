"""Blind grading: copy transcripts under anonymous codes, grade them, then join the grades to their origin.

Usage (from the sim_analyst/ folder):
    python -m bench.blind make --domain toy_shell --runs bench/runs/<folder1> [<folder2> ...] --name <name> [--seed N]
    python -m bench.blind unblind --name <name>

`make` writes bench/grading/<name>/G001.txt, ... (the transcripts, shuffled), bench/grading/<name>/sheet.csv
(to fill in: criteria A1-A6 and B1-B5 as 0, 1, 2 or NA, flags separated by ';', notes) and, OUTSIDE the
grading folder, bench/grading/<name>_key.csv (code -> run folder and file). The person grading never
opens the key. `unblind` joins sheet and key and writes bench/grading/<name>_results.csv.

The sheet also has one yes/no column per key number of the question's family (auto_checks in bench.yaml):
whether the final answer contains that number, within 2%, and an `answer` column (yes/no: no when the run
ends with a tool call, an empty message or a tool result). They help the grading; they are not grades.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from sim_analyst.checker import extract_numbers
from sim_analyst.config import DOMAINS_DIR, ROOT, load_yaml

GRADING_DIR = ROOT / "bench" / "grading"
CRITERIA = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 6)]
TOLERANCE = 0.02                      # relative tolerance when looking for a key number
MESSAGE_HEADER = re.compile(r"^=+ (\w+) Message =+$", re.MULTILINE)
VALID_GRADES = {"0", "1", "2", "NA"}
REASONING = re.compile(r"\[REASONING\].*?\[/REASONING\]\n?", re.DOTALL)
TRANSCRIPT = re.compile(r"^(?P<question>.+)_r(?P<rep>\d+)\.txt$")


def family_of(question_id: str) -> str:
    """The family of a question is the first character of its id ('1a' -> '1')."""
    return question_id[0]


def final_answer_of(transcript: str) -> str:
    """The final answer: the text of the last message, if it is a model message that is an answer.

    Empty (no answer) if the transcript ends with a tool result, or with a model message that has
    no text or asks for a tool. The [REASONING] block is removed.
    """
    headers = list(MESSAGE_HEADER.finditer(transcript))
    if not headers or headers[-1].group(1) != "Ai":
        return ""
    body = REASONING.sub("", transcript[headers[-1].end():]).strip()
    if "Tool Calls:" in body.splitlines():
        return ""
    return body


def has_number(answer: str, target: float) -> bool:
    """True if the answer writes a number within TOLERANCE (relative) of the target."""
    return any(abs(value - target) <= TOLERANCE * abs(target) for _, value, _ in extract_numbers(answer))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to a CSV file with a header."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file into a list of dicts."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make(run_dirs: list[Path], name: str, auto_checks: dict, grading_dir: Path = GRADING_DIR, seed: int = 1) -> None:
    """Copy every transcript of the run folders under a random code and write the sheet and the key."""
    files = [(folder, path) for folder in run_dirs for path in sorted(folder.glob("*.txt"))
             if TRANSCRIPT.match(path.name)]
    random.Random(seed).shuffle(files)
    target = grading_dir / name
    target.mkdir(parents=True)
    numbers = sorted({n for family in auto_checks.values() for n in family})
    check_columns = [f"has_{n}" for n in numbers]
    sheet, key = [], []
    for i, (folder, path) in enumerate(files, start=1):
        code = f"G{i:03d}"
        shutil.copyfile(path, target / f"{code}.txt")
        question = TRANSCRIPT.match(path.name)["question"]
        family = family_of(question)
        answer = final_answer_of(path.read_text(encoding="utf-8"))
        row = {"id": code, "question": question, "family": family, "answer": "yes" if answer else "no"}
        for n, column in zip(numbers, check_columns):
            # empty when the number is not a key number of this family
            row[column] = ("yes" if has_number(answer, n) else "no") if n in auto_checks.get(family, []) else ""
        row.update({c: "" for c in CRITERIA})
        row.update({"flags": "", "notes": ""})
        sheet.append(row)
        key.append({"code": code, "run": folder.name, "file": path.name})
    write_csv(target / "sheet.csv", ["id", "question", "family", "answer"] + check_columns + CRITERIA + ["flags", "notes"], sheet)
    write_csv(grading_dir / f"{name}_key.csv", ["code", "run", "file"], key)


def unblind(name: str, grading_dir: Path = GRADING_DIR) -> Path:
    """Join sheet and key; sum the points per run folder, family and criterion; count the flags."""
    key = {r["code"]: r["run"] for r in read_csv(grading_dir / f"{name}_key.csv")}
    totals = defaultdict(lambda: {"points": 0, "graded": 0, "na": 0})
    flag_counts = defaultdict(int)
    for row in read_csv(grading_dir / name / "sheet.csv"):
        run, family = key[row["id"]], row["family"]
        for criterion in CRITERIA:
            cell = row[criterion].strip()
            if cell and cell not in VALID_GRADES:
                raise ValueError(f"{row['id']}: criterion {criterion} has {cell!r}; "
                                 "allowed: 0, 1, 2, NA or an empty cell")
            count = totals[(run, family, criterion)]    # created even if nothing is graded
            if cell == "NA":
                count["na"] += 1
            elif cell:
                count["points"] += int(cell)
                count["graded"] += 1
        for flag in filter(None, (f.strip() for f in row["flags"].split(";"))):
            flag_counts[(run, family, f"flag:{flag}")] += 1
    rows = [{"run": r, "family": f, "item": c, "points": t["points"], "graded": t["graded"], "na": t["na"],
             "flag_count": ""} for (r, f, c), t in sorted(totals.items())]
    rows += [{"run": r, "family": f, "item": item, "points": "", "graded": "", "na": "", "flag_count": n}
             for (r, f, item), n in sorted(flag_counts.items())]
    out = grading_dir / f"{name}_results.csv"
    write_csv(out, ["run", "family", "item", "points", "graded", "na", "flag_count"], rows)
    return out


def main() -> None:
    """Command line: make | unblind."""
    parser = argparse.ArgumentParser(description="Blind grading of bench transcripts.")
    sub = parser.add_subparsers(dest="command", required=True)
    make_parser = sub.add_parser("make", help="anonymize transcripts and write the grading sheet and the key")
    make_parser.add_argument("--domain", required=True)
    make_parser.add_argument("--runs", nargs="+", required=True, type=Path, help="run folders")
    make_parser.add_argument("--name", required=True)
    make_parser.add_argument("--seed", type=int, default=1)
    unblind_parser = sub.add_parser("unblind", help="join the grades to the key and sum them")
    unblind_parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "make":
        bench = load_yaml(DOMAINS_DIR / args.domain / "bench.yaml")
        make(args.runs, args.name, {str(k): v for k, v in bench["auto_checks"].items()}, seed=args.seed)
        print(f"written bench/grading/{args.name}/ and bench/grading/{args.name}_key.csv")
    else:
        try:
            print(f"written {unblind(args.name)}")
        except ValueError as error:
            raise SystemExit(f"invalid sheet: {error}")


if __name__ == "__main__":
    main()
