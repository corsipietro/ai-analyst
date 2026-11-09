"""Replay the tool calls of finished bench runs with the CURRENT tools (no model involved, data read-only).

For every tool call the planner or explorer made in a bench folder, it runs the same call again on the same
domain and compares the old outcome (read from the transcript) with the new one. It prints one line per
call, then how many calls went from each old outcome to each new one, and the calls whose numbers changed.

Usage (from the sim_analyst/ folder):
    python -m bench.replay_tool_calls bench/runs/20261031_0107_gen_rentals_* bench/runs/20261031_0127_gen_app_*
    python -m bench.replay_tool_calls bench/runs/20261030_2247_toy_shell_*_step2     (toy numbers must not change)

Outcomes: ok, note (a result with notes), error (<kind>). A call is "different" if an old ok result had a
number that the new result no longer contains, or if the new call is an error.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

from sim_analyst.config import ROOT, load_domain
from sim_analyst.data_access import DataSource
from sim_analyst.tool_schemas import make_langchain_tools
from sim_analyst.tools import Toolbox

HEADER = re.compile(r"^=+ (.+?) =+$")
CALL = re.compile(r"^  (\w+) \((\S+)\)$")
ARG = re.compile(r"^    (\w+): ?(.*)$")
TRANSCRIPT = re.compile(r"^(.+)_r(\d+)\.txt$")
NOT_REPLAYED = ("calculate",)          # arithmetic only: nothing to replay


def parse_value(text: str):
    """An argument as printed by LangChain: JSON or a Python literal if it is one, else the plain text."""
    for reader in (json.loads, ast.literal_eval):
        try:
            return reader(text)
        except (ValueError, SyntaxError):
            continue
    return text


def outcome_of(content: str) -> str:
    """'ok', 'note' or 'error (<kind>)' for the text of one tool message."""
    try:
        data = json.loads(content)
    except ValueError:
        return "error (argument)" if content.lower().startswith("error") else "ok"
    return outcome_of_result(data)


def outcome_of_result(data) -> str:
    """The same for a result that is already a dict."""
    if isinstance(data, dict) and "error" in data:
        return f"error ({data.get('error_type', 'other')})"
    return "note" if isinstance(data, dict) and data.get("notes") else "ok"


def read_transcript(text: str) -> list[dict]:
    """The tool calls of one transcript, in order: {'tool', 'args', 'old', 'content'}; calculate is left out."""
    calls, results = [], []
    mode, current, arg = None, None, None
    for line in text.split("\n"):
        header = HEADER.match(line)
        if header:
            mode, arg = header.group(1), None
            if mode == "Tool Message":
                results.append({"name": None, "lines": []})
            continue
        if mode == "Ai Message" and line == "Tool Calls:":
            mode = "calls"
        elif mode == "calls":
            call = CALL.match(line)
            if call:
                current = {"tool": call.group(1), "args": {}}
                calls.append(current)
                arg = None
            elif line.startswith(" Call ID:") or line == "  Args:":
                continue
            elif (match := ARG.match(line)) and current is not None:
                arg = match.group(1)
                current["args"][arg] = match.group(2)
            elif arg is not None:
                current["args"][arg] += "\n" + line
        elif mode == "Tool Message":
            if line == "[REASONING]":      # the next message starts here: its text is not part of this result
                mode = "reasoning"
            elif line.startswith("Name: ") and results[-1]["name"] is None:
                results[-1]["name"] = line[6:]
            else:
                results[-1]["lines"].append(line)
    out, used = [], set()
    for call in calls:
        call["args"] = {k: parse_value(v.rstrip("\n")) for k, v in call["args"].items()}
        # The transcripts print no call id: a result belongs to the first unused call of its tool, in order.
        index = next((i for i, r in enumerate(results) if i not in used and r["name"] == call["tool"]), None)
        if index is not None:
            used.add(index)
        call["content"] = "\n".join(results[index]["lines"]).strip() if index is not None else ""
        call["old"] = outcome_of(call["content"]) if index is not None else "unknown"
        if call["tool"] not in NOT_REPLAYED:
            out.append(call)
    return out


def numbers_in(value) -> set:
    """Every number inside a JSON value, rounded to 3 decimals."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {round(float(value), 3)}
    if isinstance(value, dict):
        return set().union(*(numbers_in(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(numbers_in(v) for v in value)) if value else set()
    return set()


def run_call(tools: dict, call: dict):
    """Run one call through the LangChain tool (so the argument schema is checked too); return the result dict.

    Logs written before g5.1-d name the independent variable `time`: for the tools that now take `x`, it is renamed.
    """
    try:
        args = call["args"]
        if call["tool"] in ("summarize_results", "describe_trends", "compare_groups", "check_quality") and "time" in args and "x" not in args:
            args = {("x" if k == "time" else k): v for k, v in args.items()}
        return tools[call["tool"]].invoke(args)
    except KeyError:
        return {"error": f"unknown tool {call['tool']}", "error_type": "argument"}
    except Exception as exc:   # a schema refusal: the planner's step would have been refused too
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "error_type": "argument"}


def default_open_toolbox(domain_name: str) -> Toolbox:
    """The tools on the data of a domain (read-only)."""
    domain = load_domain(domain_name)
    return Toolbox(DataSource(domain.data), domain.settings)


def replay(folders: list[Path], open_toolbox=default_open_toolbox) -> dict:
    """Replay every call of every transcript of the folders; return the rows, the old -> new counts and the changes."""
    toolboxes, rows, different = {}, [], []
    for folder in folders:
        first = (folder / "runs.jsonl").read_text(encoding="utf-8").split("\n")[0]
        domain = json.loads(first)["domain"]
        if domain not in toolboxes:
            toolboxes[domain] = open_toolbox(domain)
        tools = {t.name: t for t in make_langchain_tools(toolboxes[domain])}
        for path in sorted(folder.glob("*_r*.txt")):
            match = TRANSCRIPT.match(path.name)
            if not match:
                continue
            for call in read_transcript(path.read_text(encoding="utf-8")):
                new_result = run_call(tools, call)
                row = {"folder": folder.name, "question": match.group(1), "rep": int(match.group(2)),
                       "tool": call["tool"], "old": call["old"], "new": outcome_of_result(new_result)}
                rows.append(row)
                if call["old"] in ("ok", "note"):
                    try:
                        lost = numbers_in(json.loads(call["content"])) - numbers_in(json.loads(json.dumps(new_result, default=str)))
                    except ValueError:
                        lost = set()
                    if lost or row["new"].startswith("error"):
                        different.append({**row, "numbers_lost": sorted(lost)[:5]})
    counts = {}
    for row in rows:
        counts.setdefault(row["old"], Counter())[row["new"]] += 1
    return {"rows": rows, "counts": {old: dict(new) for old, new in counts.items()}, "different": different}


def format_report(report: dict) -> str:
    """The report as a compact text table."""
    lines = [f"{r['question']:>4} r{r['rep']} {r['tool']:<18} {r['old']} -> {r['new']}" for r in report["rows"]]
    lines.append("")
    lines.append("old -> new:")
    for old, new in sorted(report["counts"].items()):
        lines.append(f"  {old}: " + ", ".join(f"{n} {kind}" for kind, n in sorted(new.items())))
    if report["different"]:
        lines.append("")
        lines.append("calls whose result differs (an old number is missing, or the call now fails):")
        lines += [f"  {d['folder']} {d['question']} r{d['rep']} {d['tool']}: {d['old']} -> {d['new']}"
                  + (f", numbers lost {d['numbers_lost']}" if d["numbers_lost"] else "") for d in report["different"]]
    return "\n".join(lines)


def resolve(text: str) -> Path:
    """A bench folder given as a path, or just its name (looked up in bench/runs)."""
    path = Path(text)
    return path if path.exists() else ROOT / "bench" / "runs" / path.name


def main(argv: list[str] | None = None) -> None:
    """Print the replay of the folders given on the command line."""
    folders = [resolve(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not folders:
        raise SystemExit(__doc__)
    print(format_report(replay(folders)))


if __name__ == "__main__":
    main()
