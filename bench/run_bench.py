"""Generic test bench: asks a domain's bench questions and records everything needed to grade them.

Usage (from the repository root):
    python -m bench.run_bench --domain toy_shell --reps 2 3a 3b 3c    (some questions)
    python -m bench.run_bench --domain toy_shell --reps 2            (all questions)

For each question x repetition (seed = repetition number) it saves the transcript with the hidden
reasoning, one JSONL record (versions, settings, timings, tool use, tool errors by kind, numbers
checked) and a log. Grading is done outside the bench, against bench.yaml: the bench measures the SYSTEM
(model + tools + instructions + context), not the model alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from sim_analyst.checker import check_numbers
from sim_analyst.config import PROMPTS_DIR, ROOT, load_yaml, parse_override_items, parse_overrides, sha256_of
from sim_analyst.graph import HANDOFF_MARK, open_analyst
from sim_analyst.graphs import ANSWER_ROLE, build_graph, graph_name
from sim_analyst.tools import TOOLS_VERSION

OUT_DIR = ROOT / "bench" / "runs"
log = logging.getLogger("bench")
UNKNOWN_NODE = "unknown"   # a model message without a node name
CALL_FIELDS = ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration")


def setup_logging(run_dir: Path) -> None:
    """INFO and above on the terminal; everything (DEBUG too) in run_dir/bench.log."""
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    to_screen = logging.StreamHandler()
    to_screen.setLevel(logging.INFO)
    to_file = logging.FileHandler(run_dir / "bench.log", encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    for handler in (to_screen, to_file):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.DEBUG)


def text_hash(text: str) -> str:
    """First 12 hex digits of the SHA-256 of a text: enough to tell two versions apart."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def timing(message) -> str:
    """'(think | read 3376 tok in 2.1 s | wrote 812 tok in 41.0 s)' from the model's metadata."""
    meta = message.response_metadata
    mode = "think" if message.additional_kwargs.get("reasoning_content") else "no-think"
    read_s = meta.get("prompt_eval_duration", 0) / 1e9
    write_s = meta.get("eval_duration", 0) / 1e9
    return (f"({mode} | read {meta.get('prompt_eval_count', '?')} tok in {read_s:.1f} s | "
            f"wrote {meta.get('eval_count', '?')} tok in {write_s:.1f} s)")


def tool_outcome(message) -> tuple[str, str | None, str]:
    """Return ('ok' | 'warning' | 'error', error kind or None, short text) for one tool message."""
    content = message.content if isinstance(message.content, str) else json.dumps(message.content)
    try:
        data = json.loads(content)
    except ValueError:
        data = None
    if isinstance(data, dict) and "error" in data:
        return "error", data.get("error_type", "other"), data["error"][:300]
    if data is None and content.lower().startswith("error"):   # arguments refused by the schema
        return "error", "argument", content[:300]
    if isinstance(data, dict) and "warning" in data:
        return "warning", None, str(data["warning"])[:300]
    return "ok", None, ""


def log_step(label: str, n: int, message) -> None:
    """One readable line per graph step; tool arguments and errors go only to the file (DEBUG)."""
    head = f"STEP  {label} #{n:<2}"
    if message.type == "ai" and message.tool_calls:
        names = ", ".join(call["name"] for call in message.tool_calls)
        log.info(f"{head} model asks {names} {timing(message)}")
        for call in message.tool_calls:
            log.debug(f"ARGS  {label} #{n:<2} {call['name']} {json.dumps(call['args'], ensure_ascii=False)}")
    elif message.type == "ai":
        log.info(f"{head} model answers {timing(message)}")
    elif message.type == "tool":
        status, kind, text = tool_outcome(message)
        if status == "error":
            log.warning(f"{head} tool {message.name}: ERROR ({kind})")
            log.debug(f"ERROR {label} #{n:<2} {text}")
        elif status == "warning":
            log.warning(f"{head} tool {message.name}: ok with warning")
            log.debug(f"WARN  {label} #{n:<2} {text}")
        else:
            log.info(f"{head} tool {message.name}: ok")


def run_one(graph, question: str, label: str, recursion_limit: int) -> tuple[list, str]:
    """Ask ONE question in a fresh conversation; return the messages and how it ended."""
    messages = [HumanMessage(content=question)]
    try:
        for state in graph.stream({"messages": messages}, {"recursion_limit": recursion_limit},
                                  stream_mode="values"):
            messages = state["messages"]
            if len(messages) > 1:
                log_step(label, len(messages), messages[-1])
        return messages, "ok"
    except GraphRecursionError:
        return messages, "recursion_limit"
    except Exception as exc:  # the model server failed (e.g. out of memory): keep the batch going
        log.error(f"MODEL {label} {type(exc).__name__}: {str(exc)[:300]}")
        return messages, "model_error"


def with_reasoning(message) -> str:
    """The message as printed by LangChain, preceded by the model's hidden reasoning if any."""
    reasoning = message.additional_kwargs.get("reasoning_content", "")
    text = message.pretty_repr()
    return f"[REASONING]\n{reasoning}\n[/REASONING]\n{text}" if reasoning else text


def final_answer(messages: list) -> str:
    """Return the text of the last model message if it is an answer (no tool call), else ''."""
    last = messages[-1] if messages else None
    if last is None or last.type != "ai" or last.tool_calls:
        return ""
    return last.content if isinstance(last.content, str) else str(last.content)


def node_of(message) -> str:
    """The graph node that produced a model message (written by the graph in response_metadata)."""
    return message.response_metadata.get("node", UNKNOWN_NODE)


def tool_record(messages: list) -> dict:
    """Tool use of one run: calls per tool, errors per kind, and every error with the call that caused it."""
    calls = {}
    for m in messages:
        if m.type == "ai":
            for call in m.tool_calls:
                calls[call["id"]] = call
    per_tool, kinds, errors, warnings = Counter(), Counter(), [], 0
    node_of_call = {call["id"]: node_of(m) for m in messages if m.type == "ai" for call in m.tool_calls}
    per_node = Counter()
    for m in messages:
        if m.type != "tool":
            continue
        per_tool[m.name] += 1
        per_node[node_of_call.get(m.tool_call_id, UNKNOWN_NODE)] += 1
        status, kind, text = tool_outcome(m)
        if status == "error":
            kinds[kind] += 1
            errors.append({"tool": m.name, "kind": kind, "args": calls.get(m.tool_call_id, {}).get("args"),
                           "error": text})
        elif status == "warning":
            warnings += 1
    return {"tool_calls": sum(per_tool.values()), "tool_calls_per_tool": dict(per_tool),
            "tool_calls_per_node": dict(per_node),
            "tool_errors": sum(kinds.values()), "tool_errors_per_kind": dict(kinds),
            "tool_warnings": warnings,
            # column mapping errors: a wrong column, JSON path or value of a column
            "mapping_errors": kinds.get("column", 0) + kinds.get("value", 0),
            "errors": errors}


def model_calls(messages: list) -> list[dict]:
    """Every model call of a run as {node, prompt_eval_count, eval_count, prompt_eval_duration, eval_duration}.

    The calls whose reply stays in the conversation, plus the discarded ones (repeated empty replies,
    the empty reply before a handoff), which the graph saved in "discarded_calls".
    """
    calls = []
    for m in messages:
        if m.type == "ai":
            meta = m.response_metadata
            calls.append({"node": node_of(m), **{k: meta.get(k, 0) or 0 for k in CALL_FIELDS}})
            calls += meta.get("discarded_calls", [])
        elif m.type == "human":
            calls += m.additional_kwargs.get("discarded_calls", [])
    return calls


def per_node_record(calls: list[dict]) -> dict:
    """Model calls, tokens read and written, and seconds spent reading and writing, for each node."""
    nodes = {}
    for c in calls:
        n = nodes.setdefault(c["node"], {"model_calls": 0, "tokens_read": 0, "tokens_written": 0,
                                         "seconds_reading": 0.0, "seconds_writing": 0.0})
        n["model_calls"] += 1
        n["tokens_read"] += c["prompt_eval_count"]
        n["tokens_written"] += c["eval_count"]
        n["seconds_reading"] += c["prompt_eval_duration"] / 1e9
        n["seconds_writing"] += c["eval_duration"] / 1e9
    for n in nodes.values():
        n["seconds_reading"], n["seconds_writing"] = round(n["seconds_reading"], 1), round(n["seconds_writing"], 1)
    return nodes


def model_record(messages: list) -> dict:
    """Model use of one run, discarded calls included: calls, tokens, reading and writing times, per node."""
    ai = [m for m in messages if m.type == "ai"]
    calls = model_calls(messages)
    return {
        "model_calls": len(calls),
        "prompt_tokens_last": ai[-1].response_metadata.get("prompt_eval_count") if ai else None,
        "tokens_written": sum(c["eval_count"] for c in calls),
        "seconds_reading": round(sum(c["prompt_eval_duration"] for c in calls) / 1e9, 1),
        "seconds_writing": round(sum(c["eval_duration"] for c in calls) / 1e9, 1),
        "empty_answer": not ai or not str(ai[-1].content).strip(),
        "empty_retries": sum(1 for c in calls if c.get("reason") == "empty_retry"),
        "handoffs": sum(1 for m in messages if m.type == "human" and str(m.content).startswith(HANDOFF_MARK)),
        "per_node": per_node_record(calls),
    }


def rewoo_record(messages: list) -> dict:
    """Measures of the ReWOO graph: the plan (rejections, retry, steps) and what happened to its steps.

    steps_failed counts the plan's steps whose result is an error, by kind; calculate_calls counts
    the analyst's calculations. Time and tokens per node are in model_record()["per_node"].
    """
    plan = next((m for m in messages if m.type == "ai" and node_of(m) == "plan"), None)
    meta = plan.response_metadata if plan else {}
    step_ids = {call["id"] for call in plan.tool_calls} if plan else set()
    failed = Counter()
    for m in messages:
        if m.type == "tool" and m.tool_call_id in step_ids:
            status, kind, _ = tool_outcome(m)
            if status == "error":
                failed[kind] += 1
    return {"plan_rejections": len(meta.get("plan_rejections", [])),
            "plan_rejection_reasons": meta.get("plan_rejections", []),
            "plan_retries": meta.get("plan_retries", 0),
            "why_in_args": meta.get("why_in_args", 0),
            "steps_planned": len(step_ids),
            "steps_failed": sum(failed.values()),
            "steps_failed_per_kind": dict(failed),
            "calculate_calls": sum(1 for m in messages if m.type == "tool" and m.name == "calculate"
                                   and m.tool_call_id not in step_ids)}


def repair_record(messages: list) -> dict:
    """Measures of the repair rounds of the rewoo_repair graph (zeros when nothing failed).

    steps_failed_before_repair: failed steps of the first plan. steps_repaired: steps written by the repair node.
    steps_failed_after_repair: failed steps among those repair steps (a first-plan step that was not
    replaced is not counted again).
    """
    plans = []
    for m in messages:
        if m.type == "ai" and node_of(m) in ("plan", "repair"):
            plans.append((m, []))
        elif m.type == "tool" and plans:
            plans[-1][1].append(m)
    failed = lambda results: sum(1 for r in results if tool_outcome(r)[0] == "error")
    repairs = [p for p in plans if node_of(p[0]) == "repair"]
    first = next((p for p in plans if node_of(p[0]) == "plan"), (None, []))
    return {"repair_rounds_done": len(repairs),
            "steps_repaired": sum(len(p[0].tool_calls) for p in repairs),
            "steps_failed_before_repair": failed(first[1]),
            "steps_failed_after_repair": sum(failed(p[1]) for p in repairs),
            "repair_rejections": sum(len(p[0].response_metadata.get("repair_rejections", [])) for p in repairs)}


def number_sources(messages: list, analyst) -> list[str]:
    """The texts the analyst could take numbers from: question, tool results, context, database reference."""
    texts = [m.content if isinstance(m.content, str) else json.dumps(m.content)
             for m in messages if m.type in ("human", "tool") and not str(m.content).startswith(HANDOFF_MARK)]
    return texts + [analyst.domain.context, analyst.reference]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line (argv: for tests; default sys.argv)."""
    parser = argparse.ArgumentParser(description="Run a domain's bench questions.")
    parser.add_argument("--domain", required=True, help="folder name under domains/")
    parser.add_argument("--reps", type=int, default=2, help="repetitions per question (seed = 1, 2, ...)")
    parser.add_argument("--tag", default="", help="optional word added to the output folder name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="replace a setting for this batch, e.g. --set graph=rewoo_repair --set "
                             "roles.planner.reasoning=false (repeatable; applied on top of settings.yaml and the "
                             "domain config; recorded in bench.log and runs.jsonl)")
    parser.add_argument("questions", nargs="*", help="question ids (default: all)")
    return parser.parse_args(argv)


def data_fingerprint(spec):
    """SHA-256 of the database file, or {table: SHA-256 of its file} for Parquet/CSV tables."""
    if spec.database is not None:
        return sha256_of(spec.database)
    return {name: sha256_of(path) for name, path in spec.tables.items()} or None


def versions_record(analyst, graph_id: str, overrides: dict) -> dict:
    """What identifies the system under test: prompt, method, roles, context, tools, graph, data, overrides."""
    settings = analyst.domain.settings
    return {
        "prompt": settings["prompt"],
        "method_hash": text_hash((PROMPTS_DIR / f"method_{settings['prompt']}.md").read_text(encoding="utf-8")),
        "roles_hash": text_hash((PROMPTS_DIR / f"roles_{settings['prompt']}.yaml").read_text(encoding="utf-8")),
        "context_hash": text_hash(analyst.domain.context),
        "tools": TOOLS_VERSION,
        "graph": graph_id,
        "data_sha256": data_fingerprint(analyst.domain.data),
        "overrides": overrides,
    }


def main() -> None:
    """Run every chosen question x repetition; save transcripts, runs.jsonl and bench.log."""
    args = parse_args()
    overrides = parse_override_items(args.set)
    analyst = open_analyst(args.domain, overrides=parse_overrides(args.set))
    settings = analyst.domain.settings
    bench = load_yaml(analyst.domain.bench_file)
    questions = {str(k): v for k, v in bench["questions"].items()}
    chosen = args.questions or list(questions)
    unknown = [q for q in chosen if q not in questions]
    if unknown:
        raise SystemExit(f"unknown question ids {unknown}; known: {list(questions)}")
    graph_id = graph_name(analyst)
    model = settings["roles"][ANSWER_ROLE[graph_id]]["model"].replace(":", "-")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{stamp}_{analyst.domain.name}_{model}_{settings['prompt']}_t{TOOLS_VERSION}_{graph_id}" + (f"_{args.tag}" if args.tag else "")
    run_dir = OUT_DIR / name
    run_dir.mkdir(parents=True)
    setup_logging(run_dir)
    versions = versions_record(analyst, graph_id, overrides)
    log.info(f"BATCH {name} | questions {chosen} x {args.reps} reps | versions {versions}")
    answered = []
    for qid in chosen:
        for rep in range(1, args.reps + 1):
            label = f"{qid} r{rep}"
            log.info(f"START {label} | {questions[qid][:70]}...")
            graph = build_graph(analyst, seed=rep)
            start = time.time()
            messages, ending = run_one(graph, questions[qid], label, settings["limits"]["recursion_limit"])
            seconds = round(time.time() - start, 1)
            (run_dir / f"{qid}_r{rep}.txt").write_text("\n".join(with_reasoning(m) for m in messages), encoding="utf-8")
            answer = final_answer(messages)
            numbers = check_numbers(answer, number_sources(messages, analyst)) if answer else {}
            record = {"question": qid, "rep": rep, "seed": rep, "domain": analyst.domain.name, "mode": bench.get("mode"),
                      **versions, "provider": settings["provider"], "roles": settings["roles"],
                      "answer_language": settings["answer_language"], "ending": ending, "seconds": seconds,
                      **model_record(messages), **tool_record(messages), **numbers,
                      **(rewoo_record(messages) if graph_id in ("rewoo", "rewoo_repair") else {}),
                      **(repair_record(messages) if graph_id == "rewoo_repair" else {})}
            with open(run_dir / "runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ok = ending == "ok" and not record["empty_answer"]
            line = (f"END   {label} | {ending} | {seconds:.0f} s | {record['model_calls']} model calls | "
                    f"{record['tool_calls']} tool calls {record['tool_calls_per_tool']} | "
                    f"{record['tool_errors']} errors {record['tool_errors_per_kind']} | "
                    f"{record['tool_warnings']} warnings | {record['empty_retries']} empty-answer retries | "
                    f"{record['handoffs']} handoffs"
                    + ("" if ok else " | NO ANSWER"))
            (log.info if ok and not record["tool_errors"] else log.warning)(line)
            if numbers:
                nums = (f"NUMS  {label} | {numbers['numbers_cited']} cited: {numbers['numbers_direct']} direct, "
                        f"{numbers['numbers_derived']} derived, {numbers['numbers_untraceable']} untraceable")
                if numbers["untraceable_list"]:
                    log.warning(nums + " -> " + ", ".join(numbers["untraceable_list"]))
                else:
                    log.info(nums)
            answered.append(ok)
    analyst.close()
    log.info(f"DONE  {sum(answered)}/{len(answered)} runs with an answer | files in {run_dir}")


if __name__ == "__main__":
    main()
