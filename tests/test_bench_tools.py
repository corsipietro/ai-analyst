"""Tests of the bench helpers: run settings overrides (--set), data fingerprints, and the replay script."""
from __future__ import annotations

import json
import types

import pytest

from bench import replay_tool_calls
from bench.run_bench import data_fingerprint, parse_args, versions_record
from sim_analyst.config import DataSpec, load_domain, parse_overrides, parse_override_items, sha256_of


# ---------------------------------------------------------------- --set

def test_override_items_are_parsed_with_yaml_types():
    flat = parse_override_items(["graph=rewoo", "roles.planner.reasoning=false", "limits.repair_rounds=2",
                                 "roles.planner.temperature=0.5", "prompt=v4"])
    assert flat == {"graph": "rewoo", "roles.planner.reasoning": False, "limits.repair_rounds": 2,
                    "roles.planner.temperature": 0.5, "prompt": "v4"}
    assert parse_overrides(["roles.planner.reasoning=false", "roles.planner.top_k=10"]) == {
        "roles": {"planner": {"reasoning": False, "top_k": 10}}}
    with pytest.raises(ValueError):
        parse_override_items(["no_equal_sign"])


def test_overrides_are_applied_on_top_of_settings_and_domain_config():
    plain = load_domain("toy_shell")
    changed = load_domain("toy_shell", overrides=parse_overrides(["graph=rewoo_repair", "roles.planner.reasoning=false"]))
    assert changed.settings["graph"] == "rewoo_repair" and changed.settings["roles"]["planner"]["reasoning"] is False
    assert changed.settings["roles"]["planner"]["model"] == plain.settings["roles"]["planner"]["model"]   # the rest stays
    assert plain.settings["graph"] != "rewoo_repair" or plain.settings["roles"]["planner"]["reasoning"] is not False


def test_set_is_repeatable_on_the_command_line():
    args = parse_args(["--domain", "x", "--set", "graph=rewoo", "--set", "prompt=v4", "1a"])
    assert args.set == ["graph=rewoo", "prompt=v4"] and args.questions == ["1a"]
    assert parse_args(["--domain", "x"]).set == []


def test_overrides_and_data_fingerprints_are_recorded_in_the_versions(tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("x\n1\n")
    domain = types.SimpleNamespace(data=DataSpec(tables={"a": csv}), context="c", name="d",
                                   settings={"prompt": "v4"})
    analyst = types.SimpleNamespace(domain=domain)
    versions = versions_record(analyst, "rewoo", {"graph": "rewoo"})
    assert versions["overrides"] == {"graph": "rewoo"} and versions["graph"] == "rewoo"
    assert versions["data_sha256"] == {"a": sha256_of(csv)}
    assert data_fingerprint(DataSpec(database=csv)) == sha256_of(csv)


# ---------------------------------------------------------------- replay

TRANSCRIPT = '''================================ Human Message =================================

question
================================== Ai Message ==================================

1. summarize_results: a
Tool Calls:
  summarize_results (step_1)
 Call ID: step_1
  Args:
    value: value
    group_by: config.strategies[0].amount
    time: step
    at: [10]
    filters: {'quantity': 'x', 'experiment': 'exp_a'}
  query_results (step_2)
 Call ID: step_2
  Args:
    sql: SELECT COUNT(*) AS n
FROM runs
  summarize_results (step_3)
 Call ID: step_3
  Args:
    value: nothing_here
  calculate (step_4)
 Call ID: step_4
  Args:
    expression: 1+1
================================= Tool Message =================================
Name: summarize_results

{"error": "NO RESULT: old refusal", "error_type": "column"}
================================= Tool Message =================================
Name: query_results

{"columns": ["n"], "rows": [[8]], "truncated": false}
================================= Tool Message =================================
Name: summarize_results

{"error": "unknown column 'nothing_here'", "error_type": "column"}
================================= Tool Message =================================
Name: calculate

{"expression": "1+1", "result": 2}
================================== Ai Message ==================================

answer
'''


@pytest.fixture()
def bench_folder(tmp_path):
    folder = tmp_path / "runs" / "20261031_0000_fake_x"
    folder.mkdir(parents=True)
    (folder / "runs.jsonl").write_text(json.dumps({"domain": "fake", "question": "1a", "rep": 1}) + "\n")
    (folder / "1a_r1.txt").write_text(TRANSCRIPT)
    return folder


def test_calls_and_old_outcomes_are_read_from_a_transcript():
    calls = replay_tool_calls.read_transcript(TRANSCRIPT)
    assert [c["tool"] for c in calls] == ["summarize_results", "query_results", "summarize_results"]   # no calculate
    assert calls[0]["args"] == {"value": "value", "group_by": "config.strategies[0].amount", "time": "step",
                                "at": [10], "filters": {"quantity": "x", "experiment": "exp_a"}}
    assert calls[1]["args"]["sql"] == "SELECT COUNT(*) AS n\nFROM runs"
    assert [c["old"] for c in calls] == ["error (column)", "ok", "error (column)"]


def test_replay_reruns_the_calls_and_counts_old_to_new(bench_folder, toolbox):
    report = replay_tool_calls.replay([bench_folder], open_toolbox=lambda domain: toolbox)
    rows = report["rows"]
    assert [(r["question"], r["rep"], r["tool"], r["old"], r["new"]) for r in rows] == [
        ("1a", 1, "summarize_results", "error (column)", "ok"),
        ("1a", 1, "query_results", "ok", "ok"),
        ("1a", 1, "summarize_results", "error (column)", "error (column)")]
    assert report["counts"] == {"error (column)": {"error (column)": 1, "ok": 1}, "ok": {"ok": 1}}
    assert report["different"] == []                     # the ok call gives the same numbers


def test_replay_flags_a_call_whose_numbers_changed(bench_folder, toolbox):
    text = (bench_folder / "1a_r1.txt").read_text().replace('"rows": [[8]]', '"rows": [[9]]')
    (bench_folder / "1a_r1.txt").write_text(text)
    report = replay_tool_calls.replay([bench_folder], open_toolbox=lambda domain: toolbox)
    assert [d["tool"] for d in report["different"]] == ["query_results"]


def test_replay_prints_a_compact_table(bench_folder, toolbox):
    text = replay_tool_calls.format_report(replay_tool_calls.replay([bench_folder], open_toolbox=lambda d: toolbox))
    assert "1a" in text and "summarize_results" in text and "error (column) -> ok" in text


def test_text_after_the_last_tool_message_is_not_part_of_its_result():
    """A [REASONING] block (and the answer) follows the last result: the result must stay a clean error."""
    text = TRANSCRIPT.replace('{"error": "unknown column \'nothing_here\'", "error_type": "column"}\n'
                              '================================= Tool Message =================================\n'
                              'Name: calculate\n\n{"expression": "1+1", "result": 2}\n',
                              '{"error": "unknown column \'nothing_here\'", "error_type": "column"}\n'
                              '[REASONING]\nthinking about the numbers\n[/REASONING]\n')
    calls = replay_tool_calls.read_transcript(text)
    assert calls[-1]["old"] == "error (column)" and calls[-1]["content"].startswith("{") and "thinking" not in calls[-1]["content"]


def test_results_are_paired_with_the_call_of_the_same_tool_even_if_the_order_differs():
    """Transcripts print no call id, so a result is matched to the first unused call of its tool."""
    text = TRANSCRIPT.replace("Name: summarize_results\n\n{\"error\": \"NO RESULT: old refusal\", \"error_type\": \"column\"}",
                              "Name: PLACEHOLDER").replace("Name: query_results\n\n{\"columns\": [\"n\"], \"rows\": [[8]], \"truncated\": false}",
                              "Name: summarize_results\n\n{\"error\": \"NO RESULT: old refusal\", \"error_type\": \"column\"}")
    text = text.replace("Name: PLACEHOLDER", "Name: query_results\n\n{\"columns\": [\"n\"], \"rows\": [[8]], \"truncated\": false}")
    calls = replay_tool_calls.read_transcript(text)
    assert [(c["tool"], c["old"]) for c in calls] == [("summarize_results", "error (column)"), ("query_results", "ok"),
                                                      ("summarize_results", "error (column)")]
