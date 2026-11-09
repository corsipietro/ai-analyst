"""Tests of the error classification script (bench/classify_errors.py), on one fake run folder."""
import csv
import json

from bench import classify_errors as ce

HUMAN = "================================ Human Message =================================\n\nquestion\n"
AI = "================================== Ai Message ==================================\n\n"
TOOL = "================================= Tool Message =================================\nName: query\n\n"


def transcript(sql, tool_output, answer):
    """A transcript with one query call (sql), its output and a final answer; empty answer = no final message."""
    text = HUMAN
    if sql is not None:
        text += AI + f"Tool Calls:\n  query (call_1)\n Call ID: call_1\n  Args:\n    sql: {sql}\n"
        text += TOOL + tool_output + "\n"
    if answer:
        text += AI + answer + "\n"
    return text


# question, rep, ending, sql, tool output, final answer
ANSWERS = [
    ("1a", 1, "ok", "SELECT COUNT(DISTINCT id) FROM clienti WHERE age IS NOT NULL", '{"rows": [[397]]}', "There are 397 customers."),
    ("1a", 2, "ok", "SELECT count(*) FROM clienti", '{"rows": [[420]]}', "There are about 400 customers."),
    ("1a", 3, "ok", "SELECT count(*) FROM clienti", '{"rows": [[420]]}', "There are 400 customers."),
    ("2a", 1, "ok", "SELECT count(*) FROM noleggi JOIN stazioni USING (station_id)", '{"rows": [[5202]]}', "There are 5202 rentals."),
    ("2a", 2, "ok", "SELECT DISTINCT station_id FROM noleggi", '{"rows": [[5200]]}', "There are 5200 rentals."),
    ("2a", 3, "data_stop", None, "", ""),
    ("2a", 4, "ok", "SELECT count(*) FROM noleggi JOIN stazioni USING (station_id)", '{"rows": [[5202]]}',
     "There are 5202 rentals; ST01 appears twice, so 2 rentals are counted twice."),
]
BENCH = """auto_checks:
  "1": [420]
  "2": [5200]
defects:
  2a: {defect: duplicate_lookup_rows, trap_value: 5202, cleanable: true, declare_value: 2}
"""
REPORT = """Labels written by us on our bench data (defects, trap values, D3 signals): indicative, not a general measure of data quality.
Answers: 7; wrong 4, to read 2, doubt 0; verdicts: 2 graded, 5 automatic.
D1 per question:
  dom 1a [no defect]: 3 answers, 0 wrong (data 0, tools 0, model 0, grey 0), 2 to read
  dom 2a [duplicate_lookup_rows]: 4 answers, 4 wrong (data 2, tools 0, model 1, grey 1), 0 to read
D1 wrong by source: data 2, tools 0, model 1, grey 1
D2 data and cleanable: 2/7 answers (29%); duplicate_lookup_rows 2
Declared but correctable (wrong, source model: it saw the defect and did not correct it): 1
  run_x/2a_r4.txt
D3 unrequested interventions: 1 answers, 1 on questions without a defect (read each: declared or silent?)
  run_x/1a_r1.txt [no defect] IS NOT NULL | SELECT COUNT(DISTINCT id) FROM clienti WHERE age IS NOT NULL
To read by hand (numbers missing, or wrong with source data or grey): 5
  run_x/1a_r1.txt missing 420 (if wrong: tools)
  run_x/1a_r3.txt missing 420 (if wrong: model)
  run_x/2a_r1.txt wrong, source data
  run_x/2a_r2.txt wrong, source grey
  run_x/2a_r3.txt wrong, source data
Disagreements, graded vs automatic: 1
  run_x/2a_r2.txt graded wrong, automatic right
"""


def make_all(tmp_path):
    """Fake domains dir (domain 'dom'), run folder 'run_x' and grading dir with two graded answers."""
    domains = tmp_path / "domains"
    (domains / "dom").mkdir(parents=True)
    (domains / "dom" / "bench.yaml").write_text(BENCH, encoding="utf-8")
    run = tmp_path / "runs" / "run_x"
    run.mkdir(parents=True)
    with open(run / "runs.jsonl", "w", encoding="utf-8") as f:
        for q, rep, ending, sql, out, answer in ANSWERS:
            f.write(json.dumps({"question": q, "rep": rep, "domain": "dom", "ending": ending}) + "\n")
            (run / f"{q}_r{rep}.txt").write_text(transcript(sql, out, answer), encoding="utf-8")
    grading = tmp_path / "grading"
    (grading / "g").mkdir(parents=True)
    with open(grading / "g_key.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows([["code", "run", "file"], ["G001", "run_x", "1a_r2.txt"], ["G002", "run_x", "2a_r2.txt"]])
    with open(grading / "g" / "sheet.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows([["id", "question", "flags"], ["G001", "1a", ""], ["G002", "2a", "unit1:unchecked_or_false_fact"]])
    return run, domains, grading


def test_sql_signals_skip_count_distinct():
    """IS NOT NULL is a signal; COUNT(DISTINCT ...) counts and SELECT DISTINCT lists values: not signals."""
    text = transcript("SELECT COUNT(DISTINCT id) FROM t WHERE a IS NOT NULL", "{}", "x")
    assert ce.sql_signals(text) == [("IS NOT NULL", "SELECT COUNT(DISTINCT id) FROM t WHERE a IS NOT NULL")]
    assert ce.sql_signals(transcript("select distinct a from t", "{}", "x")) == []
    assert ce.sql_signals(transcript("CREATE TABLE u AS SELECT * FROM t", "{}", "x"))[0][0] == "CREATE"


def test_checks_are_not_signals():
    """Counting missing values (IS NULL) and looking for orphans (DISTINCT ... EXCEPT) check the data: no signal."""
    assert ce.sql_signals(transcript("SELECT count(*) FROM t WHERE a IS NULL", "{}", "x")) == []
    assert ce.sql_signals(transcript("SELECT DISTINCT a FROM t EXCEPT SELECT b FROM u", "{}", "x")) == []


def test_trap_close_to_target_is_compared_exactly():
    """5202 is within 2% of 5200: the answer 5202 misses the target and contains the trap."""
    assert ce.auto_verdict("There are 5202 rentals.", "ok", [5200], 5202) == ("wrong", [5200], True, False)
    assert ce.auto_verdict("There are 5200 rentals.", "ok", [5200], 5202) == ("right", [], False, False)
    assert ce.auto_verdict("", "data_stop", [5200], None)[0] == "wrong"
    assert ce.auto_verdict("", "ok", [5200], None)[0] == "wrong"


def test_missing_number_alone_is_to_read():
    """A missing target without the trap value is not wrong: the answer may say it in words. It is 'read'."""
    assert ce.auto_verdict("Web 0.41, android 0.31, ios 0.21.", "ok", [0.31, 0.21, 0.41, 0.29], None) == (
        "read", [0.29], False, False)


def test_declared_correctable_trap_stays_wrong():
    """cleanable true: the model could correct; declaring the trap keeps it wrong, marked declared, source model."""
    answer = "Ostiense 999; the counts add up to 5202, not 5200: ST01 is listed twice."
    assert ce.auto_verdict(answer, "ok", [997], 999, [5202], cleanable=True) == ("wrong", [997], True, True)
    assert ce.source_of("ok", answer, [997], True, True, "", 999) == "model"
    assert ce.source_of("ok", "Ostiense 999.", [997], True, False, "", 999) == "data"
    assert ce.source_of("ok", "", [997], False, False, "", 999) == "model"


def test_declared_exclusion_is_not_an_error():
    """28133.25 (total without the 25 orphans) is wrong if silent, right if the answer also writes 25."""
    targets = [15981, 28274.25]
    silent = "Base earns 15981 EUR, 56.8% of 28133.25 EUR."
    declared = silent + " This leaves out 25 rentals without a customer."
    assert ce.auto_verdict(silent, "ok", targets, 28133.25, 25) == ("wrong", [28274.25], True, False)
    assert ce.auto_verdict(declared, "ok", targets, 28133.25, 25) == ("right", [], False, True)
    assert ce.auto_verdict(declared, "ok", [15981, 12], 28133.25, 25)[1] == [12]
    amount = silent + " The 141.00 EUR of rentals without a customer are excluded."
    assert ce.auto_verdict(amount, "ok", targets, 28133.25, [25, 141]) == ("right", [], False, True)
    assert ce.auto_verdict(silent, "ok", targets, 28133.25, [25, 141])[0] == "wrong"


def test_classify_rows(tmp_path):
    """Verdict, its origin and the source of every answer of the fake run."""
    run, domains, grading = make_all(tmp_path)
    rows = ce.classify_runs([run], domains_dir=domains, grading_dir=grading)
    got = [(r["file"], r["verdict"], r["verdict_by"], r["source"]) for r in rows]
    assert got == [
        ("1a_r1.txt", "read", "automatic", "-"),
        ("1a_r2.txt", "right", "graded", "-"),
        ("1a_r3.txt", "read", "automatic", "-"),
        ("2a_r1.txt", "wrong", "automatic", "data"),
        ("2a_r2.txt", "wrong", "graded", "grey"),
        ("2a_r3.txt", "wrong", "automatic", "data"),
        ("2a_r4.txt", "wrong", "automatic", "model"),
    ]


def test_report_in_full(tmp_path):
    """The whole report of the fake run, line by line."""
    run, domains, grading = make_all(tmp_path)
    rows = ce.classify_runs([run], domains_dir=domains, grading_dir=grading)
    assert ce.report(rows) == REPORT
