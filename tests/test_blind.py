"""Tests of the blind grading script, on two fake run folders."""
import csv

import pytest

from bench import blind

TRANSCRIPT = """================================ Human Message =================================

question
[REASONING]
I thought of 999.
[/REASONING]
================================== Ai Message ==================================

The answer is {text}.
"""
AUTO_CHECKS = {"1": [21, 570], "2": [570]}


def make_runs(tmp_path):
    """Two run folders with three transcripts: 1a (family 1) in both, 2a (family 2) in one."""
    folders = []
    for name, text in (("run_alpha", "21 and 570"), ("run_beta", "20.7 and 600")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "1a_r1.txt").write_text(TRANSCRIPT.format(text=text), encoding="utf-8")
        folders.append(folder)
    (folders[0] / "2a_r1.txt").write_text(TRANSCRIPT.format(text="570"), encoding="utf-8")
    return folders


def read(path):
    """Rows of a CSV file as dicts."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_final_answer_skips_the_reasoning():
    """Only the text after the last 'Ai Message' header counts, without the reasoning block."""
    assert blind.final_answer_of(TRANSCRIPT.format(text="42")).strip() == "The answer is 42."


def test_make_hides_the_origin_and_keeps_a_key(tmp_path):
    """Unique codes; no folder name in the sheet; the key maps each code back to its source."""
    folders = make_runs(tmp_path)
    blind.make(folders, "t1", AUTO_CHECKS, tmp_path / "grading", seed=3)
    sheet = read(tmp_path / "grading" / "t1" / "sheet.csv")
    key = read(tmp_path / "grading" / "t1_key.csv")
    assert len(sheet) == 3 and len({r["id"] for r in sheet}) == 3
    text = (tmp_path / "grading" / "t1" / "sheet.csv").read_text()
    assert "run_alpha" not in text and "run_beta" not in text
    assert sorted(p.name for p in (tmp_path / "grading" / "t1").glob("G*.txt")) == sorted(r["id"] + ".txt" for r in sheet)
    origin = {r["code"]: (r["run"], r["file"]) for r in key}
    assert sorted(origin.values()) == [("run_alpha", "1a_r1.txt"), ("run_alpha", "2a_r1.txt"),
                                       ("run_beta", "1a_r1.txt")]
    row = next(r for r in sheet if origin[r["id"]] == ("run_alpha", "1a_r1.txt"))
    assert row["question"] == "1a" and row["family"] == "1"
    assert (row["has_21"], row["has_570"]) == ("yes", "yes")
    beta = next(r for r in sheet if origin[r["id"]] == ("run_beta", "1a_r1.txt"))
    assert (beta["has_21"], beta["has_570"]) == ("yes", "no")     # 20.7 is within 2% of 21
    assert all(c in row for c in ["A1", "A6", "B1", "B5", "flags", "notes"])


def test_same_seed_same_codes(tmp_path):
    """The shuffle is reproducible."""
    folders = make_runs(tmp_path)
    blind.make(folders, "a", AUTO_CHECKS, tmp_path / "g", seed=5)
    blind.make(folders, "b", AUTO_CHECKS, tmp_path / "g", seed=5)
    assert read(tmp_path / "g" / "a_key.csv") == read(tmp_path / "g" / "b_key.csv")


def test_unblind_sums_points_per_run_family_and_criterion(tmp_path):
    """Points are added up, graded and NA answers counted, flags counted; blank cells are not graded."""
    folders = make_runs(tmp_path)
    blind.make(folders, "t1", AUTO_CHECKS, tmp_path / "grading", seed=3)
    key = {(r["run"], r["file"]): r["code"] for r in read(tmp_path / "grading" / "t1_key.csv")}
    sheet_path = tmp_path / "grading" / "t1" / "sheet.csv"
    rows = read(sheet_path)
    marks = {key[("run_alpha", "1a_r1.txt")]: ("2", "invented number"),
             key[("run_alpha", "2a_r1.txt")]: ("1", ""),
             key[("run_beta", "1a_r1.txt")]: ("NA", "invented number;no answer")}
    for row in rows:
        row["A1"], row["flags"] = marks[row["id"]]
    with open(sheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = blind.unblind("t1", tmp_path / "grading")
    result = read(out)
    cell = lambda run, family, item: next(r for r in result if (r["run"], r["family"], r["item"]) == (run, family, item))
    assert cell("run_alpha", "1", "A1")["points"] == "2" and cell("run_alpha", "1", "A1")["graded"] == "1"
    assert cell("run_alpha", "2", "A1")["points"] == "1"
    assert cell("run_beta", "1", "A1")["na"] == "1" and cell("run_beta", "1", "A1")["points"] == "0"
    assert cell("run_alpha", "1", "A2")["graded"] == "0"
    assert cell("run_beta", "1", "flag:invented number")["flag_count"] == "1"
    assert cell("run_beta", "1", "flag:no answer")["flag_count"] == "1"


TOOL_CALL_END = """================================== Ai Message ==================================

Let me look.
Tool Calls:
  summarize_results (call_1)
 Call ID: call_1
  Args:
    value: value
"""
EMPTY_END = """================================== Ai Message ==================================

"""
TOOL_MESSAGE_END = TRANSCRIPT.format(text="570") + """================================= Tool Message =================================
Name: summarize_results

{}
"""


def test_answer_column_says_whether_the_last_message_is_an_answer(tmp_path):
    """yes for a text answer; no for a tool call, an empty message, or a run that ends on a tool result."""
    folder = tmp_path / "run"
    folder.mkdir()
    cases = {"1a": TRANSCRIPT.format(text="21"), "1b": TRANSCRIPT.format(text="21").split("question")[0] + TOOL_CALL_END,
             "1c": TRANSCRIPT.format(text="21").split("question")[0] + EMPTY_END, "2a": TOOL_MESSAGE_END}
    for question, text in cases.items():
        (folder / f"{question}_r1.txt").write_text(text, encoding="utf-8")
    blind.make([folder], "t1", AUTO_CHECKS, tmp_path / "g", seed=1)
    key = {r["code"]: r["file"] for r in read(tmp_path / "g" / "t1_key.csv")}
    answers = {key[r["id"]]: r["answer"] for r in read(tmp_path / "g" / "t1" / "sheet.csv")}
    assert answers == {"1a_r1.txt": "yes", "1b_r1.txt": "no", "1c_r1.txt": "no", "2a_r1.txt": "no"}


def write_sheet(tmp_path, cell):
    """Make a sheet for one transcript with `cell` typed in A1; return the grading folder."""
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "1a_r1.txt").write_text(TRANSCRIPT.format(text="21"), encoding="utf-8")
    blind.make([folder], "t1", AUTO_CHECKS, tmp_path / "g", seed=1)
    rows = read(tmp_path / "g" / "t1" / "sheet.csv")
    rows[0]["A1"] = cell
    blind.write_csv(tmp_path / "g" / "t1" / "sheet.csv", list(rows[0]), rows)
    return tmp_path / "g"


@pytest.mark.parametrize("cell", ["0", "1", "2", "NA", ""])
def test_unblind_accepts_valid_cells(tmp_path, cell):
    """0, 1, 2, NA and an empty cell are fine."""
    blind.unblind("t1", write_sheet(tmp_path, cell))


@pytest.mark.parametrize("cell", ["3", "-1", "1.5", "x", "yes"])
def test_unblind_rejects_other_values_naming_code_criterion_and_value(tmp_path, cell):
    """Anything else is an error that says which answer, which criterion and what was found."""
    with pytest.raises(ValueError) as error:
        blind.unblind("t1", write_sheet(tmp_path, cell))
    message = str(error.value)
    assert "G001" in message and "A1" in message and repr(cell) in message
