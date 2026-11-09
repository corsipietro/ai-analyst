"""Tests for checker.py (ported from analyst/test_number_check.py), on numbers taken from the 29/10 bench runs (2a and 3a)."""
from sim_analyst.checker import check_numbers, extract_numbers, parse_number, tool_marks_without_data

TOOL = ('{"rows": [[40, 60, 3, 569.884, 419.517, 722.455], [60, 60, 3, 1211.517, 616.612, 1993.459], '
        '[70, 60, 3, 975.646, 834.367, 1157.467], [120, 60, 3, 2827.189, 1863.414, 3461.224]]}')


def test_parse_number_handles_separators():
    """Thousands commas, decimal commas and decimals are read correctly."""
    assert parse_number("1,376.85") == (1376.85, 2)
    assert parse_number("569,9") == (569.9, 1)
    assert parse_number("570") == (570.0, 0)


def test_list_markers_and_identifiers_are_skipped():
    """'1.' at the start of a line and digits glued to letters are not numbers."""
    found = [w for w, _, _ in extract_numbers("1. gemma4 says S1 is 569.88.\n2) done")]
    assert found == ["569.88"]


def test_direct_numbers_match_with_rounding():
    """Rounded tool values are traced directly."""
    result = check_numbers("Means: 569.88, 1211.5, about 570 and 2827.2 debris.", [TOOL])
    assert result["numbers_direct"] == 4 and result["numbers_untraceable"] == 0


def test_sums_and_differences_are_derived():
    """Difference of two means and a min-max spread are traced as derived."""
    result = check_numbers("Difference 2257.31; spread 1376.85; gap 235.87.", [TOOL])
    assert result["numbers_derived"] == 3 and result["numbers_untraceable"] == 0


def test_invented_numbers_are_untraceable():
    """A mean that is not in the data (the 29/10 error 878.1) and a ratio are flagged."""
    result = check_numbers("The mean for 60 launches is 878.1, a ratio of 4.96.", [TOOL])
    assert result["untraceable_list"] == ["878.1", "4.96"]


def test_tool_marks_without_data_counts_the_marks_only_when_nothing_was_read():
    """[tool] in an answer written after failed reading steps is a measure of a claim without data."""
    answer = "S = 400 [tool]. 3 [computed]"
    assert tool_marks_without_data(answer, False) == 1
    assert tool_marks_without_data(answer, True) == 0
    assert tool_marks_without_data("a [tool] b [tool]", False) == 2
    assert tool_marks_without_data("", False) == 0
