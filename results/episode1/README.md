# Episode 1: evidence

Everything behind the numbers in the episode 1 post.

| Path | Contents |
|---|---|
| `runs/` | 30 runs = 6 setups × 5 domains. Each folder has one transcript per answer (`<question>_r<seed>.txt`, with tool calls and results), `runs.jsonl` (versions, settings, timings, tool calls, errors, number-checker result) and `bench.log`. Folder names end with the setup tag (`g5_react`, `g5_rewoo`, `g5_repair`, and the same with `_26b`). |
| `summary.csv` | one row per setup, computed by `scripts/post_numbers.py` |
| `hard_points_by_problem.txt` | the keyword count for each hidden problem, by `scripts/hard_points.py` (problem labels in Italian) |
| `grading/` | blind grading of the 12B setups: anonymised answers (`G001.txt`...), the key that maps codes to runs, and the grading sheets. Grades were given by an AI against the expected answers and reviewed by the author; notes partly in Italian. |
| `scripts/` | counting scripts. Run them from `runs/`. |
| `reproduce.sh` | reruns all 30 runs with the exact settings |

Column meanings in `summary.csv`:
- `answers_nonempty`: runs that ended normally with a non-empty answer.
- `hard_points`: keyword checks passed (a raw count, spot-checked by hand).
- `raw_sql_reads` / `all_data_reads`: `query_results` calls (free SQL) out of all data reads (free SQL plus the checked tools).
- `steps_failed`: failed plan steps, counted only for the ReWOO graphs. `tool_errors`: errors returned by the tools.
- `numbers_untraceable`: numbers in the answer that the number checker could not trace back to tool results or context.
