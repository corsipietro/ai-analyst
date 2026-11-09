# Episode 2: evidence

Everything behind the numbers in the episode 2 post: where the wrong answers come from.

| Path | Contents |
|---|---|
| `runs/` | 30 runs = 6 setups × 5 domains, g5.1 tools. Each folder has one transcript per answer (`<question>_r<seed>.txt`, with tool calls and results), `runs.jsonl` (versions, settings, timings, tokens per graph node, tool calls, errors, number-checker result) and `bench.log`. Folder names end with the setup tag (`g51_react_12b`, `g51_rewoo_12b`, `g51_repair_12b`, and the same with `_26b`). |
| `grading/` | the verdicts on the answers that the automatic checks could not settle: `g5_g51_read_key.csv` maps codes to runs, `g5_g51_read/sheet.csv` holds the verdicts. The sheet also covers answers of the episode 1 runs (folders ending in `_g5_...`, in `results/episode1/runs/`), graded with the same rules for the comparison. |
| `scripts/post_numbers.py` | every number of the post, printed with its source; writes `numbers.json` (what the carousel draws). Run from the repository root: `python3 -B results/episode2/scripts/post_numbers.py` |
| `numbers.txt`, `numbers.json` | its output, as used for the post |
| `reproduce.sh` | reruns all 30 runs with the exact settings |

## How an answer is counted

- **Right or wrong.** `bench/classify_errors.py` checks each answer against the expected numbers in `domains/*/bench.yaml`
  (written before the runs) and against the defects listed there. The answers it cannot settle were read one by one
  (`grading/`): rules fixed in advance, applied by an AI (Claude), read by the author. A wrong answer is one with no
  answer, or one built on a false or unchecked fact. Silent loss of data counts as a false fact.
- **Source of each wrong answer.** *data*: the answer uses a value made wrong by a flaw in the data (the trap value
  listed in `bench.yaml`) without noticing it. *tools*: the tools could not give what the question needed.
  *model*: the data and tools were enough, the reasoning was wrong. On question 4a, the four 12B answers that name the
  duplicated row and keep the inflated count were set to *model* by hand (`SOURCE_FIX` in `post_numbers.py`).
- **Flawed questions.** 3 of the 22 questions are on data with a known flaw, all on `gen_rentals`: 2a and 4a
  (a station listed twice: a join with it counts some rentals twice), 3a (rentals whose customer is missing).
  2a can be answered without the table that holds the duplicate.

## Results

| Model | Strategy | Wrong / 44 | data / tools / model | Free-SQL share of tool calls | Tokens per answer | Spent on figuring out the data |
|---|---|---|---|---|---|---|
| 12B | ReAct | 8 | 3 / 1 / 4 | 30% | 35k | 61% |
| 12B | ReWOO | 3 | 0 / 1 / 2 | 13% | 16k | 48% |
| 12B | ReWOO + repair | 4 | 0 / 2 / 2 | 17% | 20k | 52% |
| 26B | ReAct | 4 | 4 / 0 / 0 | 45% | 35k | 72% |
| 26B | ReWOO | 5 | 3 / 2 / 0 | 18% | 17k | 46% |
| 26B | ReWOO + repair | 3 | 3 / 0 / 0 | 18% | 17k | 52% |

Tokens are read + written. "Figuring out the data" = the explore, plan and repair nodes of the graph.

- **Flawed vs clean data.** Flawed questions: 17/36 wrong (47%). Clean questions: 10/228 (4%).
- **The duplicated row.** 4a: 12/12 wrong (every setup, both seeds). 2a, same data: 0/12. 3a: 5/12 (12B 1, 26B 4).
- **Free SQL.** All 12 wrong answers on 4a used `query_results` (free SQL), 11 of them with a join written by the
  model; all 5 wrong answers on 3a used free SQL. This is a count, not an explanation.
- **Time.** 91-97% of the time is the model writing; 3.8-4.9 minutes per answer.
- **Stuck.** Only 12B ReAct gave empty answers (3).
- One answer was left in doubt; it is not counted as wrong.

## Compared with episode 1

On the 21 questions shared with episode 1 (g5 tools): 13 → 9 wrong answers (12B), 7 → 6 (26B). With these
numbers, that is noise. The two benches are never added together.

## Limits

- Synthetic data with known flaws, so that the answers can be checked. 3 flawed questions on one dataset: 47% is not a
  general rate.
- 2 seeds per setup, not independent; ±1 per cell does not change the picture.
- The defects, trap values and SQL signals are written by the author on these data: the source counts are indicative.
- The verdicts are provisional: graded with an AI, read by the author, not by a second person.
- Same model family and same instructions as in episode 1 (ReAct `prompt=v3`, ReWOO `prompt=v4`).
