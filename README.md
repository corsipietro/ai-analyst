# ai-analyst

[![tests](https://github.com/corsipietro/ai-analyst/actions/workflows/tests.yml/badge.svg)](https://github.com/corsipietro/ai-analyst/actions/workflows/tests.yml)

A small AI data analyst that runs on a laptop: open models through [Ollama](https://ollama.com), a [LangGraph](https://github.com/langchain-ai/langgraph) graph, and data tools written in Python that check what the model does.

You give it a results database (DuckDB, or CSV/Parquet files) and a few lines of context, then you ask questions in plain language. The model chooses what to compute; the tools compute it and report what they find: empty values, units, how the tables link and whether a difference is larger than the noise.

**Episode 1 in one line:** every setup caught about half of the hidden problems in the data; no setup wins.
**Episode 2 in one line:** 17 of the 27 wrong answers came from the 3 questions on data with a hidden flaw (47% wrong there, 4% elsewhere).

This repository holds the code and the full evidence of the first two episodes of a LinkedIn series on how AI agents fail, on a system I built and can open up: 6 setups (2 models × 3 strategies), 5 datasets; 252 answers in episode 1, 264 in episode 2.

The code on `main` is the one of episode 2 (tools g5.1). Episode 1 used the g5 tools: `git checkout episode1` gives that exact code.

## How it works

The analyst is built from three independent layers:

1. **A generic method** (`sim_analyst/prompts/method_v3.md` for ReAct, `method_v4.md` for ReWOO): good data-science practice, the same for every dataset. Explore the data before analysing it, never average different cases, compare differences with the spread between repetitions, compare the results with what the equations in the context imply, and state the limits.
2. **Generic tools** (`sim_analyst/tools.py`): computations a language model does badly or slowly, done in code on any results database. The model chooses the columns.
3. **A system context** (`domains/<name>/context.md`): a few lines on what the system is, its variables and units, its equations, its time scale and what was varied. No results and no hints about results.

A domain's `config.yaml` only says **where** the data are.

### Tools

| Tool | What it does |
|---|---|
| `describe_database` | tables, columns, types, row counts, distinct values, numeric ranges, paths inside JSON columns, shared columns, sample rows (given to the model once, in its instructions) |
| `query_results` | one read-only `SELECT`, limited rows, warnings on cut, empty or all-NULL results |
| `summarize_results` | n, mean, std, min, max of a column per group at given times |
| `describe_trends` | per group: initial and final value, maximum and minimum with their times, slope of the last part, shape and turning points, sampled series with the spread between replicates |
| `compare_groups` | difference of means between two groups, Welch's t interval (it decides the verdict *established / not established*), whether the ranges overlap; a percentile bootstrap interval (fixed seed) is added, never deciding, only with at least 30 values per group |
| `check_quality` | a compact data-quality report on one table: NULLs, runs of NULLs, duplicates, text variants, orphan keys |
| `calculate` | exact arithmetic on an expression sent as text (only numbers, `+ - * / **` and parentheses; parsed, never executed) |

Columns are arguments: `year`, `series.year`, or a value inside a JSON column written `config.strategies[0].launches`. Tables are joined automatically through their shared columns. If one group mixes rows that differ in an unused text column (for example two different quantities in the same value column), the tool refuses with an error that names the column and its values, and returns no numbers.

### Three strategies

| Strategy | Graph | In short |
|---|---|---|
| ReAct | `graph=v2`, `prompt=v3` | think, call a tool, read the result, repeat (an explorer collects the data, an interpreter writes the answer and may ask for more data) |
| ReWOO | `graph=rewoo`, `prompt=v4` | plan all the tool calls first, run them in code, then write the answer |
| ReWOO + repair | `graph=rewoo_repair`, `prompt=v4` | as ReWOO, plus one round that fixes the steps that failed |

Models, sampling and token caps for each role are set in `settings.yaml`. Any setting can be changed for one bench run with `--set key=value` (`python -m bench.run_bench ...`) and is recorded in `runs.jsonl`; `python -m sim_analyst ask` uses `settings.yaml` as it is.

## Quick start

Requirements: Python ≥ 3.12 (tested with 3.14) and [Ollama](https://ollama.com) with the models you want to use.

```bash
ollama pull gemma4:12b                 # and/or gemma4:26b
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"               # versions pinned in pyproject.toml
pytest                                 # tools, checker, configuration; no model needed

python -m sim_analyst describe --domain toy_shell
python -m sim_analyst ask --domain toy_shell "How does the debris at year 60 depend on the number of launches?"
python -m bench.run_bench --domain toy_shell --reps 2      # the domain's test bench -> bench/runs/
```

With the default `settings.yaml` the analyst runs a ReAct-style graph with a non-reasoning explorer: this is not one of the six episode 1 setups. For those, use the `--set` options in `results/episode2/reproduce.sh` (or `results/episode1/reproduce.sh` on the `episode1` tag).

**A new dataset:** copy `domains/toy_shell/`, edit `context.md` and the `data` section of `config.yaml` (a DuckDB file, or one Parquet/CSV file per table), and optionally write a `bench.yaml` with questions and expected answers.

## Episode 1: results

**Setup.** Two models: Gemma 4 12B (dense) and Gemma 4 26B A4B (mixture of experts, about 4B parameters active per token), both run locally through Ollama. Three strategies: ReAct, ReWOO and ReWOO + repair. The same tools (version g5) were used for all six setups. There are 5 datasets: `toy_shell` (a small simulation of satellites and debris in one orbital shell; its results are in `data/toy_shell/`) and 4 synthetic business datasets with hidden problems. The test has 21 questions, each asked twice (seeds 1 and 2), which gives 42 answers per setup and 252 in total. Questions and expected answers were written **before** any run (`domains/*/bench.yaml`).

| Model | Strategy | Non-empty answers | Hidden-problem checks passed | Free-SQL share of data reads |
|---|---|---|---|---|
| 12B | ReAct | 39/42 | 25/56 | 22% |
| 12B | ReWOO | 40/42 | 27/56 | 11% |
| 12B | ReWOO + repair | 42/42 | 29/56 | 12% |
| 26B | ReAct | 42/42 | 27/56 | 39% |
| 26B | ReWOO | 42/42 | 26/56 | 30% |
| 26B | ReWOO + repair | 42/42 | 32/56 | 24% |

The 56 checks per setup: 24 on `toy_shell` (4 expected findings × 6 answers) and 32 on the business datasets (16 hidden problems × 2 answers).

Full table (time, tokens, tool errors, untraceable numbers): `results/episode1/summary.csv`. Per-problem counts: `results/episode1/hard_points_by_problem.txt`.

**What these numbers say:**
- Every setup caught roughly half of the hidden problems (45-57%). With 56 checks per setup, not independent (two seeds per question; 24 of the 56 on `toy_shell`), differences of up to about 10 checks (about 18 percentage points) are within the noise. No setup wins.
- The 26B model bypasses the checked tools and writes free SQL more often: 31% of its data reads, against 14% for the 12B (totals over the three strategies). This test does not show whether that is why problems get missed.

**How to reproduce the counts** (no model needed, from the saved transcripts):

```bash
cd results/episode1/runs
python3 ../scripts/post_numbers.py          # -> the table above (summary.csv)
python3 ../scripts/hard_points.py react rewoo repair react_26b rewoo_26b repair_26b
```

**How to rerun the bench:** `git checkout episode1`, then `bash results/episode1/reproduce.sh` (all 30 runs, several hours on a laptop). New runs are saved in `bench/runs/`: run the counting scripts from there. Model outputs are sampled at temperature 1, so a rerun gives different transcripts and somewhat different counts.

## Episode 2: results

**Setup.** The same 6 setups and datasets, with clearer tools (g5.1, see `CHANGELOG.md`) and one new question
(`gen_rentals` 4a): 22 questions × 2 seeds × 6 setups = 264 answers. This time every wrong answer is also classified
by its source: a flaw in the data, the limits of the tools, or the model's reasoning.

| Model | Strategy | Wrong / 44 | data / tools / model | Free-SQL share of tool calls |
|---|---|---|---|---|
| 12B | ReAct | 8 | 3 / 1 / 4 | 30% |
| 12B | ReWOO | 3 | 0 / 1 / 2 | 13% |
| 12B | ReWOO + repair | 4 | 0 / 2 / 2 | 17% |
| 26B | ReAct | 4 | 4 / 0 / 0 | 45% |
| 26B | ReWOO | 5 | 3 / 2 / 0 | 18% |
| 26B | ReWOO + repair | 3 | 3 / 0 / 0 | 18% |

**What these numbers say:**
- A flaw in the data matters more than the model or the strategy: 17/36 wrong on the 3 flawed questions, 10/228 on the others.
- One station is listed twice. The question that joins with it (4a) is wrong in 12/12 answers, every setup, both seeds;
  a question on the same data that does not need that table (2a) is right in 12/12.
- The 12B fails mostly on its own reasoning (8 of 15), the 26B mostly walks into the flaws in the data (10 of 12).
- 46-72% of the tokens go into figuring out the data, again at every question; 91-97% of the time is the model writing.

Full counts, how answers were graded and the limits: `results/episode2/README.md`. To reproduce every number of the
post from the saved transcripts (no model needed): `python3 -B results/episode2/scripts/post_numbers.py`.

## Limits

- **Small test.** 6 answers per new dataset and setup, 56 checks per setup: the bench can only detect large differences.
- **Synthetic data.** The four business datasets were generated by an AI with deliberate problems; the simulation is a toy model with parameters chosen by hand, not calibrated.
- **Keyword counting.** Hidden problems are counted with keyword rules on the final answer (`scripts/hard_points.py`) and spot-checked by hand. A keyword match does not prove the reasoning is right, and some rules are broad (for example a bare number). The keyword rules were revised once (1/11), after reading some answers: 4 rules, listed in `hard_points.py`. Questions and expected answers were not changed.
- **Grading.** For the 12B setups, an AI graded one rubric criterion (B2, hits the target) against the expected answers; the sheets are marked provisional (`results/episode1/grading/`). The author reviewed by hand the answers that stated false facts (notes not included). For the 26B setups, only the keyword count exists.
- **One model family.** Both models are Gemma 4. ReAct and ReWOO also use slightly different instructions (`prompt=v3` vs `v4`), so the strategy is not the only difference between them.
- **Code limits.** Only one provider is implemented (Ollama). The number checker traces values, sums and differences, but not products, ratios or units. Old tool results are not compacted in long conversations. `compare_groups` decides with Welch's t interval, which assumes roughly normal values within each group; the bootstrap interval is shown only with at least 30 values per group.

## Status and next steps

- **This code is frozen.** It is the "before" for the next comparisons: one agent that, at every question, explores,
  cleans, interprets and answers. `tools.py` (about 1,800 lines) grew with every fix, each with a test.
- **A flat view was tried and set aside** (one clean table, two tools): in a short probe most errors came from the
  texts the model reads, written at different times for different readers.
- **Next: Archimede**, a workflow instead of one agent, built from scratch: *Prepare* the data once per dataset
  (AI + code), *Analyze* with two tools (`query`, `calculate`), *Review* the answer with code (every number must come
  from the data). Each step gets its own test; then the same questions, compared with these 264 answers.

## Safety

The DuckDB file is opened read-only; CSV/Parquet files are only read, into an in-memory copy (the files on disk are never written). After opening, DuckDB's external access is switched off and its configuration locked, so a query cannot read other files, write files or load extensions. Each `query_results` call runs one statement starting with `SELECT` or `WITH`. No code written by the model is ever executed: `calculate` parses arithmetic into a tree and evaluates only numbers and the basic operators.

## Repository layout

```
sim_analyst/        the analyst: tools, graphs (ReAct, ReWOO, repair), prompts, number checker
domains/            one folder per dataset: context.md, config.yaml, bench.yaml (questions + expected answers)
bench/              bench runner, blind grading helpers, rubric, error classifier
tests/              pytest suite (runs on a small fake database, no model needed)
data/toy_shell/     frozen results of the toy simulation (DuckDB)
data/synthetic/     the 4 synthetic datasets (CSV) + SHA256SUMS.txt
results/episode1/   transcripts of all 30 runs, grading, counting scripts, summary
results/episode2/   the same for episode 2, plus the post's numbers (numbers.txt, numbers.json)
CHANGELOG.md        version history of the tools (episode 1 used g5, episode 2 g5.1)
```

## Data

- `data/toy_shell/`: frozen results of a toy simulation written by the author (one orbital shell, satellites and debris; parameters chosen by hand). The simulation code is not part of this repository: the bench only needs its results.
- `data/synthetic/`: synthetic datasets (bike rentals, app sessions, power plants and weather stations; column names in Italian) generated with an AI assistant (Claude) for a study project. They contain no real people. Names of places are real Italian locations, but every value is invented.

Both are released under the same licence as the code.

The four `context.md` files are translations of each dataset's description (`CONTESTO.md`, not included). They are left exactly as they were given to the analyst during the runs.

## Licence

MIT, see [LICENSE](LICENSE).
