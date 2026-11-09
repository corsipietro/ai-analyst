# Changelog of the tools

`TOOLS_VERSION` in `sim_analyst/tools.py`. Episode 1 used **g5**, episode 2 **g5.1**. Dates are 2026.

- g5.1 (2-3/11; `TOOLS_VERSION` still reads `g5`, so the runs record `tools: g5`): `x` instead of `time`, and the tools work with any x (irregular, decimal, date, timestamp); every result repeats its filters and groups and names the metric it aggregates; a warning when a filter mixes several quantities; shorter sampled series and 3 significant digits; JSON paths written `a[0].b`; filters accept `x`, `at` and `equals`; groups describe themselves; the thresholds of the tools moved to `settings.yaml` (`analysis`); both ReWOO graphs repair a plan once when none of its steps read any data (`check_quality` counts as reading data). New question `gen_rentals` 4a, and a `defects` list in `domains/gen_rentals/bench.yaml`, both written before the runs. New: `bench/classify_errors.py` (wrong answers by source) and `bench/extract_to_read.py`.

- g5 (31/10): the model declares, the code verifies. describe_database adds nulls, role (unique per row / repeats) and links (columns of other tables that contain the column's values, with the orphans); the mixing check only looks at text columns that are categories (not identifiers); the g4 refusal of several rows per unit is gone: summarize_results and compare_groups take 'unit' (average per unit first) and, without it, add a note on repeated rows (compare_groups then gives no verdict); describe_trends accepts a DATE/TIMESTAMP time with 'period'; new tool check_quality (NULLs, runs of NULLs, duplicates, text variants, numbers as text, orphans); summarize_results adds n_rows, n_null and sum.

- g5 fixes (31/10, before the first g5 bench): mixing refuses only a varying JSON path or a "measure-name" category (removing it makes at least half of the rows duplicates: the metric column of a long table); any other category that varies gives a note "aggregated over ..."; filters accept {"between": [a, b]}, ">=", ">", "<", "<=" and {"is_null": true/false}, and refuse None in plain values; joins are LEFT JOINs from the value's table (unmatched rows keep NULL in the joined columns, with a note); wording without domain words; the error of describe_trends says which argument is wrong; the repetition note mentions time + at only when there is a time column.

- g4.1 (31/10): serialization only (dates, datetimes, times, intervals, decimals, uuids and bytes become JSON-safe in stats.clean; date and timestamp columns get min and max in describe_database); no change on data without dates.

- g4 (30/10): compare_groups refuses (error 'column', no numbers) a group that holds several rows for the same value of the key linking the value's table to the other tables (e.g. all the times of one run pooled).

- g3 (30/10): compare_groups decides from Welch's t interval (bootstrap only added from bootstrap.min_n values per group, never deciding); summarize_results applies the 20-values limit on 'at' and no longer says that each row is a replicate; query_results' all-NULL warning mentions json_paths; new tool calculate (safe arithmetic, calculator.py); 'observations' instead of 'replicates' in every text the model reads.

- g2 (29/10): mixed rows give an error, not a warning; g1 = first version.
