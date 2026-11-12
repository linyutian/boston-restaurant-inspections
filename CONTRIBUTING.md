# Contributing

How the repo is organized and how to work in it. What the data contains and why the methodology
looks the way it does is in [TECHNICAL.md](TECHNICAL.md).

## Layout
```
README.md          public overview: motivation, problem statement, results
TECHNICAL.md       methodology: data findings, prediction set, metric, results, limitations
PLAN.md            working plan: next steps, open questions, backlog, risk register, decision log
CONTRIBUTING.md    this file
pyproject.toml     package metadata and compatible dependency ranges
requirements-lock.txt  the exact environment that produced the results
src/inspections/   the package (below)
tests/             pytest suite: synthetic-data unit tests plus one check against the real data
notebooks/         the analysis, run in order: 01_data_and_labels, 02_metric_and_baselines, 03_tabular_models,
                   03b_tuning_round, 04_text_models, 05_final_test
results/           committed outputs: figures/ and experiments.csv
data/              local only, git-ignored: raw/ (dated snapshots), processed/ (built by prepare), outputs/ (priority lists)
```

**The package** (`src/inspections/`). The notebooks import from it, so each rule is written once.

| Module | What it holds |
|---|---|
| `config.py` | paths, data source IDs, split dates, seed |
| `download.py` | dated raw snapshot: inspections, 311, weather, Census income (`python -m inspections.download`) |
| `prepare.py` | raw → processed tables, the rules derived in notebook 01 (`python -m inspections.prepare`) |
| `features.py` | decision-time features: model (b)'s 33, and the Phase 1 groups that were dropped |
| `text.py` | comment masking, text windows, the display-time name scan |
| `models.py` | model (b), the TF-IDF text score and stacking (`fit_and_score`) |
| `metrics.py` | the ranking metric: first-half hit rate, bootstrap CIs, days earlier, capture curve |
| `checks.py` | the `check()` helper used for the risk register |
| `rank.py` | a month's priority list (`python -m inspections.rank`) |

Exploratory code that only one notebook uses (a LightGBM search, a comparison table) stays in that
notebook.

## Setup
```bash
brew install libomp                                      # macOS only: OpenMP runtime LightGBM needs
uv venv --python 3.14 && source .venv/bin/activate      # Python 3.12+ (macOS system Python is 3.9)
uv pip install -r requirements-lock.txt && uv pip install -e . --no-deps   # the exact tested versions
# or, for compatible newer versions: uv pip install -e ".[dl]"  (drop [dl] to skip the notebook 04 text experiments)
python -m ipykernel install --user --name boston-inspections --display-name "Python 3.14 (boston-inspections)"
python -m inspections.download                           # or: --only 311 --years 2024
python -m inspections.prepare                            # builds data/processed/
pytest                                                   # about 10 seconds
```
The download writes a dated snapshot plus `data/raw/manifest.json` (row counts, the portal's own
count, SHA-256), so every result traces back to the data it used. The source updates daily, so
don't overwrite old snapshots. Then run the notebooks in order; notebook 01 explains and checks the
rules that `prepare` applies, and confirms both give identical tables.

**A month's priority list:**
```bash
python -m inspections.rank --month 2026-12 --due due.csv     # due.csv: a licenseno column, the licenses due that month
python -m inspections.rank --month 2026-10                   # replay a past month's actual routine visits
```
It refits model (b) on every labelled routine visit up to the end of the previous month and writes
`data/outputs/priority_list_<month>.csv`: rank, risk score, the last routine inspection's result
and its comments (masked for personal data). `--with-text` adds the comment score.

## Rules
1. **Decision-time features only.** A feature for a visit may use only data from before the first
   day of the visit's month. The truncation and timestamp checks in notebook 03 and the truncation
   tests in `tests/` must pass.
2. **The test period (2025-01 → 2026-10) is evaluated once**, at the end of Phase 2. Choose models
   and settings on the 2022–23 backtest folds, and judge them on 2024, which is never used for
   selection.
3. **Every notebook runs top to bottom** (Restart & Run All) before it is committed, so saved
   outputs are real and in order. Checks use `assert`, so a failure stops the notebook. Change
   shared code in `src/inspections/`, not in a notebook, and run `pytest` before committing.
4. **Log every model run** in `results/experiments.csv` (date, notebook, model, HR@50, comparisons,
   PR-AUC, parameters). Re-running a notebook replaces its own rows.
5. **No data, model files, secrets (`.env`) or personal names** in git.
6. Branch per topic, open a pull request to `main`, and keep `main` runnable.
