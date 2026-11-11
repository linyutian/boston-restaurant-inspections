# Contributing

How the repo is organized and how to work in it. What the data contains and why the methodology
looks the way it does is in [TECHNICAL.md](TECHNICAL.md).

## Layout
```
README.md          public overview: motivation, problem statement, results
TECHNICAL.md       methodology: data findings, prediction set, metric, results, limitations
PLAN.md            working plan: next steps, open questions, backlog, risk register, decision log
CONTRIBUTING.md    this file
pyproject.toml     package metadata and dependencies
src/inspections/   reusable code: config.py (paths, data IDs, split dates), download.py
notebooks/         the analysis, run in order: 01_data_and_labels, 02_metric_and_baselines, 03_tabular_models, 03b_tuning_round, ...
results/           committed outputs: figures/ and experiments.csv
data/              local only, git-ignored: raw/ (dated snapshots) and processed/ (built by notebook 01)
```
Folders are created when there is something to put in them. During Phases 1–2 the analysis lives
in notebooks; stable code moves into `src/inspections/` with tests in Phase 3 (see PLAN.md).

## Setup
```bash
brew install libomp                                      # macOS only: OpenMP runtime LightGBM needs
uv venv --python 3.14 && source .venv/bin/activate      # any Python 3.10+ (macOS system Python is 3.9)
uv pip install -e ".[dl]"                                # drop [dl] if you skip the text models
python -m ipykernel install --user --name boston-inspections --display-name "Python 3.14 (boston-inspections)"
python -m inspections.download                           # or: --only 311 --years 2024
```
The download writes a dated snapshot plus `data/raw/manifest.json` (row counts, the portal's own
count, SHA-256), so every result traces back to the data it used. The source updates daily, so
don't overwrite old snapshots. Then run the notebooks in order; notebook 01 builds
`data/processed/`.

## Rules
1. **Decision-time features only.** A feature for a visit may use only data from before the first
   day of the visit's month. The truncation and timestamp checks in notebook 03 must pass.
2. **The test period (2025-01 → 2026-10) is evaluated once**, at the end of Phase 2. Compare models
   on the rolling 2022–24 backtest.
3. **Every notebook runs top to bottom** (Restart & Run All) before it is committed, so saved
   outputs are real and in order. Checks use `assert`, so a failure stops the notebook.
4. **Log every model run** in `results/experiments.csv` (date, notebook, model, HR@50, comparisons,
   PR-AUC, parameters). Re-running a notebook replaces its own rows.
5. **No data, model files, secrets (`.env`) or personal names** in git.
6. Branch per topic, open a pull request to `main`, and keep `main` runnable.
