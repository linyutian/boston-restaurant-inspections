# Which Boston restaurants should inspectors visit first?

> **Status:** complete. Analysis in `notebooks/`, reusable code with tests in `src/inspections/`.

## Motivation
Boston's health inspectors routinely visit about 4,000 restaurants and food stores, mostly in
calendar order. A kitchen with active hazards can wait weeks while a clean one is visited first.
We keep the number of inspections the same and change only the **order**: a monthly priority list
that puts the places most likely to have serious food-safety violations first. Inspectors find more
problems per visit, and the public is exposed to hazards for fewer days.

## Problem statement
- **Who:** every sit-down restaurant, take-out restaurant and retail food store that the city has
  due for a **routine** inspection in the coming month.
- **What we predict:** the probability that the visit will find **at least one Priority
  violation**, the FDA Food Code category for hazards that directly cause foodborne illness
  (holding temperatures, cross-contamination, handwashing, sanitizing). We use only information
  known **before the month starts**:
  - the establishment's own inspection record and its recent trend,
  - how strict recent inspections have been in its area,
  - the notes inspectors wrote at its last two inspections (Phase 2).

  Other signals (its owner's other establishments, nearby 311 complaints, violation types, season)
  were tested and added nothing.
- **The action:** sort the month's list by the score and visit the highest-risk places first.
- **What stays fixed:** the number of inspections, and who is due (the city's schedule decides).
- **What we optimize:** hazards found **earlier**. The headline is the share of visits that find a
  Priority violation in the first half of each month, compared with today's order and with the
  rule "go back first to the places that were worst last time". That translates into:
  - **Residents:** hazards found and fixed days sooner.
  - **The city:** more problems found per inspector visit, with the same staff.
- **Out of scope:**
  - deciding who is due or how often,
  - complaint-driven and other irregular visits,
  - pass/fail grades, closures, or illness outbreaks.

## Approach
Twenty years of public inspection records, plus 311 complaints. Every model is tested on years it
never saw, and compared with today's order and with simple rules a city analyst could apply
without machine learning.

- **Phase 1 · structured data.** Classic machine learning on each establishment's inspection
  history and on recent inspection outcomes in its area; more complex models, tuning and extra
  signals were tested along the way.
- **Phase 2 · inspector comments.** Language models, from simple keyword models to a fine-tuned
  transformer, read the notes inspectors wrote at earlier visits and are added to the Phase 1
  model.
- **Final test.** The chosen model is tested once on January 2025 – October 2026, with a fairness
  check by neighbourhood and income.

How the data was prepared, how the metric was chosen, and what the results assume:
**[TECHNICAL.md](TECHNICAL.md)**.

## Results

### Final test: January 2025 – October 2026
Every model and setting was built and chosen on 2020–2024. The final model was then tested **once**
on the 22 months that followed (about 11,200 routine inspections), and nothing was changed after
seeing the results.

| Order of visits within the month | Share of first-half visits that find a Priority violation | Violations found earlier than today |
|---|---|---|
| Today's order | 23.2% | — |
| Simple rule: places whose previous inspection found a Priority violation or failed go first | 30.3% | 3.0 days |
| **Machine-learning model** | **33.3%** | **4.9 days** |
| Perfect hindsight (the ceiling) | 44.7% | — |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/readme_capture_curve_test_dark.png">
  <img src="results/figures/readme_capture_curve_test_light.png" width="720" alt="Share of each month's Priority violations found as inspectors work through the month's list, January 2025 to October 2026. By the halfway point, today's order has found 52% of them, the simple rule 68%, the machine-learning model 74%, and perfect hindsight 100%.">
</picture>

*How to read it:* each line follows inspectors through a month's list and shows how many of that
month's serious violations they have found so far. Every order finds them all by the end of the
month; the higher the line, the sooner they are found. By the halfway point, today's order has
found about half of them, the simple rule 68%, and the model 74%.

- **The model finds about 3 more Priority violations per 100 early visits than the simple rule**
  (+3.0, 95% range +2.3 to +3.7). That meets the project's target of +3, though only just. The
  gain holds in both 2025 and 2026, for every type of establishment, and with two-week lists
  instead of monthly ones.
- **No neighbourhood or income group is worse off than today.** The largest gains are in
  lower-income areas, where serious violations are more common, and early visits find violations
  equally often in every income group.

### Business impact
What the same result means for a health department, per year, in four tiers: today's order is the
baseline, then the simple rule, the machine-learning model, and perfect hindsight as the ceiling.
Each line shows how many serious violations are still waiting to be found on each day of the
month; the shaded areas are the days of exposure each tier removes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/readme_impact_tiers_dark.png">
  <img src="results/figures/readme_impact_tiers_light.png" width="760" alt="Average number of serious violations not yet found on each day of the month, 2025–26, under four orders. Today's order declines roughly in a straight line from about 114 on day 1 to zero at month end; the simple rule and the model fall faster, and perfect hindsight reaches zero by mid-month. The simple rule removes 4,142 hazard-days a year, the model 2,539 more, and 8,582 remain between the model and the ceiling.">
</picture>

| Per year, 2025–26 | Today | Simple rule | Machine learning | Ceiling |
|---|---|---|---|---|
| Serious violations found in the first half of the month | 709 | 925 | **1,017** | 1,364 |
| Days of uncorrected hazards removed ("hazard-days") | — | 4,142 | **6,682** | 15,263 |
| Days sooner each serious violation is found | — | 3.0 | **4.9** | 11.2 |
| Share of the possible gain | — | 27% | **44%** | 100% |
| Extra inspections today's order would need to match it | — | 930 | **1,327** | 2,824 |
| Value of that inspector capacity, per year | — | $124,000 | **$177,000** | $376,000 |

- **Same staff, about 6,700 fewer days of uncorrected hazards a year.** The model finds about 300
  more serious violations in the first half of the month than today's order (43% more), each about
  5 days sooner. Matching that by adding inspections would take about 1,330 more a year, a fifth of
  the routine workload.
- **The simple rule delivers 60–70% of that on its own.** The model's addition, about 90 more
  violations found early and 2,500 hazard-days a year, is consistent: it beat the rule in 20 of the
  22 test months.
- **More than half of the possible gain remains out of reach.** It sits in places with clean
  records whose kitchens slipped, which past inspections give no reason to visit first.
- **Worth about $177,000 a year in inspector time, for about $7,200 a year to run.** Valued at the
  city's own payroll (about $133 per inspection), the extra capacity is worth $105,000–$279,000 a
  year across the assumptions; the simple rule alone accounts for about $124,000 of it. This is
  capacity the department gains, not cash it saves.
- **The health value is plausible but unmeasured.** At the USDA's average cost of a foodborne
  illness ($4,766 in 2023), preventing about 1.5 illnesses a year would cover the running cost. The
  data cannot show how many it prevents: that is for a pilot to measure.

Every calculation and assumption is in
[notebook 06](notebooks/06_business_impact.ipynb).

### How we got here
**Phase 1 · structured data.** Developed on 2020–2024 and checked on 2024: 34.8% against 32.4% for
the simple rule.
- **The city's current order carries no risk information** within the month.
- **The simplest model worked best.** More complex models, careful tuning and extra signals did no
  better: the limit is the information in the data, not the algorithm. The model is a logistic
  regression on 33 inputs.
- **Besides an establishment's own record, how strict recent inspections have been nearby
  matters.** This likely reflects inspectors as much as kitchens, but it barely changes which areas
  are visited early.
- **What the model misses is mostly surprises:** places with clean records whose kitchens slipped.

**Phase 2 · inspector comments.** A score from the inspectors' notes at the last two inspections is
added to the Phase 1 model.
- **The comments add a little, only at the very top of the list:** roughly one extra serious
  violation found each month in the first few days. On the final test, the model without comments
  reaches 33.0%, a difference too small to detect.
- **What the comments add is what kind of kitchen a place is** (raw meat, heavy prep). Most of the
  rest, such as many notes or mentions of mice, cockroaches or "repeat" violations, the inspection
  record already shows.
- **The simplest text model worked best.** Word counts beat sentence embeddings and a fine-tuned
  transformer (DistilBERT).
- **For explaining the list to inspectors,** showing a place's last inspection notes is clearer
  than snippets picked by the model.

## Implications and next steps
- **Start with the simple rule.** Sorting the month's list by the previous inspection needs no
  model and delivers most of the benefit. The model adds about 90 more serious violations found
  early each year.
- **If a model is used, the version without comments is the practical choice.** It is simpler to
  run and explain, and on the final test it found violations as often as the version with comments,
  within measurement error.
- **Use the scores to rank, and refit yearly.** The model under-predicted 2026, when violations
  became more common. The order stayed right, but scores shown as probabilities would need
  updating.
- **Explain the inputs openly.** Nearby inspection outcomes partly reflect inspectors, and the
  fairness check covered the order within the month only.
- **The economics favour trying it.** The list costs a few analyst hours a month and adds no
  inspections. Its extra inspector capacity is worth roughly 25 times that cost, and preventing one
  or two foodborne illnesses a year would pay for it on its own.
- **A working starting point exists.** `python -m inspections.rank` produces a month's list from
  the public data, refitting the model each time, with each place's last inspection result and
  notes beside its rank.
- **Next: a live pilot.** Only a pilot can confirm that inspectors can follow the order and that
  conditions hold between the list and the visit. A twice-monthly list would give fresher inputs
  at the same per-visit gain.

## Runbook: reuse the code
The reusable code is the Python package in `src/inspections/`. Clone the whole repository: the
package installs from it together with its pinned dependencies, and the data is downloaded
separately, never stored in the repo. The notebooks explain how each choice was made; you don't
need them to run the code.

### 1 · Set up (once, about 5 minutes)
```bash
git clone https://github.com/linyutian/boston-restaurant-inspections.git && cd boston-restaurant-inspections
uv venv --python 3.14 && source .venv/bin/activate     # Python 3.12 or newer
uv pip install -r requirements-lock.txt && uv pip install -e . --no-deps   # exact tested versions
python -m inspections.download                          # public data snapshot into data/raw/ (about 400 MB)
python -m inspections.prepare                           # clean tables into data/processed/ (seconds)
pytest                                                  # 34 tests, about 10 seconds
```

### 2 · Produce a month's priority list
```bash
python -m inspections.download --only inspections && python -m inspections.prepare   # refresh the data first
python -m inspections.rank --month 2026-12 --due due.csv
```
- **Input:** `due.csv` has one column, `licenseno`, listing the establishments due for a routine
  inspection that month. The city's schedule decides who is due, so the list must come from the
  city.
- **Output:** `data/outputs/priority_list_2026-12.csv`, highest risk first: rank, risk score,
  license type, date and result of the last routine inspection, and its inspector notes with
  personal names removed. Visit the top half in the first half of the month.
- **Options:**
  - `--with-text` adds the inspector-comment score, which gave no detectable gain.
  - `--out` chooses another output path.
  - Leaving out `--due` replays a past month's actual routine visits, useful for checking the list.
- The model is refit on every run, on all routine inspections from 2020 to the end of the previous
  month, using only data from before the month starts.

### 3 · Use the pieces in your own code
```python
from inspections import features, models, prepare
from inspections.metrics import bootstrap_diff, hit_rate, per_month

t = prepare.load_processed()                                  # visits, violations, licenses, complaints
visits = t['visits']
rows = visits[visits.in_prediction_set & visits.split.isin(['train', 'valid'])]   # routine visits 2020–2024
X = features.build_model_b(features.decision_keys(rows), visits,                  # 33 inputs as of each month's start
                           features.violation_group_counts(t['violations']), t['licenses'])
year = rows.visit_ts.dt.year.to_numpy()
scores = models.fit_and_score(X, rows.hit.to_numpy(), train=year < 2024, score=year == 2024)

ev = rows.loc[year == 2024, ['month', 'visit_ts', 'hit']].assign(
    tiebreak=0.0, model=scores.model_b, rule=features.rule1_score(X.loc[year == 2024]))
print(hit_rate(per_month(ev, 'model')), hit_rate(per_month(ev, 'rule')))          # 34.8% vs 32.4%
print(bootstrap_diff(per_month(ev, 'model'), per_month(ev, 'rule')))              # 95% CI of the difference
```
- **Score your own ordering:** put it in a column of an evaluation frame like `ev` (higher = visit
  earlier) and pass the column name to the `inspections.metrics` functions.
- **Try your own model:** use `X` and `rows.hit` as features and labels, and keep training years
  before the years you score.
- **Text tools:** `inspections.text` holds the comment masking (`clean_comment`) and the
  display-time name scan (`NameScan`), for any work that shows inspector notes.

| Module | Purpose |
|---|---|
| `download`, `prepare` | fetch the public data; turn it into clean tables of visits, violations, licenses and complaints |
| `features` | the model's inputs, computed from data before each month only |
| `models` | model (b), and model (b) plus the comment score |
| `metrics` | first-half hit rate, confidence intervals, days earlier, capture curve |
| `text` | comment cleaning and personal-data checks |
| `rank` | the monthly priority list |

### 4 · Adapt it to another city
The method needs only a history of inspections with dates, results and violation severity per
establishment. To port it:
- **Data source:** replace the IDs in `config.py` and the download functions in `download.py`.
- **Data rules (`prepare.py`):**
  - map the city's result codes to fail / pass / filed (`RESULT_ROLE`);
  - set which violation level counts as serious (`viol_level == '***'` here);
  - check the 90-day rule that separates routine visits from follow-ups;
  - adjust the address and coordinate parsing.
- **Time splits:** set the dates in `config.py`. Choose settings on earlier years, and keep the
  last period for one final test.
- **Validation:** rerun `pytest`, then notebooks 01–02 to check the visit rules and the baselines on
  the new data before trusting any model.

### 5 · Re-run the analysis
Run the notebooks in `notebooks/` in order, with Restart & Run All. Each writes its results to
`results/experiments.csv`, and each stops if one of its checks fails. Notebook 04's transformer
experiment retrains for hours unless its cached scores are present, so keep the machine awake.
Contributor guide and rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## Data
Public open data from [Analyze Boston](https://data.boston.gov/dataset/food-establishment-inspections),
plus median household income by ZIP code from the US Census (American Community Survey) for the
fairness check, and the city's Employee Earnings Report (pay by job title, no names) for the cost of
an inspection. Personal names are dropped or hashed; no raw data is stored in this repo.
