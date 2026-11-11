# Which Boston restaurants should inspectors visit first?

> **Work in progress.** Results are added as each phase completes.

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
  - the establishment's own inspection record and the types of violations it had,
  - its owner's other establishments,
  - nearby 311 complaints and rodent reports,
  - how strict recent inspections have been in its area.
- **The action:** sort the month's list by the score and visit the highest-risk places first.
- **What stays fixed:** the number of inspections, and who is due. The city's schedule still
  decides who is due; we change only the **order** within the month.
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
  history, its violation types, its owner's other establishments, nearby complaints, and recent
  inspection outcomes in its area.
- **Phase 2 · inspector comments.** Language models, from simple keyword models to a fine-tuned
  transformer, read the notes inspectors wrote at earlier visits and are added to the Phase 1
  model.

How the data was prepared, how the metric was chosen, and what the results assume:
**[TECHNICAL.md](TECHNICAL.md)**.

## Results

### Phase 1 · structured data
Tested on 2022–2024 (about 17,000 routine inspections, each year predicted from earlier years
only):

| Order of visits within the month | Share of first-half visits that find a Priority violation | Violations found earlier than today |
|---|---|---|
| Today's order | 21.5% | — |
| Simple rule: places whose previous inspection found a Priority violation or failed go first | 31.0% | 4.1 days |
| **Machine-learning model** | **33.8%** | **5.8 days** |
| Perfect hindsight (the ceiling) | 43.2% | — |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/readme_capture_curve_dark.png">
  <img src="results/figures/readme_capture_curve_light.png" width="720" alt="Share of each month's Priority violations found as inspectors work through the month's list, 2022–2024. By the halfway point, today's order has found 50% of them, the simple rule 72%, the machine-learning model 78%, and perfect hindsight 100%.">
</picture>

*How to read it:* each line follows inspectors through a month's list and shows how many of that
month's serious violations they have found so far. Every order finds them all by the end of the
month; the higher the line, the sooner they are found. By the halfway point, today's order has
found half of them, the simple rule 72%, and the model 78%.

- **The city's current order carries no risk information** within the month. A simple rule
  already finds far more violations early.
- **The model adds about 3 more Priority violations found per 100 early visits** on top of the rule,
  consistently in every year and for every type of establishment, and its risk scores are well
  calibrated.
- **The simplest model worked best.** More complex models did no better: the limit is the
  information in the data, not the algorithm.
- **Besides an establishment's own record, how strict recent inspections have been nearby matters.**
  This likely reflects inspectors as much as kitchens, a point to handle carefully in any use.
- **What the model misses is mostly surprises:** places with clean records whose kitchens slipped.

### Phase 2 · inspector comments
_Coming soon._ What we expect, and what to watch for:
- **Why it could help:** comments describe what counts can't, such as how severe a problem was,
  whether it keeps recurring, and how the kitchen is managed. Among places with a clean last
  inspection, those with more inspector notes are far riskier (8% vs 30%), and mentions of mice or
  "repeat" violations signal higher risk.
- **Why it may not help much:** much of that signal overlaps with violation counts the model
  already uses. The comments are also short, and inspectors write in different styles.
- **Expectation:** a small gain at most. The experiments run cheapest-first and stop early if the
  simple keyword model finds nothing beyond the counts. Even then, comment snippets can make the
  priority list easier to explain to inspectors.

## Implications and next steps
_Coming after the final evaluation._

## Reproduce
```bash
uv venv --python 3.14 && source .venv/bin/activate    # any Python 3.10+
uv pip install -e ".[dl]"
python -m inspections.download                         # dated data snapshot into data/raw/
```
Then run the notebooks in `notebooks/` in order. Full setup: [CONTRIBUTING.md](CONTRIBUTING.md).

## Data
Public open data from [Analyze Boston](https://data.boston.gov/dataset/food-establishment-inspections).
Personal names are dropped or hashed; no raw data is stored in this repo.
