# Plan

Working plan for you and me: what's next, how we work, open questions, risks, backlog and the
decision log. The reasoning behind settled decisions lives in [TECHNICAL.md](TECHNICAL.md); the
public overview is [README.md](README.md).

## Goal
With the **same number of inspections**, give Boston's inspectors a monthly priority list that puts
the establishments most likely to have a Priority violation first. The city gets more problems
found per inspector visit; residents get hazards found and fixed sooner.
Problem statement: README.md · task definition, metric and baselines: TECHNICAL.md §1–5.

## Status and next steps
**Where we are**
- **Phase 1 (tabular) done:** notebooks 01–03. Logistic regression beats Rule 1 (previous
  inspection) by +2.9 hits per 100 visits [+2.3, +3.4], narrowly under the +3 exit bar
  (TECHNICAL.md §7). Phase 1 results are in the README.
- **Review:** notebook 01 ✅ reviewed. Notebook 02 review in progress: §1–§5 discussed, with these
  changes applied:
  - Rule 2 explained, with the hit rate at each score level;
  - capture-curve reading added under the chart;
  - comparators renamed (Rule 1 / Rule 2 are our proposals; only current practice is the city's).

  Notebook 03 is not yet reviewed.
- **Documentation:** README (public, high-level: problem statement, approach by phase, Phase 1
  results, Phase 2 expectations) · TECHNICAL.md (methodology, first pass, extended as the review
  goes) · CONTRIBUTING.md (how to work) · this file.
- **Repository:** [linyutian/boston-restaurant-inspections](https://github.com/linyutian/boston-restaurant-inspections) (private). Phase 1 work and the docs are pushed; from here, one commit per reviewed notebook or change set.

**What's next, in order**
1. **Finish the review** of notebooks 02 and 03, and apply your feedback (including additions to
   TECHNICAL.md).
2. **Tuning round for judgment calls** (new experiment E7 in notebook 03): tune the settings marked
   "tune" in [Judgment calls](#judgment-calls-tune-or-test) on the backtest, run the sensitivity
   checks marked "test", and log every run. Expected gain: modest (the notebook 03 tuning surface
   was flat), but the exit bar was missed by only 0.1.
3. **Two-week replay** (new section in notebook 03, after §5): the same comparison (current
   practice, Rule 1, model, oracle) on two-week blocks, so no visit moves more than about 7 days.
   It bounds how much the results rely on within-month stability (TECHNICAL.md §4).
4. **Phase 2 go / no-go:** after steps 2–3, decide whether to start the inspector-comment
   experiments. Current recommendation: go. Phase 2 is part of the project design, and the
   remaining misses need information the counts don't have. Expectations are in the README.
5. **Phase 2, then the one-time test** (notebook 05), then Phase 3.

## How we work
1. **Phases 1 and 2 happen in Jupyter notebooks**, one per step, each reading top to bottom:
   purpose → short markdown / code / output units → `assert` checks for the risk register →
   findings and decisions.
2. **I run each notebook top to bottom** before handing it over, so saved outputs are real and in
   order. I stop at its open decisions for your review.
3. **Helpers start inside the notebook.** Stable code moves into `src/inspections/` in Phase 3.
4. **Documentation:** settled reasoning goes in TECHNICAL.md (a living document, extended as you
   review); this file keeps one-line decisions; the README stays high-level.
5. **Commits:** one per reviewed notebook.
6. **Environment:** project `.venv` (Homebrew Python 3.14) with all dependencies, kernel
   "Python 3.14 (boston-inspections)".

| # | Notebook | Question | Status |
|---|---|---|---|
| 01 | `01_data_and_labels` | What is one inspection, a routine visit, a hit? | ✅ done · reviewed |
| 02 | `02_metric_and_baselines` | How good are current practice and simple rules, and what is the ceiling? | ✅ done · in review |
| 03 | `03_tabular_models` | Does ML on tabular data beat the best rule? | ✅ done · not yet reviewed |
| 04 | `04_text_models` | Does inspector-comment text add value? | Phase 2 |
| 05 | `05_final_test` | One-time test-period evaluation, impact, fairness check | Phase 2 |

## Phases
**Phase 1, tabular (done).** Exit criterion: beat Rule 1 (previous inspection) on the pooled 2022–24 backtest by ≥ 3 hits per
100 visits with a CI excluding zero. Outcome: +2.9 [+2.3, +3.4], narrowly not met.

**Phase 2, inspector-comment text (nb 04–05).** Input: comments from the establishment's
*previous* visits, de-duplicated. Experiments, cheapest first, each kept only if it beats the
current best on the backtest:
1. TF-IDF + logistic regression
2. Frozen sentence embeddings as extra features
3. Fine-tuned small transformer (DistilBERT / ModernBERT); its out-of-fold score feeds the
   tabular model
4. Tuning rounds (past visits included, text length, model, learning rate)

Stop after two experiments in a row add nothing. Then calibrate, retrain on 2020–2024, and run the
test period once (nb 05), including impact numbers and a fairness check by neighbourhood.

**Phase 3, consolidate and publish.** Move stable code into `src/inspections/` with tests, pin
dependencies, complete the README (final results, implications and next steps).

## Judgment calls: tune or test
Every value set by judgment or heuristic rather than learned. The split matters:
- **Tune:** values that only change *how well we predict*. Tune them on the backtest: folds 1–2
  (2022–23) for selection, 2024 as the clean check, the test period never.
- **Test:** values that define *what* is predicted or *how* it is judged. Tuning them for a higher
  score would move the goalposts, so we only check that conclusions hold when they change.
- **Fixed by design:** values that must stay fixed for a comparison to mean anything.

| Judgment call | Current value | Where | Action | Search range / check |
|---|---|---|---|---|
| **Model hyperparameters** | | | | |
| Logistic regression strength | `C = 0.1`, L2, *never tuned* | nb 03 E1 | **tune** | C ∈ {0.001 … 10}; L1 vs L2 |
| LightGBM hyperparameters | 16-trial random search | nb 03 E3 | **tune** | wider search (e.g. 60 trials) incl. `min_child_samples`, `reg_alpha`, `subsample` |
| Class weighting | none | nb 03 | **tune** | balanced vs none (affects calibration more than ranking) |
| **Feature settings** | | | | |
| Local-rate radius and windows | 1 km; 90 / 180 days | nb 03 area features | **tune** | 0.5 / 1 / 2 km; 60 / 90 / 180 / 365 days |
| 311 radius and windows | 100 m; 90 / 365 days | nb 03 | **tune** | 50 / 100 / 250 m; or drop the group (it adds about nothing) |
| Recency weight of the EWMA rates | α = 0.5 | nb 03 trend | **tune** | 0.2 / 0.35 / 0.5 / 0.7 |
| "Last N routine visits" window | N = 3 | nb 03 trend | **tune** | 2 / 3 / 5 |
| Violation keyword groups | 8 groups, hand-written rules | nb 03 §2.1 | **tune** (lightly) | merge or split groups; check coverage stays ≥ 95% |
| Training window | 2020+ (E5 tried 2012+) | nb 03 | **tune** | 2019+ / 2020+ / 2021+ |
| **Definitions (what is predicted)** | | | | |
| Cycle gap for a new routine visit | 90 days | nb 01 §6 | **test** | already shown stable for 60–180 days; re-check that results hold |
| Irregular cut-off | < 30 days after a pass | nb 01 §6 | **test** | 14 / 30 / 45 days |
| Complaint attribution | nearest within 25 m, 0–30 days | nb 01 §7 | **test** | 15 / 25 / 50 m; 14 / 30 / 45 days |
| Label start | 2020 | nb 01 §8 | **test** | fixed by the Food Code change; no tuning |
| **Evaluation design (how it is judged)** | | | | |
| First-half cut | 50% | nb 02 | **test** | capture curve already reported at 5–100% |
| Planning period | monthly | nb 02 | **test** | the queued two-week replay |
| Exit bar | +3 hits per 100 visits | nb 02 | **fixed by design** | changing it after seeing results would defeat its purpose |
| **Comparators** | | | | |
| Rule 2 (points checklist) weights | +2 / +1 / +1 / +1 | nb 02 §4 | **fixed by design** | it represents a no-data checklist. Optional extra comparator: a *learned* scorecard (logistic regression on the same four items) |

## Known weak points
1. **Gains over the best rule are modest** (about +3 per 100 visits). The rule itself is a valid
   recommendation; ML has to earn its place.
2. **Within-month stability is assumed** by the evaluation; gains are likely somewhat optimistic
   (the queued two-week replay bounds this).
3. **Part of the signal is the inspector, not the kitchen** (local strictness), which raises a
   fairness and communication question.
4. **The text phase may add little;** the cheapest-first ladder and stop rule limit the cost.

## Risk register and checks
Each risk has a check that runs in the named notebook. "Assert" means the notebook stops if it
fails.

| ID | Risk | Check | Where |
|---|---|---|---|
| **Data integrity** | | | |
| R1 | Download truncated or broken (the URL changes daily) | ✅ Rows parsed = rows counted = portal total (908,247 on Nov 10); manifest SHA-256 recorded | nb 01, assert |
| R2 | Timestamps are UTC; evening visits land on the wrong day or month | ✅ Converted to America/New_York (0.5% of rows change date) | nb 01, assert tz-aware |
| R3 | An inspection's rows don't share one timestamp, or there are several results per license-day | ✅ Visit = (`licenseno`, `resultdttm`); only 19 visits carry two result codes | nb 01, assert |
| R4 | **History is rewritten later**: `viol_status` / `status_date` updated after follow-ups, so past rows carry future information | ✅ Real but narrow (6% of rows, mostly on follow-ups). `viol_status` / `status_date` are never used as features | nb 01 |
| R5 | **Snapshot-only fields leak the future**: `licstatus` (Inactive = closed later), `expdttm`, current `legalowner`, `businessname` | ✅ Not saved to any processed table (asserted) | nb 01, assert |
| R6 | Cycle rule misclassifies follow-ups (`HE_FailExt` and hearings can stretch beyond 60 days) | ✅ 99% of `HE_Fail` are openers, 3% of escalations; insensitive to the cut-off | nb 01, assert |
| R7 | 311 complaint match over-flags dense areas (food halls, downtown) or misses ISD-direct complaints | ✅ Confirmed for radius rules; replaced by one-complaint-one-visit attribution (6% of openers). Complaints made directly to ISD remain invisible | nb 01, assert |
| R8 | Junk values (`DATAERR`, `viol_level = 1919`, legacy `Pass`/`Fail` codes) | ✅ Every result code mapped to a role (fail / pass / filed / admin / closed) | nb 01, assert |
| R9 | One business, many licenses: ownership change creates a new `licenseno` (resetting history), or one site holds FS + RF licenses (duplicates in the pool) | ✅ `property_id` is a building or parcel (half of shared ones overlap in time), so it is too coarse to link licenses; each license stands alone. Predecessor linking by address is in the backlog | nb 01 |
| **Labels and metric** | | | |
| R10 | Base rate too high, so the metric has a ceiling | ✅ Avoided by the target choice: Priority violation (about 22%) instead of fail (about 57%); the oracle can about double the first-half hit rate; headroom share reported | nb 01–02 |
| R11 | Monthly pools too small (especially RF) for stable estimates | ✅ About 470 visits a month pooled (RF about 62); one year gives about 3.7-point CIs, so models are compared on the 36-month backtest; RF reported separately | nb 02–03 |
| R12 | **Drift**: the label definition changed in 2019; policy or staffing changes; the pandemic period | ✅ Labels start in 2020 (hit rate 19–25% per year since); model calibration is good on the backtest; retrain on 2020–2024 for the test run | nb 01, 03, 05 |
| R13 | Violation code system changes around 2018, breaking violation-group features | ✅ Keyword map on `violdesc` covers 96% (2020+) and 99.7% (earlier) of rows | nb 03, assert coverage |
| R14 | Group-level encodings (neighborhood, owner fail rate) computed on all years leak the future | ✅ Everything is computed as of decision time; truncation test (features identical when all later data is deleted: 0 mismatches) | nb 03, assert |
| **Text** | | | |
| R15 | Follow-up rows repeat earlier comments; text of the *current* visit leaks the answer | De-duplicate text per inspection; assert every text source date < *t* | nb 04, assert |
| R16 | Text over 512 tokens is silently truncated | Token length distribution; choose N past visits / truncation side on purpose | nb 04 |
| R17 | The model learns inspector writing style, or the 2018 wording shift, instead of risk | Performance by year and by era; strip Food Code tags such as `(P)` | nb 04 |
| R18 | **Personal data in a public repo**: owner and contact names; names inside free-text comments | ✅ Names dropped / hashed at load (asserted). Still to do in Phase 2: scan any comment shown in outputs or the README for names | nb 01 onward, assert columns absent |
| **Compute and environment** | | | |
| R19 | System Python is 3.9; the libraries need ≥ 3.10 | ✅ Resolved Nov 10: `.venv` on Homebrew Python 3.14.10, kernel "Python 3.14 (boston-inspections)", `libomp` installed for LightGBM. Versions printed in each notebook's first cell | setup |
| R20 | CSV is large (likely several hundred MB; about 2–3 GB in pandas) | ✅ 388 MB file, 516 MB in memory; read once as text, saved as Parquet | nb 01 |
| R21 | Fine-tuning is slow on a laptop | Apple-Silicon `mps` is confirmed working (torch 2.15); time one epoch on a subsample first; Colab or frozen embeddings as a fallback | nb 04 |
| R22 | Weather source needs a NOAA token | Downloaded without a token, but **not used**: visit-day weather isn't known when the monthly list is made (R25) | — |
| R23 | Results not reproducible (seeds, hidden notebook state) | Fixed `SEED`; Restart & Run All before every handoff; manifest SHA-256 printed | every nb |
| R24 | `issdttm` is not the original issue date (51% of licenses issued more than 30 days *after* their first inspection) | ✅ Not used; experience = days since the license's first recorded visit | nb 01 |
| R25 | **Decision-time leakage**: features measured at the visit date (days since last visit, visit-day weather) encode the visit order being evaluated | ✅ All features as of the first day of the visit's month; truncation test (features identical when later data is deleted) and timestamp test | nb 02–03, assert |

## Not using (and why)
| Data | Reason |
|---|---|
| 311 complaint photos | Zero photos on food or rodent complaints, 2023–2026 |
| 311 complaint text | Only the inspector's closing note, written after the visit (leaks the answer) |
| Online reviews (Yelp / Google) | No legitimate bulk access for Boston |
| Street View / storefront images | Weak link to kitchen hygiene; terms-of-service limits |
| Visit-day weather | Not known when the monthly list is made (R25) |

## Backlog
| Item | Effort | Note |
|---|---|---|
| Replicate in a second city (NYC or Chicago) to support "Boston is a pilot for the US" | 3–5 days per city | No national dataset. Until done, say only "applies to any jurisdiction with digitized inspections" |
| Link new licenses to their predecessor at the same address | ~1 day | Reduces history resets after ownership changes; `property_id` is too coarse (TECHNICAL.md §2) |
| Online review text as a third modality | Blocked | Only if a licensed dataset turns up |
| Inspector-level effects | Needs city data | Inspector ID is not public |
| Mobile food (MFW) | ~1 day | Too few inspections today |

## Decisions
| Date | Decision |
|---|---|
| 2026-11-10 | Same number of inspections; output is a priority list; value for government workforce + public safety |
| 2026-11-10 | Scope FS + FT + RF; MFW out |
| 2026-11-10 | Unit = routine cycle-opening inspection; follow-ups and complaint-driven visits excluded |
| 2026-11-10 | Primary metric HR@50 (first-half hit rate) vs. current practice, the best rule and the oracle; month-level bootstrap CIs (TECHNICAL.md §4) |
| 2026-11-10 | Modalities = tabular + inspector-comment text only |
| 2026-11-10 | Phased: tabular first, then deep learning on text (originally gated on the Phase 1 exit criterion; with the bar narrowly missed, the gate is now the tuning round and the go / no-go in step 4 of "What's next") |
| 2026-11-10 | Work in notebooks (one per step, markdown + code + checks); migrate to `src/` only in Phase 3 |
| 2026-11-10 | Cycle rule changed from a 60-day gap to a chain rule |
| 2026-11-10 | **Hit = Priority (`***`) violation at a routine visit, labelled from 2020** (nb 01) |
| 2026-11-10 | Decision time = first day of the month; weather dropped (nb 02–03) |
| 2026-11-10 | The bar is **Rule 1 (previous inspection)**, not Rule 2: the points checklist is weaker (nb 02) |
| 2026-11-10 | Phase 1 model = **logistic regression** (simplest within noise of best): +2.9 hits per 100 visits vs Rule 1 [+2.3, +3.4], 57% of headroom vs 44%; exit bar of +3 **narrowly not met** (nb 03) |
| 2026-11-10 | Complaint-driven and irregular visits stay **out of scope**: irregular visits are few (about 8 a month), with an unknown cause and an average hit rate. Complaint triage is a different decision (response urgency) with incomplete data |
| 2026-11-10 | Problem statement written into README.md |
| 2026-11-10 | Documentation split: README (public, high-level) · TECHNICAL.md (methodology, living) · PLAN.md (working) · CONTRIBUTING.md (how to work) |
| 2026-11-10 | Two-week replay queued for notebook 03, after the review of notebooks 02–03 |
| 2026-11-10 | Comparators renamed: B0 → current practice (the city), B1 → **Rule 1 · previous inspection**, B2 → **Rule 2 · points checklist**. Both rules are our no-ML proposals |
| 2026-11-10 | Tuning round queued: judgment calls split into *tune* (model, features), *test* (definitions, evaluation design) and *fixed by design* (exit bar, Rule 2 weights) |
| 2026-11-10 | Rule 2 weights stay hand-set (no tuning); explained in nb 02 §4 with the hit rate per score level |
| 2026-11-10 | README gets an approach-by-phase summary, Phase 1 results and Phase 2 expectations; TECHNICAL.md stays methodology-only |
| 2026-11-10 | GitHub repository created as **private**; can be made public once the work is final |
