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
- **Phase 1 (tabular) done and reviewed:** notebooks 01, 02, 03 and 03b.
  - **Adopted model (b):** a simplified logistic regression on 33 features.
  - **On the clean year 2024:** 34.8% first-half hit rate vs 32.4% for Rule 1, i.e. +2.4 hits per
    100 visits [+1.5, +3.5]; hits found 4.9 vs 3.4 days earlier. The gain holds on two-week blocks
    (+2.5).
  - **Exit bar (+3) narrowly not met.** Neither complex models nor tuning moved it.
  - TECHNICAL.md §4 and §7; README results.
- **Phase 2 done. Notebook 05 (one-time test) done and reviewed (2026-11-11).**
  - **Test period Jan 2025 – Oct 2026, scored once:** primary model (b) + text 33.3% first-half hit
    rate vs Rule 1 30.3%: **+3.0 per 100 [+2.3, +3.7], exit bar met at the threshold**. Model (b)
    alone +2.8 [+2.2, +3.4]. Text vs model (b) +0.3 [−0.1, +0.7], not detectable.
  - Holds in 2025 and 2026, for every licence type and with two-week lists. 2026 under-predicted by
    about 4 points (calibration drift; ranking unaffected).
  - **Impact:** about 300 more Priority violations a year found in the first half than today (about
    90 more than Rule 1), about 4.9 days sooner each.
  - **Fairness:** no neighbourhood or income group clearly worse off than today; gains largest in
    lower-income areas, where hazards are more common; local-strictness features barely change the
    allocation.
  - TECHNICAL.md §7 "The one-time test"; README headline, chart and implications now from the test period.
- **Notebook 04 (text experiments) done and reviewed (2026-11-11).**
  - **Selected:** model (b) + a TF-IDF text score (2 cycles of earlier comments). PR-AUC improves in
    every year (2024: 0.423 → 0.432).
  - **The headline is unchanged:** 34.8% first-half hit rate on 2024, +0.0 [−0.3, +0.4] vs model
    (b). Still +2.5 vs Rule 1, so the bar is not met. The gain is in the first 10% of the list
    (about one extra hit a month).
  - **Ladder:** E1 TF-IDF kept · E2 frozen embeddings and E3 fine-tuned DistilBERT add nothing →
    stop rule met; E4 not run.
  - **Decided:** D1 model (b) + TF-IDF primary, model (b) secondary for the test · D2 transformer
    work paused · D3 README, TECHNICAL.md and this file updated.
  - **D4 approved:** notebook 05 agreed and run (2026-11-11).
- **Documentation** is consistent with the latest run:
  - README: problem statement, approach, final test results with the chart from the test period,
    how we got there (Phases 1–2), implications and next steps;
  - TECHNICAL.md: methodology, living document;
  - CONTRIBUTING.md: how to work;
  - this file.
- **Repository:** [linyutian/boston-restaurant-inspections](https://github.com/linyutian/boston-restaurant-inspections)
  (private), with collaborators Jess-Bezof and carlarus99 invited (write access).
- **Open, not blocking Phase 2:** branch protection on `main` needs GitHub Pro for a private repo.
  Options: free Pro via the GitHub Student Developer Pack (recommended), paid Pro, going public, or a
  soft local guard. Waiting on your choice.

**What's next, in order**
1. ✅ Review of notebooks 01–03 (2026-11-11).
2. ✅ Tuning round, Tier A (nb 03b §1–§7). No gain from tuning (−0.2 [−0.6, +0.2] on 2024).
   **Model (b) adopted.** Tiers B and C not run (Judgment calls; backlog).
3. ✅ Two-week replay (nb 03b §8). Gain over Rule 1 unchanged (+2.5 vs +2.4 on 2024); "days
   earlier" halves by construction.
4. ✅ **Phase 2 go / no-go: GO** (2026-11-11).
5. ✅ **Phase 2:** notebook 04 (text experiments) and notebook 05 (one-time test, impact,
   fairness), both reviewed; README updated with the test results.
6. **Phase 3:** consolidate code into `src/inspections/` with tests, pin dependencies, and complete
   the README.

## How we work
1. **Phases 1 and 2 happen in Jupyter notebooks**, one per step, each reading top to bottom:
   purpose → short markdown / code / output units → `assert` checks for the risk register →
   findings and decisions.
2. **I run each notebook top to bottom** before handing it over, so saved outputs are real and in
   order. I stop at its open decisions for your review.
3. **Helpers start inside the notebook.** Stable code moves into `src/inspections/` in Phase 3.
4. **Documentation:** settled reasoning goes in TECHNICAL.md (a living document, extended as you
   review); this file keeps one-line decisions; the README stays high-level.
5. **Commits:** one per reviewed notebook or change set, made on a short-lived branch and fast-forwarded into `main`.
6. **Environment:** project `.venv` (Homebrew Python 3.14) with all dependencies, kernel
   "Python 3.14 (boston-inspections)".

| # | Notebook | Question | Status |
|---|---|---|---|
| 01 | `01_data_and_labels` | What is one inspection, a routine visit, a hit? | ✅ done · reviewed |
| 02 | `02_metric_and_baselines` | How good are current practice and simple rules, and what is the ceiling? | ✅ done · reviewed |
| 03 | `03_tabular_models` | Does ML on tabular data beat the best rule? | ✅ done · reviewed |
| 03b | `03b_tuning_round` | Does tuning close the gap to the +3 bar? Do results hold on two-week blocks? | ✅ done · no gain from tuning; model (b) adopted; gain holds on two-week blocks |
| 04 | `04_text_models` | Does inspector-comment text add value? | ✅ done · reviewed; TF-IDF kept (ranking gain, headline unchanged); stop rule met; transformer paused |
| 05 | `05_final_test` | One-time test-period evaluation, impact, fairness check | ✅ done · reviewed; exit bar met at the threshold (+3.0 [+2.3, +3.7]); text not detectable; no group worse off than today |

## Phases
**Phase 1, tabular (done).** Exit criterion: beat Rule 1 by ≥ 3 hits per 100 visits with a CI
excluding zero.
- Outcome on the pooled 2022–24 backtest: +2.9 [+2.3, +3.4] (notebook-03 model).
- After tuning, judged on the clean year 2024: +2.4 [+1.5, +3.5] (adopted model b).
- Narrowly not met. Phase 2 goes ahead anyway (decision 2026-11-11).

**Phase 2, inspector-comment text (nb 04–05): done and reviewed.** Experiments run
cheapest first; each is kept only if it beats the current best:
1. TF-IDF + logistic regression
2. Frozen sentence embeddings as extra features
3. Fine-tuned small transformer (DistilBERT / ModernBERT); its out-of-fold score feeds the
   tabular model
4. Tuning rounds (past visits included, text length, model, learning rate)

Stop after two experiments in a row add nothing. Then calibrate, retrain on 2020–2024, and run the
test period once (nb 05), including impact numbers and a fairness check by neighbourhood.
- Outcome (nb 04): E1 kept (+0.008 PR-AUC on folds 1–2); E2 −0.001 and E3 +0.000 on top of E1 →
  stopped; E4 not run. On 2024: +0.0 per 100 visits vs model (b), +2.5 [+1.6, +3.6] vs Rule 1.
- Outcome (nb 05, test period): primary +3.0 [+2.3, +3.7] vs Rule 1 (exit bar met at the
  threshold); secondary +2.8 [+2.2, +3.4]; text +0.3 [−0.1, +0.7] vs model (b).

**Phase 2 pick-up list** (flags and queued items collected during Phase 1):

| # | Item | Where it came from | What Phase 2 does |
|---|---|---|---|
| P1 | **Baseline = model (b)** (33 features); bar = +3 hits per 100 visits vs Rule 1 | nb 03b §7 | Every text experiment is added on top of model (b) and compared with it *and* with Rule 1 |
| P2 | **Evaluation protocol:** choose on 2022–23 folds, judge on 2024 (clean), test period once at the end | nb 03b guardrails; CONTRIBUTING rule 2 | Same protocol in nb 04; the 2025–26 test only in nb 05 |
| P3 | **Pre-check the text signal formally** | README Phase 2 expectations ("8% vs 30%", mice / "repeat" keywords) came from a quick check outside the notebooks | First section of nb 04 reproduces it, and the README is corrected if it doesn't hold |
| P4 | **Text leakage and de-duplication (R15):** follow-ups repeat earlier comments; the current visit's text gives the answer away | Risk register | Use only comments from visits **before the first day of the month**; de-duplicate; assert it |
| P5 | **Truncation (R16)** | Risk register | Check token-length distribution; choose how many past visits to include and which end to truncate |
| P6 | **Inspector style and the 2019 wording change (R17)** | Risk register; local-strictness finding (TECHNICAL.md §8) | Strip Food Code tags such as `(P)`; check performance by year; watch for the text acting as a proxy for *which inspector* rather than *which kitchen* |
| P7 | **Personal data in comment text (R18)** | Risk register | Scan any comment shown in a notebook output or the README for names before it is displayed |
| P8 | **Violation-type information returns through text** | nb 03b A4 dropped the violation-mix group | Expect overlap; report text gains *on top of* model (b), which no longer has violation types |
| P9 | **Compute choice (R21)** | Compute comparison (2026-11-11) | Before fine-tuning, time one epoch on a subsample on this Mac (MPS) and on Colab (T4); pick the faster |
| P10 | **Explanations** | README Phase 2 expectations | Even if accuracy gains are small, test comment snippets as "why this place is on the list" explanations |
| P11 | **Fairness check by neighbourhood** | Local-strictness signal; limitations | nb 05: flag rates and hit rates by neighbourhood and income tract |
| P12 | **Days-earlier depends on the planning period** | nb 03b §8 | Report per-visit lift and days earlier separately; note what a twice-monthly list would add |

Status after nb 04:
- **Done in nb 04:** P1–P10 and P12.
  - P3: the raw claims hold, but they largely overlap with model (b) (§3).
  - P6: Food Code tags are in `violdesc` only. Inspector emails and phone numbers were masked. No
    sign of an inspector proxy, but it can't be ruled out (§9).
  - P9: DistilBERT run locally. Colab not timed (needs an interactive session).
  - P10: model-picked snippets are only partly meaningful. Show the last inspection's comments
    instead (§10).
- **Done in nb 05:** P11 (fairness by neighbourhood and ZIP income) and P12 (two-week lists on the
  test period: +3.2 vs Rule 1; days earlier halves).

**Phase 3, consolidate and publish.** Move stable code into `src/inspections/` with tests, pin
dependencies, complete the README (final results, implications and next steps).

## Judgment calls: tune or test
Every value set by judgment or heuristic rather than learned. The split matters:
- **Tune:** values that only change *how well we predict*. Tune them on the backtest: folds 1–2
  (2022–23) for selection, 2024 as the clean check, the test period never.
- **Test:** values that define *what* is predicted or *how* it is judged. Tuning them for a higher
  score would move the goalposts, so we only check that conclusions hold when they change.
- **Fixed by design:** values that must stay fixed for a comparison to mean anything.

Scope run so far: **Tier A only** (agreed 2026-11-11). Outcomes are in the last column.

| Judgment call | Current value | Where | Action | Search range / check · **outcome** |
|---|---|---|---|---|
| **Model hyperparameters** | | | | |
| Logistic regression strength | `C = 0.1`, L2, *never tuned* | nb 03 E1 | **tune** | C ∈ {0.001 … 10}; L1 vs L2 · **A1: flat (all L2 within 0.002); C = 0.003 chosen, no gain on 2024** |
| LightGBM hyperparameters | 16-trial random search | nb 03 E3 | **tune** | wider search (e.g. 60 trials) incl. `min_child_samples`, `reg_alpha`, `subsample` · *Tier B, not run* |
| Class weighting | none | nb 03 | **tune** | balanced vs none (affects calibration more than ranking) · *Tier B, not run* |
| **Feature settings** | | | | |
| Local-rate radius and windows | 1 km; 90 / 180 days | nb 03 area features | **tune** | 0.5 / 1 / 2 km; 90 / 180 / 365 days · **A2: 1 km, 90 + 180 days (unchanged) is best; 2 km clearly worse** |
| 311 radius and windows | 100 m; 90 / 365 days | nb 03 | **tune** | 50 / 100 / 250 m; or drop the group · **A4: group dropped (no loss)** |
| Recency weight of the EWMA rates | α = 0.5 | nb 03 trend | **tune** | 0.2 / 0.35 / 0.5 / 0.7 · **A3: irrelevant (≤ 0.0001)** |
| "Last N routine visits" window | N = 3 | nb 03 trend | **tune** | 2 / 3 / 5 · **A3: N = 2 slightly better** |
| Violation keyword groups | 8 groups, hand-written rules | nb 03 §2.1 | **tune** (lightly) | merge or split groups · *Tier B, not run*; the violation-mix group itself was dropped in A4 (no loss) |
| Training window | 2020+ (E5 tried 2012+) | nb 03 | **tune** | 2019+ / 2020+ / 2021+ · *Tier B, not run* (notebook 03 E5 showed 2012+ on par) |
| **Definitions (what is predicted)** | | | | |
| Cycle gap for a new routine visit | 90 days | nb 01 §6 | **test** | already shown stable for 60–180 days · *Tier C, deferred* |
| Irregular cut-off | < 30 days after a pass | nb 01 §6 | **test** | 14 / 30 / 45 days · *Tier C, deferred* |
| Complaint attribution | nearest within 25 m, 0–30 days | nb 01 §7 | **test** | 15 / 25 / 50 m; 14 / 30 / 45 days · *Tier C, deferred* |
| Label start | 2020 | nb 01 §8 | **test** | fixed by the Food Code change; no tuning |
| **Evaluation design (how it is judged)** | | | | |
| First-half cut | 50% | nb 02 | **test** | capture curve already reported at 5–100% |
| Planning period | monthly | nb 02 | **test** | two-week replay · **done (03b §8): per-visit gain unchanged (+2.5 vs +2.4); days earlier halves** |
| Exit bar | +3 hits per 100 visits | nb 02 | **fixed by design** | changing it after seeing results would defeat its purpose |
| **Comparators** | | | | |
| Rule 2 (points checklist) weights | +2 / +1 / +1 / +1 | nb 02 §4 | **fixed by design** | it represents a no-data checklist. Optional extra comparator: a *learned* scorecard (logistic regression on the same four items) |

## Known weak points
1. **Gains over the best rule are modest** (about +3 per 100 visits; +3.0 [+2.3, +3.7] on the
   test period). The rule itself is a valid recommendation and delivers 60–70% of the benefit.
2. **Stability over up to two weeks is assumed** by the evaluation. The two-week replay showed the
   gains don't depend on month-long stability; only a live pilot removes the assumption entirely.
3. **Part of the signal is the inspector, not the kitchen** (local strictness), which raises a
   communication question. The nb 05 fairness check found it barely changes where early visits go.
4. **The text phase added little** (nb 04): a consistent ranking gain at the very top of the list,
   no change to the first-half hit rate.

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
| R12 | **Drift**: the label definition changed in 2019; policy or staffing changes; the pandemic period | ✅ Labels start in 2020 (hit rate 19–25% per year since); model calibration is good on the backtest; retrained on 2020–2024 for the test run. Test: 2025 calibrated, 2026 under-predicted by about 4 points (base rate 25%); ranking unaffected. Last test month shows no sign of incomplete recording | nb 01, 03, 05 |
| R13 | Violation code system changes around 2018, breaking violation-group features | ✅ Keyword map on `violdesc` covers 96% (2020+) and 99.7% (earlier) of rows | nb 03, assert coverage |
| R14 | Group-level encodings (neighborhood, owner fail rate) computed on all years leak the future | ✅ Everything is computed as of decision time; truncation test (features identical when all later data is deleted: 0 mismatches) | nb 03, assert |
| **Text** | | | |
| R15 | Follow-up rows repeat earlier comments; text of the *current* visit leaks the answer | ✅ Text only from visits before *t0*, de-duplicated within each row (pass visits repeat 98% of comments); truncation test 0 mismatches; < 0.1% of comments mention a later date (deadlines) | nb 04, assert |
| R16 | Text over 512 tokens is silently truncated | ✅ 2-cycle text: median 125 tokens, 5% over 512. TF-IDF and embeddings see all of it; DistilBERT reads 256 tokens, newest comments first | nb 04 |
| R17 | The model learns inspector writing style, or the 2018 wording shift, instead of risk | ✅ Tags are only in `violdesc` (not used); inspector emails and phones masked; text gain stable 2022–24, correlation with local strictness 0.26, gain not concentrated in strict areas. Some phrases look like inspector habits; can't be ruled out without inspector IDs | nb 04 |
| R18 | **Personal data in a public repo**: owner and contact names; names inside free-text comments | ✅ Names dropped / hashed at load (asserted). nb 04: emails, phones, titled and role-named persons masked before modelling; every displayed term and snippet passes a name scan (asserted) | nb 01 onward, assert |
| **Compute and environment** | | | |
| R19 | System Python is 3.9; the libraries need ≥ 3.10 | ✅ Resolved Nov 10: `.venv` on Homebrew Python 3.14.10, kernel "Python 3.14 (boston-inspections)", `libomp` installed for LightGBM. Versions printed in each notebook's first cell | setup |
| R20 | CSV is large (likely several hundred MB; about 2–3 GB in pandas) | ✅ 388 MB file, 516 MB in memory; read once as text, saved as Parquet | nb 01 |
| R21 | Fine-tuning is slow on a laptop | ✅ Timed first: DistilBERT ≈ 1.3 h estimate for the E3 backtest on `mps`, ModernBERT ≈ 3.8 h. The real run took 9.3 h wall-clock (machine sleep/throttling; the largest fold matched the estimate). Use `caffeinate -i` or a GPU next time | nb 04 |
| R22 | Weather source needs a NOAA token | Downloaded without a token, but **not used**: visit-day weather isn't known when the monthly list is made (R25) | — |
| R23 | Results not reproducible (seeds, hidden notebook state) | ✅ Fixed `SEED` (including solver seeds; an unseeded L1 solver was found and pinned in nb 03b); Restart & Run All before every handoff; nb 03b reproduces nb 03 exactly and gives byte-identical logs across runs | every nb |
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
| Tuning Tier B: wider LightGBM search, class weighting, training window, keyword groups | ~1.5 h | Not run (agreed scope was Tier A); low expected value given the flat Tier A results |
| Tuning Tier C: sensitivity of the definitions (cycle gap, irregular cut-off, complaint matching) | 2–3 h | Deferred; robustness rather than performance. Notebook 01 already showed the prediction set barely moves |
| Complaint triage with the same score | ~½ day | Out of scope (2026-11-10); only if the city wants complaint urgency ranked |

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
| 2026-11-11 | Review of notebooks 01–03 complete |
| 2026-11-11 | Tuning round scoped to Tier A; exit verdict judged on the clean year 2024; run in a separate notebook (03b) |
| 2026-11-11 | Tuning round result: no gain from tuning (−0.2 [−0.6, +0.2] on 2024); simplified 33-feature model equally good; exit bar not met (+2.4 on 2024) |
| 2026-11-11 | README capture-curve chart added (generated by notebook 03 §5) |
| 2026-11-11 | **Phase 1 model = model (b):** simplified 33-feature logistic regression (C = 0.003, L2; 1 km, 90 + 180 days; last N = 2; 311, owner, violation-mix and calendar groups dropped). Replaces the 55-feature notebook-03 model; logged in nb 03b §7 |
| 2026-11-11 | Two-week replay appended to nb 03b (§8) rather than a new notebook. Result: gain over Rule 1 unchanged (+2.5 vs +2.4 on 2024); the within-month stability concern is bounded at two weeks |
| 2026-11-11 | **Phase 2: GO.** Inspector-comment experiments approved; not started until instructed. Pick-up list P1–P12 in Phases |
| 2026-11-11 | Docs updated to model (b): README Phase 1 results now report 2024 (clean year); the README chart is drawn from model (b) in nb 03b §9 (moved from nb 03 §5, so re-running nb 03 cannot overwrite it) |
| 2026-11-11 | Phase 2 started on `main` (author's contributions go directly to `main`; workflow wording unchanged) |
| 2026-11-11 | nb 04 run (pending review): text enters by stacking (out-of-fold text score as one extra input to model (b)); E1 TF-IDF kept, E2 and E3 add nothing, stop rule met. Decisions D1–D4 open |
| 2026-11-11 | nb 04 reviewed. D1: model (b) + TF-IDF primary, model (b) pre-declared secondary for nb 05. D2: transformer work paused (E4 not run). D3: README, TECHNICAL.md, PLAN.md updated with the findings. D4: nb 05 approved after its effort estimate; paused until the author says go |
| 2026-11-11 | nb 05 go. Census income data approved for the fairness check (ACS 2020–2024 by ZIP, public summary file; added to `inspections.download --only census`) |
| 2026-11-11 | **Test period scored once** (nb 05, pending review): primary +3.0 [+2.3, +3.7] vs Rule 1, exit bar met at the threshold; secondary +2.8; text not detectable; no group clearly worse off than today. Fairness flag rules (relative calibration gap, month-bootstrap CI) fixed on the 2024 rehearsal before scoring |
| 2026-11-11 | nb 05 reviewed. README reports the primary model (declared before scoring) and names model (b) alone as the simpler practical option; README headline, chart and implications moved to the test period |
