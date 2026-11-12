"""Produce a month's priority list: the routine inspections due that month, highest risk first.

    python -m inspections.rank --month 2026-12 --due due.csv     # licenses the city has due (column: licenseno)
    python -m inspections.rank --month 2026-10                   # replay: the routine visits actually made that month
    python -m inspections.rank --month 2026-12 --due due.csv --with-text

The model is refit on every labelled routine visit from 2020 up to the end of the previous month,
and only data from before the first day of the month is used, as in the evaluation (TECHNICAL.md
§6). By default it is model (b), the simpler option with no detectable loss on the final test;
`--with-text` adds the inspector-comment score (the pre-declared primary model of notebook 05).

The list carries, for each place, the facts an inspector can check: the result of its last routine
inspection and the comments written there, masked for personal data. Scores rank places within the
month; as probabilities they drift with the city-wide rate, so the model is refit each run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inspections import config, features, models, prepare, text

LABEL_START = config.TRAIN_START


def routine_rows(visits):
    """Routine, scheduled visits: cycle openers that are neither irregular nor complaint-driven."""
    return visits[visits.opener & ~visits.irregular & ~visits.complaint_driven]


def priority_list(month, tables, due=None, with_text=False, n_comments=3):
    """The ranked list for `month` ('YYYY-MM'), as a DataFrame (rank 1 = visit first).

    `due`: license numbers the city has due that month; if None, the routine visits recorded in that month."""
    t0 = pd.Timestamp(f'{month}-01')
    visits, violations, licenses = tables['visits'], tables['violations'], tables['licenses']
    naive = visits.visit_ts.dt.tz_localize(None)
    past = visits[naive < t0]                                       # nothing from the month itself, or later

    routine = routine_rows(visits)
    train = routine[(routine.visit_ts.dt.tz_localize(None) >= LABEL_START) & (routine.visit_ts.dt.tz_localize(None) < t0)]
    if due is None:
        due = routine.loc[routine.month == month, 'licenseno'].drop_duplicates()
        if due.empty:
            raise ValueError(f'no routine visits recorded in {month}; pass the due list with --due')
    due = pd.Series(pd.unique(pd.Series(due, dtype=str)), name='licenseno')

    keys = pd.concat([features.decision_keys(train), pd.DataFrame({'licenseno': due, 't0': t0})], ignore_index=True)
    is_train = np.r_[np.ones(len(train), bool), np.zeros(len(due), bool)]
    y = np.r_[train.hit.to_numpy(), np.zeros(len(due), bool)]      # due rows carry no label; only train rows are fit
    groups = keys.licenseno.to_numpy()

    past_viol = violations[violations.visit_ts.dt.tz_localize(None) < t0]
    group_counts = features.violation_group_counts(past_viol)
    X, lr = features.build_model_b(keys, past, group_counts, licenses, with_asof=True)
    com = text.clean_comments(past_viol)
    windows = text.text_windows(keys, past, com, models.N_CYCLES)
    text_inputs = windows.comments.map(' '.join) if with_text else None
    scores = models.fit_and_score(X, y, is_train, ~is_train, text=text_inputs, groups=groups)

    last1 = text.text_windows(keys[~is_train], past, com, 1).comments
    scan = text.NameScan(com.text)
    d = keys[~is_train].copy()
    d['risk_score'] = scores['with_text' if with_text else 'model_b']
    d['category'] = d.licenseno.map(licenses.category)
    d['last_routine_date'] = lr.loc[d.index, 'routine_ts'].dt.date
    d['last_routine_priority_violation'] = lr.loc[d.index, 'severe'].map({True: 'yes', False: 'no'})
    d['last_routine_failed'] = lr.loc[d.index, 'is_fail'].map({True: 'yes', False: 'no'})
    d['days_since_last_visit'] = X.loc[d.index, 'days_since_last_visit']
    d['last_inspection_comments'] = last1.map(lambda cs: ' | '.join(scan.snippet(c, 120) for c in cs[:n_comments]))
    d['tiebreak'] = np.random.default_rng(config.SEED).random(len(d))
    d = d.sort_values(['risk_score', 'tiebreak'], ascending=[False, True]).drop(columns=['t0', 'tiebreak'])
    d.insert(0, 'rank', np.arange(1, len(d) + 1))
    assert scan.ok(d.last_inspection_comments), 'a displayed comment failed the personal-data scan'
    return d.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--month', required=True, help='the month to plan, YYYY-MM')
    parser.add_argument('--due', help='CSV with a licenseno column: the licenses due that month')
    parser.add_argument('--with-text', action='store_true', help='add the inspector-comment score (model (b) + text)')
    parser.add_argument('--out', help='output CSV (default: data/outputs/priority_list_<month>.csv)')
    args = parser.parse_args()

    due = pd.read_csv(args.due, dtype={'licenseno': str}).licenseno if args.due else None
    tables = prepare.load_processed(('visits', 'violations', 'licenses'))
    ranked = priority_list(args.month, tables, due, args.with_text)
    out = config.ROOT / 'data' / 'outputs' / f'priority_list_{args.month}.csv' if args.out is None else args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)
    half = len(ranked) // 2
    print(f'{args.month}: {len(ranked):,} routine inspections ranked ({"model (b) + text" if args.with_text else "model (b)"}); '
          f'visit the first {half:,} in the first half of the month')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
