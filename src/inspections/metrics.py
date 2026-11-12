"""The ranking metric (TECHNICAL.md §4).

Every function takes an evaluation frame with one row per routine visit and the columns
`month` (the list the visit belongs to), `hit` (bool), `visit_ts` (actual visit time), `tiebreak`
(a fixed random number, so equal scores are ordered reproducibly) and one column per ranking score,
where a higher score means "visit earlier".
"""
import numpy as np
import pandas as pd

from inspections import config


def _ranked(g, score):
    return g.sort_values([score, 'tiebreak'], ascending=[False, True])


def per_month(df, score, q=0.5):
    """Hits and visits in the top-q share of each month, ranking by `score` (higher = visit earlier)."""
    rows = []
    for month, g in df.groupby('month', sort=True):
        k = int(len(g) * q)
        top = _ranked(g, score).head(k)
        rows.append((month, top.hit.sum(), k, g.hit.sum(), len(g)))
    return pd.DataFrame(rows, columns=['month', 'hits', 'visits', 'month_hits', 'month_visits']).set_index('month')


def hit_rate(pm):
    """Pooled hit rate of a `per_month` table: the headline HR@50 when q = 0.5."""
    return pm.hits.sum() / pm.visits.sum()


def bootstrap_diff(pm_a, pm_b, n=2000, seed=config.SEED):
    """95% CI for HR(a) − HR(b), resampling months (the two tables share months)."""
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(pm_a), size=(n, len(pm_a)))
    ha, va, hb, vb = pm_a.hits.to_numpy(), pm_a.visits.to_numpy(), pm_b.hits.to_numpy(), pm_b.visits.to_numpy()
    return np.percentile(ha[idx].sum(1) / va[idx].sum(1) - hb[idx].sum(1) / vb[idx].sum(1), [2.5, 97.5])


def days_earlier(df, score):
    """Mean days by which hits are visited earlier when the month's actual dates are re-assigned in `score` order."""
    shifts = []
    for _, g in df.groupby('month'):
        ts = g.visit_ts.dt.tz_localize(None) if g.visit_ts.dt.tz is not None else g.visit_ts
        slots = np.sort(ts.to_numpy())
        new_date = pd.Series(slots, index=_ranked(g, score).index)
        hits = g.index[g.hit]
        shifts.append(((ts.loc[hits] - new_date.loc[hits]).dt.total_seconds() / 86400).to_numpy())
    return np.concatenate(shifts).mean()


def first_half(df, score):
    """Row-level version of per_month: True for the visits a method puts in the first half of their month."""
    flag = pd.Series(False, index=df.index)
    for _, g in df.groupby('month', sort=True):
        flag[_ranked(g, score).head(int(len(g) * 0.5)).index] = True
    return flag


def capture_curve(df, score, grid):
    """Share of each month's hits found after the first q of the month's visits, for each q in `grid` (pooled)."""
    out = []
    for q in grid:
        pm = per_month(df, score, q)
        out.append(pm.hits.sum() / pm.month_hits.sum())
    return np.array(out)
