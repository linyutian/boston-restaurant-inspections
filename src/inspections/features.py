"""Decision-time features (TECHNICAL.md §6).

Every feature for a row is computed as of t0, the first day of the visit's month, from data strictly
before t0. The adopted feature set is **model (b)** (33 features: history, recent trend, local
strictness; notebook 03b). The groups tested in Phase 1 and dropped (violation mix, owner, 311
complaints nearby, calendar and place) are kept so notebooks 03 and 03b still run from this module.

Inputs are the processed tables built by `inspections.prepare`: `visits`, `violations`, `licenses`
(indexed by `licenseno`) and `complaints`.
"""
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from inspections import config

EARTH_RADIUS_M = 6_371_000
ESCALATIONS = ['HE_FailExt', 'HE_Hearing', 'HE_TSOP', 'HE_Closure']

VIOLATION_GROUPS = {   # first matching group wins, so order matters
    'pests': r'pest|insect|rodent|animal|vermin',
    'temperature': r'temperature|holding|cooling|cooked|cooking|reheat|thaw|thermometer|time as a public health',
    'handwashing': r'hand|when to wash|hygien|hair|eating, drinking|tobacco|clean cloth|outer cloth|clothing|glove'
                   r'|location  accessible|number  convenient',
    'contamination': r'contamina|separation|segregat|food protection|raw|adulterat|unsafe|discard|storage|stored'
                     r'|container|label|date mark|wrapped|packag|ice|approved source|sources|shell ?stock|tags & records'
                     r'|consumer advis|allergen|toxic|poisonous|chemical|food preparation',
    'sanitizing': r'food contact|food-contact|sanit|warewash|dishwash|test kit|utensil|wiping|sponge|equipment|drain ?board',
    'management': r'pic|person-in-charge|person in charge|knowledge|certified|manager|employee health|illness|vomit'
                  r'|demonstration|duties|compliance|permit|license|posting|variance|haccp|plan|conformance|procedure'
                  r'|assignment|ceasing operations|anti-choking|reporting',
    'facility': r'floor|wall|ceiling|premises|repair|ventilation|light|fixture|shield|plumb|sewage|toilet|garbage'
                r'|refuse|dumpster|receptacle|grease|backflow|water|sink|surface|installed|designed|constructed'
                r'|nonfood|non-food|outside|outer opening|maintained|maintenance|linen|mop|dressing room|locker'
                r'|cleanab|cleaning',
}


def violation_group(description):
    """The violation-type group of a Food Code description (keyword map; 'other' if nothing matches)."""
    d = description.lower()
    return next((g for g, pattern in VIOLATION_GROUPS.items() if re.search(pattern, d)), 'other')


def violation_group_counts(violations):
    """Violations per group at each visit: one row per (licenseno, resultdttm), columns n_<group>."""
    group_of = {d: violation_group(d) for d in violations.violdesc.dropna().unique()}
    group = violations.violdesc.map(group_of).fillna('other')
    return (violations.assign(group=group)
            .pivot_table(index=['licenseno', 'resultdttm'], columns='group', values='violation', aggfunc='size', fill_value=0)
            .add_prefix('n_'))


def decision_keys(rows):
    """(licenseno, t0) for each row, t0 = the first day of the visit's month (naive Boston time)."""
    keys = rows[['licenseno']].copy()
    keys['t0'] = rows.visit_ts.dt.tz_localize(None).dt.to_period('M').dt.start_time
    return keys


def asof(keys, table, time_col, by):
    """For each key row (by, t0): the latest `table` row strictly before t0."""
    left = keys.reset_index().sort_values('t0')
    out = pd.merge_asof(left, table.sort_values(time_col), left_on='t0', right_on=time_col, by=by,
                        allow_exact_matches=False)
    return out.set_index('index').sort_index()


def history_tables(visits, group_counts, ewm_alpha=0.2, last_n=2):
    """Running per-license history: one table over all visits, one over routine (cycle-opening) visits.

    Model (b) uses ewm_alpha = 0.2 and last_n = 2; notebook 03 used 0.5 and 3."""
    group_cols = list(group_counts.columns)
    v = visits.sort_values(['licenseno', 'visit_ts']).reset_index(drop=True)
    v['ts'] = v.visit_ts.dt.tz_localize(None)
    v['is_fail'] = v.role.eq('fail')
    g = v.groupby('licenseno')
    v['cum_visits'] = g.cumcount() + 1
    v['cum_escalations'] = v.result.isin(ESCALATIONS).groupby(v.licenseno).cumsum()
    v['first_ts'] = g.ts.transform('min')
    v['last_fail_ts'] = v.ts.where(v.is_fail).groupby(v.licenseno).ffill()

    o = v[v.opener].join(group_counts, on=['licenseno', 'resultdttm'])
    o[group_cols] = o[group_cols].fillna(0)
    o['severe'] = o.n_severe.gt(0)
    o['old_code'] = o.ts.dt.year < 2020
    go = o.groupby('licenseno')
    o['cum_routine'] = go.cumcount() + 1
    o['cum_routine_fail'] = go.is_fail.cumsum()
    o['cum_routine_severe'] = go.severe.cumsum()
    o['prev2_fail'] = go.is_fail.shift()
    o['prev2_severe'] = go.severe.shift()
    for c in ['severe', 'is_fail', 'n_viol', 'n_temperature', 'n_contamination', 'n_pests']:
        o[f'last3_{c}'] = go[c].transform(lambda s: s.astype(float).rolling(last_n, min_periods=1).sum())
    o['ewm_severe'] = go.severe.transform(lambda s: s.astype(float).ewm(alpha=ewm_alpha).mean())
    o['ewm_fail'] = go.is_fail.transform(lambda s: s.astype(float).ewm(alpha=ewm_alpha).mean())
    o['new_code_routine'] = (~o.old_code).groupby(o.licenseno).cumsum()
    o['new_code_severe'] = (o.severe & ~o.old_code).groupby(o.licenseno).cumsum()
    return v, o


def history_features(keys, v, o, group_cols, violation_mix=False):
    """The history and recent-trend groups (plus the violation mix at the last routine visit if asked).

    Returns the features and the as-of routine-visit table."""
    lv = asof(keys, v[['licenseno', 'ts', 'role', 'cum_visits', 'cum_escalations', 'first_ts', 'last_fail_ts']],
              'ts', 'licenseno')
    rcols = ['licenseno', 'ts', 'is_fail', 'severe', 'old_code', 'n_viol', 'n_severe', 'cum_visits', 'cum_routine',
             'cum_routine_fail', 'cum_routine_severe', 'prev2_fail', 'prev2_severe', 'ewm_severe', 'ewm_fail',
             'new_code_routine', 'new_code_severe'] + list(group_cols) + [c for c in o.columns if c.startswith('last3_')]
    lr = asof(keys, o[rcols].rename(columns={'ts': 'routine_ts', 'cum_visits': 'cum_visits_at_routine'}),
              'routine_ts', 'licenseno')
    F = pd.DataFrame(index=keys.index)
    # history
    F['has_history'] = lv.ts.notna()
    F['prev_fail'] = lr.is_fail.astype(float)
    F['prev_severe'] = lr.severe.astype(float)
    F['prev_n_viol'] = lr.n_viol
    F['prev_n_severe'] = lr.n_severe
    F['prev_old_code'] = lr.old_code.astype(float)
    F['prev2_fail'] = lr.prev2_fail.astype(float)
    F['prev2_severe'] = lr.prev2_severe.astype(float)
    F['n_prior_routine'] = lr.cum_routine.fillna(0)
    F['prior_fail_rate'] = lr.cum_routine_fail / lr.cum_routine
    F['prior_severe_rate'] = lr.cum_routine_severe / lr.cum_routine
    F['followups_last_cycle'] = lv.cum_visits - lr.cum_visits_at_routine
    F['ever_escalated'] = lv.cum_escalations.gt(0).astype(float).where(lv.ts.notna())
    F['last_visit_filed'] = lv.role.eq('filed').astype(float).where(lv.ts.notna())
    F['days_since_last_visit'] = (keys.t0 - lv.ts).dt.days
    F['days_since_last_routine'] = (keys.t0 - lr.routine_ts).dt.days
    F['days_since_last_fail'] = (keys.t0 - lv.last_fail_ts).dt.days
    F['days_since_first_visit'] = (keys.t0 - lv.first_ts).dt.days
    if violation_mix:   # violation mix at the previous routine visit (dropped in notebook 03b)
        for c in group_cols:
            F[f'prev_{c}'] = lr[c]
    # recent trend
    for c in [c for c in lr.columns if c.startswith('last3_')]:
        F[c] = lr[c]
    F['ewm_severe'] = lr.ewm_severe
    F['ewm_fail'] = lr.ewm_fail
    F['new_code_severe_rate'] = lr.new_code_severe / lr.new_code_routine.replace(0, np.nan)
    return F, lr


def area_features(keys, o, licenses, radius_m=1000, windows=(90, 180), name='area_{name}_{days}d'):
    """Recent outcomes of routine visits to *any* establishment within radius_m: a proxy for local inspector strictness."""
    ref = o[['licenseno', 'ts', 'severe', 'is_fail']].join(licenses[['lat', 'lon']], on='licenseno').dropna(subset=['lat'])
    ref = ref[ref.ts >= keys.t0.min() - pd.Timedelta(days=max(windows))]   # older visits can never fall in a window
    xy = keys[['licenseno']].join(licenses[['lat', 'lon']], on='licenseno')
    ok = xy.lat.notna().to_numpy()
    tree = BallTree(np.radians(ref[['lat', 'lon']].to_numpy()), metric='haversine')
    near = tree.query_radius(np.radians(xy.loc[ok, ['lat', 'lon']].to_numpy()), r=radius_m / EARTH_RADIUS_M)
    rt, rs, rf = ref.ts.to_numpy(), ref.severe.to_numpy(), ref.is_fail.to_numpy()
    t0 = keys.t0.to_numpy()[ok]
    F = pd.DataFrame(index=keys.index)
    for days in windows:
        lo = t0 - np.timedelta64(days, 'D')
        sev, fail, n = (np.full(ok.sum(), np.nan) for _ in range(3))
        for k, i in enumerate(near):
            m = (rt[i] < t0[k]) & (rt[i] >= lo[k])
            n[k] = m.sum()
            if n[k]:
                sev[k], fail[k] = rs[i][m].mean(), rf[i][m].mean()
        for metric, arr in [('severe_rate', sev), ('fail_rate', fail), ('routine_visits', n)]:
            col = np.full(len(keys), np.nan)
            col[ok] = arr
            F[name.format(name=metric, days=days)] = col
    return F


def build_model_b(keys, visits, group_counts, licenses, with_asof=False):
    """Model (b)'s 33 features as of each row's t0 (optionally also the as-of routine-visit table)."""
    v, o = history_tables(visits, group_counts)
    hist, lr = history_features(keys, v, o, group_counts.columns)
    X = pd.concat([hist, area_features(keys, o, licenses)], axis=1).astype(float)
    return (X, lr) if with_asof else X


def rule1_score(X):
    """Rule 1: previous routine visit had a Priority violation (2) or failed (1); ties broken by longest since last visit."""
    return 2 * X.prev_severe.fillna(0) + X.prev_fail.fillna(0) + X.days_since_last_visit.fillna(9999) / 1e5


# --- groups tested in Phase 1 and dropped within noise (notebook 03b A4) ------------------------------------------

def owner_features(keys, o, lr, licenses):
    """Routine-visit record of the *same owner's other* licenses before t0."""
    own = o[['licenseno', 'ts', 'severe']].join(licenses['owner_id'], on='licenseno').dropna(subset=['owner_id'])
    own = own.sort_values('ts')
    g = own.groupby('owner_id')
    own['owner_routine'] = g.cumcount() + 1
    own['owner_severe'] = g.severe.cumsum()
    k = keys.join(licenses['owner_id'], on='licenseno')
    k = k[k.owner_id.notna()].drop(columns='licenseno')
    lo = asof(k, own[['owner_id', 'ts', 'owner_routine', 'owner_severe']], 'ts', 'owner_id').reindex(keys.index)
    mine_n = lr.cum_routine.fillna(0)
    mine_sev = lr.cum_routine_severe.fillna(0)
    F = pd.DataFrame(index=keys.index)
    F['owner_known'] = keys.licenseno.map(licenses.owner_id).notna().astype(float)
    F['owner_other_routine'] = (lo.owner_routine.fillna(0) - mine_n).clip(lower=0).where(F.owner_known == 1)
    F['owner_other_severe_rate'] = (lo.owner_severe.fillna(0) - mine_sev) / F.owner_other_routine.replace(0, np.nan)
    return F


def count_nearby(keys, points, point_ts, radius_m, windows, licenses):
    """Number of `points` within radius_m of each license, in each window of days before t0."""
    xy = keys[['licenseno']].join(licenses[['lat', 'lon']], on='licenseno')
    ok = xy.lat.notna().to_numpy()
    tree = BallTree(np.radians(points), metric='haversine')
    near = tree.query_radius(np.radians(xy.loc[ok, ['lat', 'lon']].to_numpy()), r=radius_m / EARTH_RADIUS_M)
    t0 = keys.t0.to_numpy()[ok]
    out = {}
    for days in windows:
        lo = t0 - np.timedelta64(days, 'D')
        counts = np.array([((point_ts[i] < t0[k]) & (point_ts[i] >= lo[k])).sum() for k, i in enumerate(near)], float)
        col = np.full(len(keys), np.nan)
        col[ok] = counts
        out[days] = col
    return out


def complaint_features(keys, complaints, licenses):
    """311 food complaints and rodent reports within 100 m in the 90 and 365 days before t0."""
    F = pd.DataFrame(index=keys.index)
    for name, mask in [('food_complaints', complaints.type.ne('Rodent Activity')),
                       ('rodent_reports', complaints.type.eq('Rodent Activity'))]:
        c = complaints[mask]
        counts = count_nearby(keys, c[['latitude', 'longitude']].to_numpy(), c.open_ts.dt.tz_localize(None).to_numpy(),
                              100, [90, 365], licenses)
        for days, col in counts.items():
            F[f'{name}_100m_{days}d'] = col
    return F


def calendar_place_features(keys, licenses):
    """License type, month of year, pandemic flag and coordinates."""
    F = pd.DataFrame(index=keys.index)
    cat = keys.licenseno.map(licenses.category)
    for c in ['FS', 'FT', 'RF']:
        F[f'cat_{c}'] = cat.eq(c).astype(float)
    F['month_of_year'] = keys.t0.dt.month
    F['pandemic'] = keys.t0.between(*config.PANDEMIC).astype(float)
    F['lat'] = keys.licenseno.map(licenses.lat)
    F['lon'] = keys.licenseno.map(licenses.lon)
    return F


def build_phase1_features(rows, visits, complaints, group_counts, licenses, ewm_alpha=0.5, last_n=3, area_radius_m=1000,
                          area_windows=(90, 180), area_name='area_{name}_{days}d'):
    """Notebook 03's full feature set (55 features), as of each row's t0, from the given source tables only."""
    keys = decision_keys(rows)
    v, o = history_tables(visits, group_counts, ewm_alpha, last_n)
    hist, lr = history_features(keys, v, o, group_counts.columns, violation_mix=True)
    parts = {'history+mix+trend': hist, 'owner': owner_features(keys, o, lr, licenses),
             '311 nearby': complaint_features(keys, complaints, licenses),
             'area recent rates': area_features(keys, o, licenses, area_radius_m, area_windows, area_name),
             'calendar & place': calendar_place_features(keys, licenses)}
    return pd.concat(parts.values(), axis=1).astype(float), keys.t0
