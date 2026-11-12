"""Build the processed tables from the raw snapshot (the rules derived in notebook 01).

    python -m inspections.prepare        # reads data/raw/ (via manifest.json), writes data/processed/

| Output | Contents |
|---|---|
| visits.parquet | every pass / filed / fail visit since 2006, with role, cycle flags, split and hit |
| violations.parquet | one row per cited violation: code, description, level, comment text |
| licenses.parquet | category, issue date, property id, hashed owner, zip, coordinates (indexed by licenseno) |
| complaints.parquet | 311 food complaints and rodent reports with coordinates |

Personal names are dropped (contact person) or hashed (legal owner) as the file is read, and the
snapshot-only license fields (status, expiry, business name) are never kept (R5, R18).
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from inspections import config

TZ = 'America/New_York'
EARTH_RADIUS_M = 6_371_000
SCOPE = ['FS', 'FT', 'RF']                  # sit-down, take-out, retail food; mobile food is out of scope
CYCLE_GAP = 90                              # days after which a visit opens a new cycle (notebook 01 §6)
IRREGULAR_GAP = 30                          # an opener this soon after a pass is not a scheduled visit
COMPLAINT_RADIUS_M, COMPLAINT_WINDOW_DAYS = 25, 30
FAIL_CODES = {'HE_Fail', 'HE_FailExt', 'HE_Hearing', 'HE_TSOP', 'HE_Closure', 'HE_FAILNOR', 'Fail', 'Failed'}
RESULT_ROLE = {
    **{c: 'fail' for c in FAIL_CODES},
    'HE_Pass': 'pass', 'Pass': 'pass', 'PassViol': 'pass', 'NoViol': 'pass',
    'HE_Filed': 'filed',
    'HE_NotReq': 'admin', 'HE_Misc': 'admin', 'HE_Hold': 'admin', 'DATAERR': 'admin',
    'HE_OutBus': 'closed', 'HE_VolClos': 'closed', 'Closed': 'closed',
}
FOOD_COMPLAINTS = ['Unsanitary Conditions - Food', 'Unsanitary Conditions - Establishment',
                   'Unsanitary Conditions - Employees', 'Food Alert - Confirmed', 'Food Alert - Unconfirmed']
PERSONAL = ['namefirst', 'namelast', 'legalowner']
VISIT_COLS = ['licenseno', 'resultdttm', 'visit_ts', 'month', 'category', 'result', 'role', 'n_viol', 'n_severe',
              'n_cited', 'gap_days', 'opener', 'first_visit', 'irregular', 'complaint_driven', 'split',
              'in_prediction_set', 'hit']
VIOLATION_COLS = ['licenseno', 'resultdttm', 'visit_ts', 'viol_ts', 'violation', 'violdesc', 'viol_level', 'comments']
LICENSE_COLS = ['category', 'issue_ts', 'property_id', 'owner_id', 'zip', 'lat', 'lon']


def load_inspections(path):
    """Read the raw inspections CSV as text; hash the legal owner and drop the contact names immediately."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])
    owner = raw['legalowner'].str.strip().str.upper()
    raw['owner_id'] = owner.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:12], na_action='ignore')
    return raw.drop(columns=PERSONAL)


def _to_local(s):
    return pd.to_datetime(s, utc=True, format='mixed', errors='coerce').dt.tz_convert(TZ)


def split_of(ts):
    """history / train / valid / test / after, by visit date (config.py)."""
    d = ts.dt.tz_localize(None)
    return np.select([d < config.TRAIN_START, d <= f'{config.TRAIN_END} 23:59:59', d <= f'{config.VALID_END} 23:59:59',
                      d <= f'{config.TEST_END} 23:59:59'],
                     ['history', 'train', 'valid', 'test'], default='after')


def license_table(raw):
    """One row per license; coordinates parsed from `location`."""
    first_val = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
    lic = raw.groupby('licenseno').agg(category=('licensecat', 'first'), property_id=('property_id', first_val),
                                       owner_id=('owner_id', first_val), location=('location', first_val),
                                       zip=('zip', first_val), issue_ts=('issue_ts', 'min'))
    coords = lic['location'].str.extract(r'\(([-\d.]+),\s*([-\d.]+)\)').astype(float)
    lic['lat'], lic['lon'] = coords[0], coords[1]
    return lic


def visit_sequence(raw):
    """One row per pass / filed / fail visit, with role, cycle-opening, first-visit and irregular flags."""
    raw = raw.assign(is_severe=raw['viol_level'].eq('***'), is_cited=raw['viol_status'].eq('Fail'))
    visits = (raw.groupby(['licenseno', 'resultdttm'], as_index=False)
                 .agg(visit_ts=('visit_ts', 'first'), result=('result', 'first'), category=('licensecat', 'first'),
                      n_viol=('violation', 'count'), n_severe=('is_severe', 'sum'), n_cited=('is_cited', 'sum'))
                 .sort_values(['licenseno', 'visit_ts'], ignore_index=True))
    unmapped = set(visits.result.dropna().unique()) - set(RESULT_ROLE)
    if unmapped:
        raise ValueError(f'unmapped result codes: {sorted(unmapped)}')
    visits['role'] = visits.result.map(RESULT_ROLE)
    seq = visits[visits.role.isin(['fail', 'pass', 'filed'])].copy()
    g = seq.groupby('licenseno')
    seq['prev_role'] = g['role'].shift()
    seq['gap_days'] = (seq['visit_ts'] - g['visit_ts'].shift()).dt.days
    seq['opener'] = seq.prev_role.isna() | seq.prev_role.isin(['pass', 'filed']) | (seq.gap_days >= CYCLE_GAP)
    seq['first_visit'] = seq.prev_role.isna()
    seq['irregular'] = seq.opener & ~seq.first_visit & (seq.gap_days < IRREGULAR_GAP)
    return seq


def complaint_driven(seq, lic, complaints):
    """Attribute each food complaint to the nearest license within 25 m with a cycle-opening visit 0–30 days later.

    Uses the visit date, so it is a label filter only, never a feature."""
    op = seq[seq.opener].join(lic[['lat', 'lon']], on='licenseno')
    op = op[op.lat.notna()].reset_index(drop=True)
    lic_xy = lic[['lat', 'lon']].dropna()
    tree = BallTree(np.radians(lic_xy.to_numpy()), metric='haversine')
    idx, dist = tree.query_radius(np.radians(complaints[['latitude', 'longitude']].to_numpy()),
                                  r=COMPLAINT_RADIUS_M / EARTH_RADIUS_M, return_distance=True)
    pairs = pd.DataFrame({'complaint': np.repeat(np.arange(len(complaints)), [len(i) for i in idx]),
                          'licenseno': lic_xy.index.to_numpy()[np.concatenate(idx)],
                          'dist_m': np.concatenate(dist) * EARTH_RADIUS_M})
    pairs['open_ts'] = complaints.open_ts.to_numpy()[pairs.complaint]
    cand = pairs.merge(op[['licenseno', 'resultdttm', 'visit_ts']], on='licenseno')
    cand['days_after'] = (cand.visit_ts - cand.open_ts).dt.total_seconds() / 86400
    cand = cand[(cand.days_after >= 0) & (cand.days_after <= COMPLAINT_WINDOW_DAYS)]
    best = cand.sort_values(['complaint', 'dist_m', 'days_after']).drop_duplicates('complaint')
    hits = set(zip(best.licenseno, best.resultdttm))
    return pd.Series([k in hits for k in zip(seq.licenseno, seq.resultdttm)], index=seq.index)


def build_tables(inspections_csv, complaints_csv):
    """All four processed tables from the raw files, as a dict of DataFrames."""
    raw = load_inspections(inspections_csv)
    raw['visit_ts'] = _to_local(raw['resultdttm'])
    raw['viol_ts'] = _to_local(raw['violdttm'])
    raw['issue_ts'] = _to_local(raw['issdttm'])
    raw = raw[raw['visit_ts'].notna() & raw['licensecat'].isin(SCOPE)].copy()

    lic = license_table(raw)
    seq = visit_sequence(raw)

    c311 = pd.read_csv(complaints_csv)
    c311['open_ts'] = pd.to_datetime(c311['open_dt'], format='mixed').dt.tz_localize(TZ, ambiguous='NaT', nonexistent='NaT')
    food = c311[c311['type'].isin(FOOD_COMPLAINTS) & c311.latitude.notna() & c311.open_ts.notna()].reset_index(drop=True)
    seq['complaint_driven'] = complaint_driven(seq, lic, food)

    seq['split'] = split_of(seq.visit_ts)
    seq['in_prediction_set'] = seq.opener & ~seq.irregular & ~seq.complaint_driven & seq.split.isin(['train', 'valid', 'test'])
    seq['hit'] = seq.n_severe.gt(0)
    seq['month'] = seq.visit_ts.dt.tz_localize(None).dt.to_period('M').astype(str)
    return {
        'visits': seq[VISIT_COLS].reset_index(drop=True),
        'violations': raw.loc[raw.violation.notna(), VIOLATION_COLS].reset_index(drop=True),
        'licenses': lic[LICENSE_COLS],
        'complaints': c311.loc[c311.latitude.notna() & c311.open_ts.notna(), ['open_ts', 'type', 'latitude', 'longitude']]
                          .reset_index(drop=True),
    }


def load_processed(names=('visits', 'violations', 'licenses', 'complaints')):
    """The processed tables written by `main`, as a dict of DataFrames."""
    return {n: pd.read_parquet(config.PROCESSED / f'{n}.parquet') for n in names}


def main():
    manifest = json.loads((config.RAW / 'manifest.json').read_text())
    m = manifest['inspections']
    print(f'building processed tables from {m["file"]} and {manifest["311_food_rodent"]["file"]}')
    tables = build_tables(config.RAW / m['file'], config.RAW / manifest['311_food_rodent']['file'])
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(config.PROCESSED / f'{name}.parquet', index=(name == 'licenses'))
        print(f'  {name}.parquet: {len(df):,} rows')


if __name__ == '__main__':
    main()
