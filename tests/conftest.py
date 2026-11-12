"""Small synthetic tables in the processed format, so the tests run without the real data."""
import numpy as np
import pandas as pd
import pytest

TZ = 'America/New_York'
DESCRIPTIONS = ['Cold Holding temperature', 'Rodents / insects present', 'Hand washing sink not accessible',
                'Food contact surfaces not sanitized', 'Floors walls ceilings not clean', 'Raw food stored above ready-to-eat']
COMMENTS = ['raw chicken stored above vegetables in walk-in', 'clean floor under the fryer', 'mouse droppings in basement',
            'cold holding at 48F discard', 'provide paper towels at hand sink', 'sanitizer below 50 ppm']


def _ts(day):
    return (pd.Timestamp('2018-01-01', tz=TZ) + pd.Timedelta(days=float(day))).as_unit('us')   # the processed tables use µs


@pytest.fixture(scope='session')
def tables():
    """visits / violations / licenses for 40 synthetic establishments, 2018–2025: routine visits about every six
    months, a follow-up a week after every fail, risk that persists within an establishment."""
    rng = np.random.default_rng(0)
    visit_rows, viol_rows, lic_rows = [], [], []
    for i in range(40):
        lic = f'L{i:03d}'
        risk = rng.uniform(0.05, 0.6)
        lic_rows.append({'licenseno': lic, 'category': ['FS', 'FT', 'RF'][i % 3], 'issue_ts': _ts(0), 'property_id': None,
                         'owner_id': f'o{i % 7}', 'zip': '02118',
                         'lat': 42.33 + 0.004 * (i % 8), 'lon': -71.07 + 0.004 * (i // 8)})
        day = rng.uniform(0, 150)
        while day < 2900:
            severe = rng.random() < risk
            failed = severe or rng.random() < 0.3
            n_viol = int(rng.integers(1, 4)) if failed else int(rng.integers(0, 2))
            visit_rows.append((lic, day, 'HE_Fail' if failed else 'HE_Pass', True, n_viol, int(severe)))
            for k in range(n_viol):
                j = int(rng.integers(len(DESCRIPTIONS)))
                viol_rows.append((lic, day, j, '***' if (severe and k == 0) else '*', COMMENTS[j]))
            if failed:
                visit_rows.append((lic, day + 7, 'HE_Pass', False, 0, 0))
            day += rng.uniform(150, 220)
    v = pd.DataFrame(visit_rows, columns=['licenseno', 'day', 'result', 'opener', 'n_viol', 'n_severe'])
    v['visit_ts'] = v.day.map(_ts).astype(f'datetime64[us, {TZ}]')
    v['resultdttm'] = v.visit_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    v['month'] = v.visit_ts.dt.tz_localize(None).dt.to_period('M').astype(str)
    v['category'] = v.licenseno.map({r['licenseno']: r['category'] for r in lic_rows})
    v['role'] = np.where(v.result.eq('HE_Fail'), 'fail', 'pass')
    v['n_cited'] = v.n_viol
    v['gap_days'] = v.groupby('licenseno').visit_ts.diff().dt.days
    v['first_visit'] = v.groupby('licenseno').cumcount().eq(0)
    v['irregular'] = False
    v['complaint_driven'] = False
    year = v.visit_ts.dt.year
    v['split'] = np.select([year < 2020, year < 2024, year < 2025], ['history', 'train', 'valid'], 'test')
    v['in_prediction_set'] = v.opener & (year >= 2020)
    v['hit'] = v.n_severe.gt(0)
    v = v.drop(columns='day').sort_values(['licenseno', 'visit_ts'], ignore_index=True)

    viol = pd.DataFrame(viol_rows, columns=['licenseno', 'day', 'j', 'viol_level', 'comments'])
    viol['visit_ts'] = viol.day.map(_ts).astype(f'datetime64[us, {TZ}]')
    viol['resultdttm'] = viol.visit_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    viol['viol_ts'] = viol.visit_ts
    viol['violation'] = 'V' + viol.j.astype(str)
    viol['violdesc'] = viol.j.map(dict(enumerate(DESCRIPTIONS)))
    viol = viol[['licenseno', 'resultdttm', 'visit_ts', 'viol_ts', 'violation', 'violdesc', 'viol_level', 'comments']]
    licenses = pd.DataFrame(lic_rows).set_index('licenseno')
    return {'visits': v, 'violations': viol, 'licenses': licenses}


@pytest.fixture
def toy_month():
    """Ten visits in one month, three days apart; the last four find a Priority violation."""
    df = pd.DataFrame({'month': ['2024-01'] * 10,
                       'visit_ts': pd.date_range('2024-01-01', periods=10, freq='3D', tz=TZ),
                       'hit': [False] * 6 + [True] * 4,
                       'tiebreak': np.random.default_rng(0).random(10)})
    df['actual'] = -df.visit_ts.rank()
    df['oracle'] = df.hit.astype(int)
    df['reverse'] = df.visit_ts.rank()
    return df
