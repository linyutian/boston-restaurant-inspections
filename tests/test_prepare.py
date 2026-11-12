import numpy as np
import pandas as pd
import pytest

from inspections import prepare

TZ = 'America/New_York'


def _raw(events):
    """Raw-style rows (already parsed) for visit_sequence: (licenseno, day, result, n_violations)."""
    rows = []
    for lic, day, result, n in events:
        ts = pd.Timestamp('2021-01-04 10:00', tz=TZ) + pd.Timedelta(days=day)
        for k in range(max(n, 1)):
            rows.append({'licenseno': lic, 'resultdttm': str(ts), 'visit_ts': ts, 'result': result, 'licensecat': 'FS',
                         'violation': f'V{k}' if n else None, 'viol_level': '***' if n else None,
                         'viol_status': 'Fail' if n else None})
    return pd.DataFrame(rows)


def test_cycle_rule():
    seq = prepare.visit_sequence(_raw([
        ('A', 0, 'HE_Fail', 2),       # first visit: opens a cycle
        ('A', 7, 'HE_Pass', 0),       # follow-up after a fail
        ('A', 180, 'HE_Filed', 1),    # after a pass: opens
        ('A', 400, 'HE_Fail', 1),     # after filed: opens
        ('A', 410, 'HE_FailExt', 1),  # escalated follow-up
        ('A', 600, 'HE_Pass', 0),     # 190 days after a fail: the follow-up never happened, opens a new cycle
        ('A', 610, 'HE_Fail', 1),     # 10 days after a pass: opens, but irregular
        ('A', 620, 'HE_NotReq', 0),   # administrative: dropped
    ]))
    assert list(seq.result) == ['HE_Fail', 'HE_Pass', 'HE_Filed', 'HE_Fail', 'HE_FailExt', 'HE_Pass', 'HE_Fail']
    assert list(seq.opener) == [True, False, True, True, False, True, True]
    assert list(seq.irregular) == [False] * 6 + [True]
    assert list(seq.role) == ['fail', 'pass', 'filed', 'fail', 'fail', 'pass', 'fail']
    assert seq.n_severe.tolist() == [2, 0, 1, 1, 1, 0, 1]


def test_unknown_result_code_is_refused():
    with pytest.raises(ValueError, match='unmapped'):
        prepare.visit_sequence(_raw([('A', 0, 'HE_Mystery', 0)]))


def test_split_boundaries():
    ts = pd.Series(pd.to_datetime(['2019-12-31 23:00', '2020-01-01 00:00', '2023-12-31 23:59', '2024-01-01',
                                   '2025-01-01', '2026-10-31 23:00', '2026-11-01'], format='mixed')).dt.tz_localize(TZ)
    assert list(prepare.split_of(ts)) == ['history', 'train', 'train', 'valid', 'test', 'test', 'after']


def test_complaint_attribution():
    lic = pd.DataFrame({'lat': [42.35, 42.351], 'lon': [-71.06, -71.06]}, index=['A', 'B'])     # B is 110 m north of A
    t = lambda d: pd.Timestamp('2022-03-01', tz=TZ) + pd.Timedelta(days=d)
    seq = pd.DataFrame({'licenseno': ['A', 'A', 'B'], 'resultdttm': ['r1', 'r2', 'r3'], 'visit_ts': [t(10), t(200), t(10)],
                        'opener': True})
    complaints = pd.DataFrame({'latitude': [42.35005, 42.3505], 'longitude': [-71.06, -71.06], 'open_ts': [t(3), t(3)]})
    flags = prepare.complaint_driven(seq, lic, complaints)
    # complaint 1 is 6 m from A, 7 days before A's visit; complaint 2 is 55 m from both, so it matches nobody
    assert list(flags) == [True, False, False]


def test_build_tables_end_to_end(tmp_path):
    base = {'licensecat': 'FS', 'descript': 'Eating & Drinking', 'property_id': '1', 'location': '(42.35, -71.06)', 'zip': '02118',
            'legalowner': 'ACME LLC', 'namefirst': 'Jane', 'namelast': 'Doe', 'licstatus': 'Active', 'businessname': 'Cafe',
            'issdttm': '2019-01-01 00:00:00+00', 'status_date': '', 'violdttm': ''}
    rows = [dict(base, licenseno='1', result='HE_Fail', resultdttm='2021-03-02 15:00:00+00', violation='V1',
                 violdesc='Cold Holding (P)', viol_level='***', viol_status='Fail', comments='at 50F, discard'),
            dict(base, licenseno='1', result='HE_Pass', resultdttm='2021-03-09 15:00:00+00', violation='', violdesc='',
                 viol_level='', viol_status='', comments=''),
            dict(base, licenseno='1', result='HE_Filed', resultdttm='2021-09-01 01:30:00+00', violation='V2',
                 violdesc='Floors', viol_level='*', viol_status='Fail', comments='clean floor'),
            dict(base, licenseno='2', licensecat='MFW', result='HE_Pass', resultdttm='2021-03-02 15:00:00+00', violation='',
                 violdesc='', viol_level='', viol_status='', comments='')]
    insp = tmp_path / 'insp.csv'
    pd.DataFrame(rows).to_csv(insp, index=False)
    c311 = tmp_path / 'c311.csv'
    pd.DataFrame({'case_enquiry_id': [1], 'open_dt': ['2021-08-25 09:00:00'], 'closed_dt': [''], 'type': ['Unsanitary Conditions - Food'],
                  'case_status': ['Closed'], 'neighborhood': ['South End'], 'latitude': [42.35], 'longitude': [-71.06]}).to_csv(c311, index=False)

    t = prepare.build_tables(insp, c311)
    v = t['visits']
    assert list(v.result) == ['HE_Fail', 'HE_Pass', 'HE_Filed']               # mobile food (MFW) is out of scope
    assert list(v.opener) == [True, False, True] and list(v.hit) == [True, False, False]
    assert v.complaint_driven.tolist() == [False, False, True]                # complaint 7 days before the filed visit, 0 m away
    assert v.month.tolist() == ['2021-03', '2021-03', '2021-08']              # 01:30 UTC on Sep 1 is Aug 31 in Boston
    assert list(v.in_prediction_set) == [True, False, False]
    assert len(t['violations']) == 2 and t['licenses'].loc['1', 'lat'] == 42.35
    everything = set().union(*(set(df.columns) for df in t.values()))
    assert not everything & {'namefirst', 'namelast', 'legalowner', 'licstatus', 'businessname'}
    assert t['licenses'].owner_id.str.fullmatch(r'[0-9a-f]{12}').all()
