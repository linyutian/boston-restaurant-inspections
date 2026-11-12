import numpy as np
import pandas as pd
import pytest

from inspections import features


@pytest.fixture(scope='module')
def built(tables):
    v = tables['visits']
    rows = v[v.in_prediction_set].copy()
    keys = features.decision_keys(rows)
    gc = features.violation_group_counts(tables['violations'])
    X = features.build_model_b(keys, v, gc, tables['licenses'])
    return rows, keys, gc, X


def test_violation_groups():
    assert features.violation_group('Rodents / insects present') == 'pests'
    assert features.violation_group('Cold Holding temperature') == 'temperature'
    assert features.violation_group('Hand washing sink not accessible') == 'handwashing'
    assert features.violation_group('Something unrelated') == 'other'


def test_model_b_has_33_features(built):
    _, _, _, X = built
    assert X.shape[1] == 33
    assert {'prev_severe', 'prev_fail', 'days_since_last_visit', 'ewm_severe', 'area_severe_rate_180d'} <= set(X.columns)
    assert not any(c.startswith(('owner', 'cat_', 'food_complaints')) for c in X.columns)


def test_decision_keys_are_month_starts(built):
    rows, keys, _, _ = built
    assert (keys.t0.dt.day == 1).all() and (keys.t0 <= rows.visit_ts.dt.tz_localize(None)).all()


def test_history_matches_a_hand_count(tables, built):
    rows, keys, _, X = built
    v = tables['visits']
    i = rows.index[20]
    lic, t0 = keys.loc[i, 'licenseno'], keys.loc[i, 't0']
    before = v[(v.licenseno == lic) & (v.visit_ts.dt.tz_localize(None) < t0)]
    last_routine = before[before.opener].iloc[-1]
    assert X.loc[i, 'prev_severe'] == float(last_routine.n_severe > 0)
    assert X.loc[i, 'prev_fail'] == float(last_routine.role == 'fail')
    assert X.loc[i, 'n_prior_routine'] == before.opener.sum()
    assert X.loc[i, 'days_since_last_visit'] == (t0 - before.visit_ts.iloc[-1].tz_localize(None)).days


def test_features_use_only_data_before_t0(tables, built):
    """Truncation test (R14, R25): deleting everything from a cut-off on leaves earlier rows' features unchanged."""
    rows, keys, _, X = built
    cut = pd.Timestamp('2022-07-01')
    early = keys.t0 < cut
    v, viol = tables['visits'], tables['violations']
    v_cut = v[v.visit_ts.dt.tz_localize(None) < cut]
    gc_cut = features.violation_group_counts(viol[viol.visit_ts.dt.tz_localize(None) < cut])
    X_cut = features.build_model_b(keys[early], v_cut, gc_cut, tables['licenses'])
    assert np.isclose(X_cut.to_numpy(), X[early].to_numpy(), equal_nan=True).all()


def test_area_features_count_neighbours_in_radius_and_window():
    lic = pd.DataFrame({'lat': [42.35, 42.3545, 42.40], 'lon': [-71.06, -71.06, -71.06]}, index=['A', 'B', 'C'])  # B 500 m, C 5.6 km
    o = pd.DataFrame({'licenseno': ['B', 'B', 'C', 'A'],
                      'ts': pd.to_datetime(['2024-05-01', '2023-01-01', '2024-05-01', '2024-06-10']),
                      'severe': [True, False, True, False], 'is_fail': [True, True, True, False]})
    keys = pd.DataFrame({'licenseno': ['A'], 't0': [pd.Timestamp('2024-06-01')]})
    F = features.area_features(keys, o, lic, radius_m=1000, windows=(90, 180))
    assert F.loc[0, 'area_routine_visits_90d'] == 1           # B's May visit; not C (too far), not A's June visit (after t0)
    assert F.loc[0, 'area_severe_rate_90d'] == 1.0


def test_rule1_orders_priority_then_fail_then_wait():
    X = pd.DataFrame({'prev_severe': [1, 0, 0, np.nan], 'prev_fail': [0, 1, 0, np.nan], 'days_since_last_visit': [10, 10, 400, np.nan]})
    s = features.rule1_score(X)
    assert list(s.sort_values(ascending=False).index) == [0, 1, 3, 2]   # never-visited ranks above a clean long wait


def test_phase1_feature_set(tables):
    v = tables['visits']
    rows = v[v.in_prediction_set].head(50)
    complaints = pd.DataFrame({'open_ts': pd.to_datetime(['2021-01-05', '2021-02-10'], utc=True).tz_convert('America/New_York'),
                               'type': ['Rodent Activity', 'Unsanitary Conditions - Food'],
                               'latitude': [42.33, 42.334], 'longitude': [-71.07, -71.066]})
    gc = features.violation_group_counts(tables['violations'])
    X, t0 = features.build_phase1_features(rows, v, complaints, gc, tables['licenses'])
    # 47 features plus one violation-mix column per group present (8 groups on the real data: 55 features)
    assert X.shape == (50, 47 + gc.shape[1]) and len(t0) == 50
