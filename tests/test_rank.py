import numpy as np
import pandas as pd
import pytest

from inspections.rank import priority_list


@pytest.fixture(scope='module')
def ranked(tables):
    return priority_list('2024-06', tables)


def test_list_is_ranked_by_score(tables, ranked):
    v = tables['visits']
    due = v[(v.month == '2024-06') & v.opener].licenseno
    assert sorted(ranked.licenseno) == sorted(due)
    assert list(ranked['rank']) == list(range(1, len(ranked) + 1))
    assert ranked.risk_score.is_monotonic_decreasing and ranked.risk_score.between(0, 1).all()
    assert {'last_routine_priority_violation', 'last_inspection_comments', 'days_since_last_visit'} <= set(ranked.columns)


def test_explicit_due_list(tables):
    out = priority_list('2025-03', tables, due=['L001', 'L005', 'L999'])     # L999 has no history: still ranked
    assert set(out.licenseno) == {'L001', 'L005', 'L999'}
    assert out.set_index('licenseno').loc['L999', 'last_inspection_comments'] == ''


def test_nothing_from_the_month_or_later_is_used(tables, ranked):
    """Changing every visit from the planned month on must not change the list."""
    v = tables['visits'].copy()
    later = v.visit_ts.dt.tz_localize(None) >= pd.Timestamp('2024-06-01')
    v.loc[later, ['hit', 'n_severe']] = [True, 5]
    v.loc[later, 'role'] = 'fail'
    altered = dict(tables, visits=v)
    out = priority_list('2024-06', altered)
    pd.testing.assert_frame_equal(out, ranked)


def test_text_option_runs(tables):
    out = priority_list('2024-06', tables, with_text=True)
    assert len(out) > 0 and out.risk_score.is_monotonic_decreasing


def test_month_without_visits_needs_a_due_list(tables):
    with pytest.raises(ValueError, match='--due'):
        priority_list('2030-01', tables)
