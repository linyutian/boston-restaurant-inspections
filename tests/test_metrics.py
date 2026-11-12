import numpy as np
import pandas as pd

from inspections.metrics import bootstrap_diff, capture_curve, days_earlier, first_half, hit_rate, per_month


def test_hit_rate_of_todays_order_and_oracle(toy_month):
    assert hit_rate(per_month(toy_month, 'actual')) == 0.0          # all four hits come late in the month
    assert hit_rate(per_month(toy_month, 'oracle')) == 4 / 5        # hindsight: four hits in the first five visits
    assert hit_rate(per_month(toy_month, 'reverse')) == 4 / 5


def test_days_earlier(toy_month):
    assert days_earlier(toy_month, 'actual') == 0
    # reversing the month moves the hits (days 18-27) to the first four slots (days 0-9): 18 days earlier each
    assert days_earlier(toy_month, 'reverse') == 18
    assert days_earlier(toy_month, 'oracle') > 0


def test_first_half_matches_per_month(toy_month):
    for score in ['actual', 'oracle', 'reverse']:
        flag = first_half(toy_month, score)
        pm = per_month(toy_month, score)
        assert flag.sum() == pm.visits.sum() and toy_month.hit[flag].sum() == pm.hits.sum()


def test_ties_are_broken_by_the_fixed_tiebreak(toy_month):
    tied = toy_month.assign(flat=0.0)
    expected = toy_month.sort_values('tiebreak').head(5).hit.sum()
    assert per_month(tied, 'flat').hits.sum() == expected


def test_capture_curve_ends_at_one(toy_month):
    curve = capture_curve(toy_month, 'oracle', np.linspace(0.1, 1, 10))
    assert curve[-1] == 1.0 and np.all(np.diff(curve) >= 0)


def test_bootstrap_ci():
    months = [f'2024-{m:02d}' for m in range(1, 13)]
    a = pd.DataFrame({'hits': [30] * 12, 'visits': [100] * 12}, index=months)
    b = pd.DataFrame({'hits': [25] * 12, 'visits': [100] * 12}, index=months)
    assert np.allclose(bootstrap_diff(a, b), [0.05, 0.05])
    assert np.allclose(bootstrap_diff(a, a), [0, 0])
    noisy = a.assign(hits=np.random.default_rng(1).integers(20, 40, 12))
    lo, hi = bootstrap_diff(noisy, b)
    assert lo <= hit_rate(noisy) - hit_rate(b) <= hi
