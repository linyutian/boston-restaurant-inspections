"""Integration check on the real data: the package reproduces the adopted model's logged result.

Skipped when data/processed/ has not been built (python -m inspections.download && python -m inspections.prepare)."""
import pandas as pd
import pytest

from inspections import config, features, models
from inspections.metrics import hit_rate, per_month

pytestmark = pytest.mark.skipif(not (config.PROCESSED / 'visits.parquet').exists(), reason='processed data not built')


def test_model_b_2024_result_matches_the_experiment_log():
    visits = pd.read_parquet(config.PROCESSED / 'visits.parquet')
    violations = pd.read_parquet(config.PROCESSED / 'violations.parquet')
    licenses = pd.read_parquet(config.PROCESSED / 'licenses.parquet')
    rows = visits[visits.in_prediction_set & visits.split.isin(['train', 'valid'])].copy()
    year = rows.visit_ts.dt.year.to_numpy()
    X = features.build_model_b(features.decision_keys(rows), visits, features.violation_group_counts(violations), licenses)
    scores = models.fit_and_score(X, rows.hit.to_numpy(), year < 2024, year == 2024)

    ev = rows.loc[year == 2024, ['month', 'visit_ts', 'hit']].copy()
    ev['tiebreak'] = 0.0
    ev['model_b'] = scores.model_b
    logged = pd.read_csv(config.RESULTS / 'experiments.csv')
    expected = logged.loc[(logged.notebook == '04_text_models') & (logged.experiment == 'Model (b)'), 'HR@50'].item()
    assert hit_rate(per_month(ev, 'model_b')) == pytest.approx(expected, abs=1e-12)
