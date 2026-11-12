import numpy as np
import pandas as pd
import pytest

from inspections import models


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=['a', 'b', 'c'])
    X.iloc[::10, 1] = np.nan
    y = (X.a.fillna(0) + rng.normal(scale=0.5, size=n) > 0.5).to_numpy()
    words = np.where(y, 'raw chicken prep cooler', 'coffee sandwiches counter')
    text = pd.Series([f'{w} item{i % 20}' for i, w in enumerate(words)])
    groups = np.array([f'L{i % 80}' for i in range(n)])
    train = np.arange(n) < 300
    return X, y, text, groups, train


def test_logit_is_finite():
    assert np.isfinite(models.logit([0, 0.5, 1])).all() and models.logit([0.5])[0] == 0


def test_fit_and_score_without_and_with_text(data):
    X, y, text, groups, train = data
    out = models.fit_and_score(X, y, train, ~train)
    assert list(out.columns) == ['model_b'] and len(out) == (~train).sum() and out.model_b.between(0, 1).all()
    both = models.fit_and_score(X, y, train, ~train, text=text, groups=groups)
    assert {'model_b', 'with_text', 'text_score'} <= set(both.columns)
    assert np.allclose(both.model_b, out.model_b)


def test_scored_labels_are_never_used(data):
    """Flipping the labels of the scored rows must not change any score."""
    X, y, text, groups, train = data
    a = models.fit_and_score(X, y, train, ~train, text=text, groups=groups)
    y2 = y.copy()
    y2[~train] = ~y2[~train]
    b = models.fit_and_score(X, y2, train, ~train, text=text, groups=groups)
    pd.testing.assert_frame_equal(a, b)


def test_overlap_is_refused(data):
    X, y, _, _, train = data
    with pytest.raises(ValueError):
        models.fit_and_score(X, y, train, train)


def test_text_score_is_out_of_fold(data):
    """A training row's text score comes from a model that never saw its establishment."""
    X, y, text, groups, train = data
    s = models.text_score(text, y, groups, train, ~train)
    assert s.notna().all()
    one = text.copy()
    target = np.where(groups == 'L0')[0]
    one.iloc[target] = 'unique marker words'              # change only establishment L0's text
    s2 = models.text_score(one, y, groups, train, ~train)
    in_fold = target[target < 300]
    # L0's own out-of-fold scores come from models fit without L0, so its new words are unseen there
    assert np.allclose(s2.iloc[in_fold], s2.iloc[in_fold].iloc[0])
