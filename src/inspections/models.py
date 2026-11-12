"""The adopted models (TECHNICAL.md §7).

- **Model (b):** L2 logistic regression (C = 0.003) on the 33 features of `features.build_model_b`.
- **Model (b) + text:** the same, plus the logit of a TF-IDF text score as a 34th input. The text
  score on training rows is out-of-fold (5 groups by establishment), so the classifier never sees a
  text score fitted on the same row (stacking, notebook 04 §4).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inspections import config

MODEL_C = 0.003      # model (b), notebook 03b
TFIDF_C = 0.3        # text model, notebook 04 (E1)
N_CYCLES = 2         # routine cycles of comment text, notebook 04 (E1)


def make_logreg(C=MODEL_C, penalty='l2'):
    """Imputation (median + missing indicators), scaling and a logistic regression."""
    # scikit-learn 1.9 expresses the penalty through l1_ratio (0 = L2, 1 = L1)
    l1, solver = (1, 'liblinear') if penalty == 'l1' else (0, 'lbfgs')
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, l1_ratio=l1, solver=solver, max_iter=5000, random_state=config.SEED))


def tfidf_model(C=TFIDF_C):
    """The selected text model (notebook 04, E1): TF-IDF words and word pairs, logistic regression."""
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.5, sublinear_tf=True, strip_accents='unicode'),
                         LogisticRegression(C=C, max_iter=3000, random_state=config.SEED))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def text_score(inputs, y, groups, train, score, make_model=tfidf_model, inner=5):
    """Out-of-fold text scores on `train` rows (GroupKFold by `groups`) and scores on `score` rows from a fit on all `train` rows.

    `train` and `score` are boolean masks over the rows of `inputs`; other rows stay NaN."""
    tr, sc = np.where(train)[0], np.where(score)[0]
    s = pd.Series(np.nan, index=inputs.index)
    for a, b in GroupKFold(inner).split(tr, groups=groups[tr]):
        s.iloc[tr[b]] = make_model().fit(inputs.iloc[tr[a]], y[tr[a]]).predict_proba(inputs.iloc[tr[b]])[:, 1]
    s.iloc[sc] = make_model().fit(inputs.iloc[tr], y[tr]).predict_proba(inputs.iloc[sc])[:, 1]
    return s


def fit_and_score(X, y, train, score, text=None, groups=None):
    """Fit model (b) on `train` rows and score `score` rows; with `text`, also model (b) + text.

    Returns a frame indexed like the scored rows with `model_b` (and `with_text`, `text_score`)."""
    if (train & score).any():
        raise ValueError('a row cannot be both trained on and scored')
    tr, sc = np.where(train)[0], np.where(score)[0]
    out = pd.DataFrame({'model_b': make_logreg().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[sc])[:, 1]}, index=X.index[sc])
    if text is not None:
        s = text_score(text, y, groups, train, score)
        X_t = X.assign(text_0=logit(s))
        out['with_text'] = make_logreg().fit(X_t.iloc[tr], y[tr]).predict_proba(X_t.iloc[sc])[:, 1]
        out['text_score'] = s.iloc[sc].to_numpy()
    return out
