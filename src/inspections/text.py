"""Inspector-comment text (TECHNICAL.md §7, Phase 2): personal-data masking and decision-time text windows."""
import re
from collections import Counter

import numpy as np
import pandas as pd

EMAIL = re.compile(r'[\w.\-]+@[\w\-]+(\.[\w\-]+)+', re.I)
PHONE = re.compile(r'\(?\b\d{3}\)?[-. ]?\d{3}[-. ]\d{4}\b')
TAG = re.compile(r'\((P|Pf|PF|C)\)')
TITLED = re.compile(r'\b(Mr|Mrs|Ms|Miss)\.? [A-Z][a-z]+( [A-Z][a-z]+)?')
ROLE_NAME = re.compile(r'\b(PIC|P\.I\.C\.|[Mm]anager|[Oo]wner|[Cc]hef|[Ss]upervisor)([,:]| named| is)? ([A-Z][a-z]+)( [A-Z][a-z]+)?')
CALENDAR = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
            'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'monday',
            'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}


def clean_comment(s):
    """Mask emails, phone numbers, Food Code tags, titled names and role-named persons."""
    s = re.sub(r'\s+', ' ', s)                          # first, so a double space cannot hide a phone number
    s = EMAIL.sub(' ', s)
    s = PHONE.sub(' ', s)
    s = TAG.sub(' ', s)
    s = TITLED.sub(' ', s)
    s = ROLE_NAME.sub(lambda m: m.group(1) + ' ', s)     # keep the role word, drop what may be a name
    return re.sub(r'\s+', ' ', s).strip()


def clean_comments(violations):
    """One row per non-empty cleaned comment: licenseno, resultdttm, visit_ts, comments (raw), text (cleaned), ts (naive)."""
    com = violations.loc[violations.comments.notna(), ['licenseno', 'resultdttm', 'visit_ts', 'comments']].copy()
    com['text'] = com.comments.map(clean_comment)
    com = com[com.text.str.len() > 0]
    com['ts'] = com.visit_ts.dt.tz_localize(None)
    return com


def text_windows(keys, visits, com, n_cycles):
    """For each row (licenseno, t0): de-duplicated comments from the last n routine cycles before t0, newest first."""
    op = visits.loc[visits.opener, ['licenseno', 'visit_ts']]
    op_by = {k: np.sort(g.visit_ts.dt.tz_localize(None).to_numpy()) for k, g in op.groupby('licenseno')}
    cs = com.sort_values(['licenseno', 'ts'], kind='stable')
    c_by = {k: (g.ts.to_numpy(), g.text.to_numpy()) for k, g in cs.groupby('licenseno')}
    texts, latest = [], []
    for license_no, t0 in zip(keys.licenseno.to_numpy(), keys.t0.to_numpy()):
        ots, cc = op_by.get(license_no), c_by.get(license_no)
        k = 0 if ots is None else np.searchsorted(ots, t0, side='left')     # routine visits strictly before t0
        if cc is None or k == 0:
            texts.append([]); latest.append(np.datetime64('NaT', 'us')); continue
        ts, tx = cc
        lo, hi = np.searchsorted(ts, ots[max(k - n_cycles, 0)], side='left'), np.searchsorted(ts, t0, side='left')
        texts.append(list(dict.fromkeys(tx[lo:hi][::-1])))                  # newest first, de-duplicated
        latest.append(ts[hi - 1] if hi > lo else np.datetime64('NaT', 'us'))
    return pd.DataFrame({'comments': texts, 'latest_ts': latest}, index=keys.index)


class NameScan:
    """Display-time personal-data filter (R18), stricter than the masking used for modelling.

    A capitalised word is treated as a possible name if it is written in lower case fewer than
    `min_lower` times anywhere in the comment corpus."""

    def __init__(self, texts, min_lower=20):
        self.lower_counts = Counter(w for t in pd.unique(pd.Series(texts)) for w in re.findall(r'\b[a-z]+\b', t))
        self.min_lower = min_lower

    def possibly_a_name(self, word):
        return word[:1].isupper() and word.lower() not in CALENDAR and self.lower_counts[word.lower()] < self.min_lower

    def is_rare(self, word):
        """For lower-case model terms: rare enough in the corpus that it might be a name."""
        return self.lower_counts[word] < self.min_lower and not word.isdigit()

    def snippet(self, text, width=160):
        """Mask likely names, then cut to `width` characters."""
        out = re.sub(r'\b[A-Z][a-z]+\b', lambda m: '[…]' if self.possibly_a_name(m.group()) else m.group(), clean_comment(text))
        return out if len(out) <= width else out[:width].rsplit(' ', 1)[0] + ' …'

    def ok(self, shown):
        """True if nothing displayed contains an email, phone number, title + name, or likely name."""
        return all(not (EMAIL.search(s) or PHONE.search(s) or TITLED.search(s)
                        or any(self.possibly_a_name(w) for w in re.findall(r'\b[A-Z][a-z]+\b', s))) for s in shown)
