import pandas as pd

from inspections.text import NameScan, clean_comment, clean_comments, text_windows


def test_clean_comment_masks_personal_data():
    s = clean_comment('Call inspector at jane.doe@boston.gov or 617-635-5300. (P) Spoke with Mr. Smith; PIC John Doe agreed.')
    assert '@' not in s and '617' not in s and '(P)' not in s
    assert 'Smith' not in s and 'John' not in s and 'PIC' in s
    assert clean_comment('phone 617  635 5300') == 'phone'      # a double space cannot hide a phone number


def test_clean_comments_drops_empty(tables):
    viol = tables['violations'].copy()
    viol.loc[viol.index[0], 'comments'] = 'Mr. Smith'
    com = clean_comments(viol)
    assert len(com) == viol.comments.notna().sum() - 1 and (com.text.str.len() > 0).all()


def _visits(days, openers):
    ts = [pd.Timestamp('2024-01-01', tz='America/New_York') + pd.Timedelta(days=d) for d in days]
    return pd.DataFrame({'licenseno': 'A', 'visit_ts': ts, 'opener': openers})


def test_text_windows_before_t0_newest_first_deduplicated():
    visits = _visits([0, 7, 100, 200], [True, False, True, True])
    at = visits.visit_ts.dt.tz_localize(None)                       # comments carry their visit's timestamp
    com = pd.DataFrame({'licenseno': 'A', 'ts': [at[0], at[1], at[1], at[2], at[3]],
                        'text': ['old', 'follow-up note', 'old', 'second cycle', 'too late']})
    keys = pd.DataFrame({'licenseno': ['A', 'A'], 't0': pd.to_datetime(['2024-07-01', '2024-01-01'])})
    W2 = text_windows(keys, visits, com, n_cycles=2)
    assert W2.comments[0] == ['second cycle', 'old', 'follow-up note']    # 'too late' is after t0; 'old' appears once
    assert W2.comments[1] == []                                            # nothing before the first visit
    assert (W2.latest_ts[0] < keys.t0[0])
    W1 = text_windows(keys, visits, com, n_cycles=1)
    assert W1.comments[0] == ['second cycle']


def test_name_scan():
    scan = NameScan([f'the walk in cooler at {t}f' for t in range(30, 60)])     # 'walk' appears in 30 distinct comments
    assert scan.possibly_a_name('Gwendolyn') and not scan.possibly_a_name('Walk') and not scan.possibly_a_name('Monday')
    assert '[…]' in scan.snippet('Gwendolyn said the walk in cooler was fixed')
    assert scan.ok(['the walk in cooler']) and not scan.ok(['email me at a@b.com']) and not scan.ok(['Gwendolyn'])
    assert len(scan.snippet('word ' * 100, width=40)) <= 42
