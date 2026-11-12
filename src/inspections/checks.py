"""Risk-register checks: record each check, print it, and stop on failure.

    CHECKS = CheckLog()
    check = CHECKS.check
    check('R15', 'text comes only from visits before t0', ok, detail)
    pd.DataFrame(CHECKS)          # the full list, at the end of a notebook
"""


class CheckLog(list):
    """A list of check records (dicts), so `pd.DataFrame(log)` gives the summary table."""

    def check(self, risk, what, ok, detail=''):
        """Record a check against risk `risk`; raise AssertionError if it fails."""
        self.append({'risk': risk, 'check': what, 'result': 'pass' if ok else 'FAIL', 'detail': detail})
        assert ok, f'{risk} failed: {what} ({detail})'
        print(f'✓ {risk}: {what}' + (f'  [{detail}]' if detail else ''))
