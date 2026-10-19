"""Conservative ratio proxy from the original synchronous target cost data.

The measurements include target padding, metadata, model and synchronization.
They exclude real draft work. This is a scheduling proxy, not a new latency
measurement or a guarantee that a branch improves whole-workload Goodput.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

EVIDENCE = Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/ROOT296_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json')
EXPECTED = '5ae413e466e751bf5e61896baab0c195c746edbd24062710092e7ba7c55b31e1'


@lru_cache(maxsize=1)
def envelope():
    data = EVIDENCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise RuntimeError('ROOT309 exact audited target cost evidence required')
    audit = json.loads(data)
    if not audit['status'].startswith('PASS_') or len(audit['target_costs']) != 33:
        raise RuntimeError('ROOT309 all original target cost configurations required')
    rows = audit['target_costs']
    first = {(row['rank'], row['N']): row for row in rows if row['branch_owners'] == 1}
    setup = max(row['median_ms'][str(b)] / row['median_ms']['4'] - 1
                for row in first.values() for b in range(5, 9))
    slopes = [0.]
    for row in rows:
        forks = row['branch_owners']
        if forks > 1:
            initial = first[row['rank'], row['N']]
            for budget in range(5, 9):
                ratio = row['median_ms'][str(budget)] / row['median_ms']['4']
                initial_ratio = initial['median_ms'][str(budget)] / initial['median_ms']['4']
                slopes.append((ratio - initial_ratio) / (forks - 1))
    slope = max(slopes)
    for row in rows:
        predicted = 1 + setup + slope * (row['branch_owners'] - 1)
        for budget in range(5, 9):
            assert predicted + 1e-12 >= row['median_ms'][str(budget)] / row['median_ms']['4']
    return setup, slope


def ratio(branches):
    if type(branches) is not int or not 0 <= branches <= 12:
        raise ValueError('ROOT309 unchanged maximum12 branch owners')
    if branches == 0:
        return 1.
    setup, slope = envelope()
    return 1 + setup + slope * (branches - 1)
