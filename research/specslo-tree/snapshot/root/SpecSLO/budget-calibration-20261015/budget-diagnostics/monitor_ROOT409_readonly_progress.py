"""Print bounded progress without touching model, evidence, locks or clocks."""
import hashlib
import json
from pathlib import Path
import time

B = Path('/root/SpecSLO/budget-calibration-20261015')
TAG = 'dual-S24-S32-global64-root-v409-full463-ownAR-v1'
DONOR = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
ACTUAL = DONOR / 'actual' / TAG


def main():
    until = time.monotonic() + 5400
    while time.monotonic() < until:
        receipt_path = DONOR / f'receipt-{TAG}.json'
        try:
            raw = receipt_path.read_bytes()
            receipt = json.loads(raw)
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        value = dict(UTC=time.strftime('%H:%M:%S', time.gmtime()),
                     supervisor=receipt['status'],
                     original227=len(list((ACTUAL / 'canonical-independent-node-math').glob('rank1-canonical-case*.json'))),
                     uniform681={p.name: len(list(p.glob('rank1-canonical-case*.json')))
                                 for p in (ACTUAL / 'target-uniform-KV-bucket-math409').glob('*') if p.is_dir()},
                     wide200={p.name: len(list(p.glob('rank1*case*.json')))
                              for p in (ACTUAL / 'target-S32-independent-math409').glob('*') if p.is_dir()},
                     windows=[p.name for p in (ACTUAL / 'windows/window64').glob('*') if p.is_dir()],
                     ownAR_cohorts=len(list((ACTUAL / 'postfull-independent-FIA-AR').glob('rank1-cohort*.json.gz'))),
                     errors={p.name: p.read_text()[-1800:] for p in ACTUAL.glob('rank*-last-error.json')})
        if receipt['status'] != 'running':
            value.update(receipt_sha256=hashlib.sha256(raw).hexdigest(), errors=receipt['errors'])
        print(json.dumps(value), flush=True)
        if receipt['status'] != 'running':
            return
        time.sleep(50)


if __name__ == '__main__':
    main()
