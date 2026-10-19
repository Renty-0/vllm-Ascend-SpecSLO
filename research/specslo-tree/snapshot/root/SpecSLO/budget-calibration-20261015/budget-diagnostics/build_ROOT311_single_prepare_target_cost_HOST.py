"""Prepare a fresh target-math/cost experiment for the305 prepare path."""
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
OLD = B / 'tree-concrete-device-span-dual-cache-target-cost-v296'
P = B / 'tree-single-prepare-target-math-cost-v311'
TAG = 'single-prepare-original-raw-kernels-v311-B1to8-408-math-cost-v1'


def main():
    P.mkdir(exist_ok=False)
    provenance = []
    for source in OLD.iterdir():
        if not source.is_file() or (source.suffix not in ('.py', '.sh')
                                    and source.name not in ('FIXTURE296.json', 'DECLARED296_CASES.json')):
            continue
        target = P / source.name.replace('296', '311')
        if source.suffix == '.json':
            target.write_bytes(source.read_bytes())
        else:
            text = source.read_text().replace(OLD.name, P.name).replace('296', '311')
            text = text.replace('ROOT311:copy-grid:', 'ROOT296:copy-grid:')
            target.write_text(text)
        provenance.append(dict(source=str(source),
                               sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               saved=str(target)))
    extra = [str(B / name) for name in (
        'tree-single-validated-prefix-prepare-host-v305',
        'tree-single-validated-prefix-descriptors-host-v304',
        'tree-concrete-device-span-dual-cache-target-cost-v296')]
    for name in ('check311.py', 'runtime311.py'):
        path = P / name
        text = path.read_text()
        anchor = 'P=Path(__file__).resolve().parent;B=P.parent\n'
        assert text.count(anchor) == 1
        path.write_text(text.replace(anchor, anchor + 'sys.path.extend(' + repr(extra) + ')\n'))
    path = P / 'startup311.sh'
    text = path.read_text()
    assert text.count('export PYTHONPATH=') == 1
    path.write_text(text.replace('export PYTHONPATH=', 'export PYTHONPATH=' + ':'.join(extra) + ':'))
    (P / 'static_state311.py').write_text(
        '"""Exact305 prepare; same296 raw kernels, cache checks and attention."""\n'
        'from static_state305 import ensure_zero,bind_checked_prepare,install,require_copy_bindings\n')
    (P / 'capture_key311.py').write_text(
        '"""Retain the exact qualified296 dispatcher and capture keys."""\n'
        'from capture_key296 import actual_source,projected,key_for,install\n')
    (P / 'device311.py').write_text(
        '"""The same concrete-device validation used by305 and296."""\n'
        'from device296 import resolve_cache_device\n')
    source = B / 'tree-single-validated-prefix-prepare-host-v305/check305_HOST.py'
    text = source.read_text().replace('import check296', 'import check311')
    text = text.replace('HOST_SPLIT305_SPANS_RESULT.json', 'HOST_SPLIT311_SPANS_RESULT.json')
    (P / 'check_split311_HOST.py').write_text(text)
    for name in ('check_grid311_HOST.py', 'check_device311_HOST.py'):
        path = P / name
        text = path.read_text()
        anchor = 'from pathlib import Path\n'
        assert text.count(anchor) == 1
        path.write_text(text.replace(anchor, anchor + 'import check311\n'))
    source = B / 'tree-capacity-request-risk-span-full-v299/check_launch299_HOST.py'
    (P / 'check_launch311_HOST.py').write_text(source.read_text().replace('299', '311'))
    auditor = B / 'budget-diagnostics/audit_ROOT296_target_math_and_cost_HOST.py'
    text = auditor.read_text().replace(OLD.name, P.name).replace('296', '311')
    text = text.replace('span-prefix-dual-cache-concrete-device-v311-B1to8-408-math-cost-v1', TAG)
    (B / 'budget-diagnostics/audit_ROOT311_target_math_and_cost_HOST.py').write_text(text)
    review = dict(
        status='SOURCE_ONLY311_fresh_NN_and_cost_not_frozen_not_executed',
        only_serving_delta_from296='HOST305 single immutable row validation, direct span/alias preparation and no unused prefix staging',
        raw_kernels_and_capture_keys_and_attention_identical_to296=True,
        fresh_target_cases_per_rank=408, target_ranks=[1, 2, 3],
        fresh_independent_committed_prefixes=72,
        paired_cost_configurations=11, rotated_cost_rounds=20,
        measured_scope='Synchronous target pad/metadata/model/sync; no real draft or whole workload performance',
        prerequisite='Finish299 and its independent full463 AR audit before NPU launch',
        parent_source_files=provenance, active299_changed=False,
        NPU_qualification_inherited=False, whole463_AR_qualified=False,
        Goodput=None, adopted=False)
    (P / 'SOURCE_ONLY311_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({key: value for key, value in review.items() if key != 'parent_source_files'}))


if __name__ == '__main__':
    main()
