"""Check planned layouts and diagnostic-auditor rejection without a backend."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

P = Path(__file__).resolve().parent


def main():
    from probe335 import configurations
    import guard335
    configs = configurations()
    assert len(configs) == len(set(configs)) == 288
    for width, groups, mode, capacity, precise, prefix in configs:
        assert 0 <= prefix < 2560 <= 4096 and width <= capacity
        ends = list(range(width, width*groups+1, width))
        assert len(ends) == groups and ends[-1] == width*groups
    rows = []
    for width, groups, mode, capacity, precise, prefix in configs:
        count = groups if mode == 'roots' else width*groups
        for phase in ('A1', 'B', 'A2'):
            rows.append(dict(width=width, groups=groups, mode=mode, mask_capacity=capacity,
                inner_precise=precise, prefix=prefix, phase=phase, M=width*groups,
                compared_queries=count, identical_Q_K_V_visibility_and_page_table=True,
                different_query_indices=[], max_abs_by_query=[0.]*count,
                all_compared_raw16_equal=True,
                whole_output_restored=True if phase == 'A2' else None))
    fixture = dict(schema='ROOT335.synthetic-FIA-group-query-invariance.v1', seed=335,
        source_sha256=hashlib.sha256((P/'probe335.py').read_bytes()).hexdigest(), rows=rows,
        models_loaded=False, model_qualified=False, performance=False, Goodput=None)
    rejected = 0
    with tempfile.TemporaryDirectory(prefix='specslo-335-HOST-') as directory:
        path = Path(directory)/'result.json'
        path.write_text(json.dumps(fixture))
        result, _, _ = guard335.actual_audit(directory)
        assert result['rows'] == 864 and len(result['groups']) == 12
        assert all(v == dict(cases=72, noninvariant=0) for v in result['groups'].values())
        for kind in ('raw_difference', 'physical_M', 'ABA_restore', 'source_pin', 'nonfinite', 'model_claim', 'missing_case'):
            bad = copy.deepcopy(fixture)
            if kind == 'raw_difference':
                bad['rows'][0]['different_query_indices'] = [1]
            elif kind == 'physical_M':
                bad['rows'][0]['M'] += 1
            elif kind == 'ABA_restore':
                bad['rows'][2]['whole_output_restored'] = False
            elif kind == 'source_pin':
                bad['source_sha256'] = '0'*64
            elif kind == 'nonfinite':
                bad['rows'][0]['max_abs_by_query'][0] = float('nan')
            elif kind == 'model_claim':
                bad['models_loaded'] = True
            else:
                bad['rows'].pop()
            path.write_text(json.dumps(bad))
            try:
                guard335.actual_audit(directory)
            except RuntimeError:
                rejected += 1
            else:
                raise AssertionError('corrupt synthetic evidence accepted: '+kind)
    namespace = guard335.execution_namespace()
    assert namespace['MAX_SECONDS'] == 300 and namespace['SCOPE']['models_loaded'] is False
    assert namespace['execute_locked'].__globals__ is namespace
    argv = guard335.launch_argv('HOST-only-not-launched')
    assert '/bin/bash' in argv and str(P/'startup335.sh') in argv
    assert not any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') for name in sys.modules)
    result = dict(status='passed_HOST_only', planned_configurations=len(configs),
        planned_ABA_observations=len(rows), rejected_corrupt_records=rejected,
        original_guard_functions_compiled=True, fake_fixture_is_not_device_evidence=True,
        Torch_imported=False, NPU_used=False, launcher_executed=False)
    with (P/'HOST335_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
