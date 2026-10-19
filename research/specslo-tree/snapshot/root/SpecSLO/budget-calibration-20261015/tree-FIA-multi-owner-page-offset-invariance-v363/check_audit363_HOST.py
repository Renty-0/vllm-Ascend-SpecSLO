"""Synthetic audit fixtures verify positive, negative and corrupt evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile


P = Path(__file__).resolve().parent


def main():
    output = P / 'HOST363_AUDIT_RESULT.json'
    assert not output.exists() and 'torch' not in sys.modules
    from probe363 import configurations
    from audit363 import audit
    rows = []
    for ordinal, case in enumerate(configurations()):
        for phase in ('A1', 'B', 'A2'):
            rows.append(dict(case_ordinal=ordinal, case=case, phase=phase,
                compared_queries=24 * sum(case['sizes']), physical_M=120, groups=24, width=5,
                identical_ordered_real_KV_and_original_query=True,
                all_compared_raw16_equal=True, same_extent_page_offset_raw16_equal=True,
                unshifted_extent_vs_FULL2560_raw16_equal=True, different_query_indices=[],
                max_abs_by_query=[0.0] * (24 * sum(case['sizes'])),
                whole_output_restored=True if phase == 'A2' else None))
    fixture = dict(schema='ROOT363.synthetic-FIA-multi-owner-page-offset.v1', seed=363, rows=rows,
                   source_sha256=hashlib.sha256((P / 'probe363.py').read_bytes()).hexdigest(),
                   complete_cases=640, visible_runtime_ids='0,1,2,3', logical_device=1,
                   physical_cards_requested=[2, 5, 6, 7], models_loaded=False, model_qualified=False,
                   performance=False, Raw=None, Goodput=None, adopted=False)
    with tempfile.TemporaryDirectory(prefix='HOST363-audit-', dir=P) as folder:
        root = Path(folder)

        def write(value, *, checkpoint=None):
            (root / 'result.json').write_text(json.dumps(value))
            with (root / 'completed-cases363.jsonl').open('w') as saved:
                for row in value['rows'] if checkpoint is None else checkpoint:
                    saved.write(json.dumps(row) + '\n')

        write(fixture)
        result, _, _ = audit(root)
        assert result['all_match_original_FULL2560'] is result['all_page_offsets_raw16_invariant'] is True
        negative = copy.deepcopy(fixture)
        negative['rows'][0].update(all_compared_raw16_equal=False,
            same_extent_page_offset_raw16_equal=False, different_query_indices=[0])
        negative['rows'][0]['max_abs_by_query'][0] = .125
        write(negative)
        result, _, _ = audit(root)
        assert result['all_match_original_FULL2560'] is result['all_page_offsets_raw16_invariant'] is False
        assert result['model_qualified'] is result['adopted'] is False
        rejected = []
        for fault in ('missing_case', 'wrong_query_count', 'invalid_ABA', 'wrong_source',
                      'false_model_qualification', 'impossible_pairwise_equality', 'checkpoint_disagreement'):
            bad = copy.deepcopy(fixture)
            checkpoint = None
            if fault == 'missing_case':
                bad['rows'].pop()
            elif fault == 'wrong_query_count':
                bad['rows'][0]['compared_queries'] = 1
            elif fault == 'invalid_ABA':
                bad['rows'][2]['whole_output_restored'] = False
            elif fault == 'wrong_source':
                bad['source_sha256'] = '0' * 64
            elif fault == 'false_model_qualification':
                bad['model_qualified'] = True
            elif fault == 'impossible_pairwise_equality':
                bad['rows'][0]['same_extent_page_offset_raw16_equal'] = False
            else:
                checkpoint = bad['rows'][:-1]
            write(bad, checkpoint=checkpoint)
            try:
                audit(root)
            except RuntimeError:
                rejected.append(fault)
            else:
                raise AssertionError('corrupt synthetic evidence accepted: ' + fault)
    result = dict(status='PASS_HOST_synthetic363_auditor_positive_negative_and_seven_corruptions',
                  synthetic_case_count=640, synthetic_ABA_rows=1920,
                  valid_negative_diagnostic_retained_without_model_qualification=True,
                  rejected_controls=rejected, synthetic_fixtures_only=True,
                  NPU_used=False, numerical_offset_invariance_qualified=False,
                  models_loaded=False, Raw=None, Goodput=None)
    with output.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
