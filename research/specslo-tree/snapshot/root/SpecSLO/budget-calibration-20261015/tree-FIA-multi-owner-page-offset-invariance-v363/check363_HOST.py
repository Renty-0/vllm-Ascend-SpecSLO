"""Independent masked KV address check for every planned synthetic FIA case."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys


P = Path(__file__).resolve().parent


def main():
    output = P / 'HOST363_RESULT.json'
    assert not output.exists()
    import torch
    torch.set_num_threads(1)
    from probe363 import configurations, host_masks, description
    cases = configurations()
    assert len(cases) == 640
    count = dict(cases=0, checked_real_query_masks=0, ordered_KV_addresses=0, rejected_controls=0)
    for case in cases:
        packed, references, meta = host_masks(case)
        assert tuple(packed.shape) == (24, 1, case['mask_capacity'], 4096)
        assert len(meta['packed_table']) == 32 and all(0 <= page < 544 for page in meta['packed_table'])
        assert meta['owners'] == [owner for owner, q in enumerate(case['sizes']) for _ in range(q)]
        for group in (1, 7, 23):
            assert torch.equal(packed[0], packed[group])
            assert all(torch.equal(mask[0], mask[group]) for mask in references)
        for query, (owner, depth) in enumerate(zip(meta['owners'], meta['depths'], strict=True)):
            length = case['prefixes'][owner] + depth + 1
            expected = list(range(owner * 32 * 128, owner * 32 * 128 + length))
            reference_visible = torch.nonzero(~references[owner][0, 0, query], as_tuple=False).flatten().tolist()
            reference_addresses = [meta['reference_tables'][owner][pos // 128] * 128 + pos % 128
                                   for pos in reference_visible]
            actual_visible = torch.nonzero(~packed[0, 0, query], as_tuple=False).flatten().tolist()
            actual_addresses = [meta['packed_table'][pos // 128] * 128 + pos % 128 for pos in actual_visible]
            assert reference_addresses == actual_addresses == expected
            assert actual_visible == list(range(meta['offsets'][owner], meta['offsets'][owner] + length))
            assert actual_visible[-1] < case['extent']
            count['checked_real_query_masks'] += 1
            count['ordered_KV_addresses'] += len(expected)
        for query in range(sum(case['sizes']), 5):
            assert torch.nonzero(~packed[0, 0, query], as_tuple=False).flatten().tolist() == [0]
        count['cases'] += 1
    for fault in ('wrong_pages', 'truncated_extent', 'source_beyond2560', 'wrong_heads', 'wrong_mask'):
        case = copy.deepcopy(next(c for c in cases if sum(c['pages']) > 20))
        if fault == 'wrong_pages':
            case['pages'] = tuple(v + 1 for v in case['pages'])
        elif fault == 'truncated_extent':
            case['extent'] = 2560
        elif fault == 'source_beyond2560':
            case['prefixes'] = tuple([3000] * len(case['sizes']))
        elif fault == 'wrong_heads':
            case['Q_heads'] = 15
        else:
            case['mask_capacity'] = 5
        try:
            description(case)
        except ValueError:
            count['rejected_controls'] += 1
        else:
            raise AssertionError('invalid planned FIA case accepted: ' + fault)
    source = (P / 'probe363.py').read_text()
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == 'torch_npu.npu_fused_infer_attention_score.out']
    assert len(calls) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    assert keywords['input_layout'] == "'TND'" and keywords['block_size'] == '128'
    assert keywords['sparse_mode'] == '1' and keywords['actual_seq_lengths'] == 'list(range(5, 121, 5))'
    assert 'torch_npu' not in sys.modules
    value = dict(status='PASS_HOST_all640_synthetic_FIA_metadata_and_ordered_KV_addresses',
                 counts=count, ABA_rows_planned=1920,
                 original_FIA_TND_block128_sparse1_call_preserved=True,
                 actual_TP3_heads_16over2_and24over3=True,
                 whole_output_ABA_restoration_required=True,
                 compared_to_original_FULL2560_and_same_extent_unshifted_controls=True,
                 NPU_used=False, numerical_offset_invariance_qualified=False,
                 model_qualified=False, Raw=None, Goodput=None,
                 probe_sha256=hashlib.sha256(source.encode()).hexdigest())
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
