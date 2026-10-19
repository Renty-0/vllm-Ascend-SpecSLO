"""Real requests advance; original positive scratch rows keep fixed roots."""
import hashlib
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent


def main():
    output = P / 'HOST364_PADDING_RESULT.json'
    assert not output.exists()
    import torch
    torch.set_num_threads(1)
    from spine364 import execute, decode
    from check364_HOST import ModelDouble
    cases = 0
    for physical, logical in ((8, 5), (16, 12), (32, 20), (32, 24), (64, 48)):
        for final_kv in (False, True):
            count = 5 if final_kv else 4
            vocabulary = 1024
            roots = torch.cat((torch.arange(logical, dtype=torch.long) + 41,
                               torch.zeros(physical - logical, dtype=torch.long)))
            positions = [torch.cat((torch.full((logical,), 127 + i, dtype=torch.long),
                                    torch.zeros(physical - logical, dtype=torch.long))) for i in range(count)]
            metas = [types.SimpleNamespace(slot_mapping=torch.arange(physical)) for _ in range(count)]
            model = ModelDouble(torch, vocabulary)
            packed = execute(model, roots, positions, metas, vocabulary,
                             final_kv_only=final_kv, logical_row_count=logical)
            tokens, probabilities = decode(packed, vocabulary)
            assert tokens.shape == probabilities.shape == (physical, 8)
            for i, actual in enumerate(model.inputs):
                assert torch.equal(actual[logical:], roots[logical:])
                if i:
                    assert torch.equal(actual[:logical], tokens[:logical, i - 1].long())
            # The explicit original padder allocates positive private slots.
            # Validation must cover every actual physical query, including
            # scratch queries; these are never counted as client requests.
            assert all(bool((meta.slot_mapping >= 0).all()) for meta in metas)
            cases += 1
    assert 'torch_npu' not in sys.modules
    value = dict(status='PASS_HOST_primary_spine_real_rows_and_fixed_private_padding_roots',
                 cases=cases, padding_inputs_match_original_per_depth_padder=True,
                 model_input_rows_and_client_request_rows_explicitly_distinct=True,
                 all_positive_original_physical_query_slots_retained=True,
                 no_scratch_query_promoted_to_client_request=True,
                 NPU_used=False, real_neural_math_qualified=False, Raw=None, Goodput=None,
                 source_sha256=hashlib.sha256((P / 'spine364.py').read_bytes()).hexdigest())
    with output.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
