"""Verify optional fifth KV-only step consumes the computed fourth token."""
import hashlib
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent


def main():
    output = P / 'HOST364_FINAL_KV_RESULT.json'
    assert not output.exists()
    import torch
    torch.set_num_threads(1)
    from spine364 import execute, decode
    from check364_HOST import ModelDouble
    cases = 0
    for n in (1, 4, 12, 32):
        vocabulary = 151936
        roots = torch.arange(n, dtype=torch.long)
        positions = [torch.full((n,), 127 + step, dtype=torch.long) for step in range(5)]
        metas = [types.SimpleNamespace(slot_mapping=torch.arange(n)) for _ in range(5)]
        reference = ModelDouble(torch, vocabulary)
        baseline = execute(reference, roots, positions[:4], metas[:4], vocabulary)
        model = ModelDouble(torch, vocabulary)
        with_leaf = execute(model, roots, positions, metas, vocabulary, final_kv_only=True)
        assert torch.equal(baseline, with_leaf)
        tokens, probabilities = decode(with_leaf, vocabulary)
        assert len(reference.inputs) == 4 and len(model.inputs) == 5
        assert torch.equal(model.inputs[4], tokens[:, 3].long())
        assert all(torch.equal(a, b) for a, b in zip(reference.inputs, model.inputs[:4], strict=True))
        assert probabilities.dtype == torch.float32
        cases += 1
    assert 'torch_npu' not in sys.modules
    value = dict(status='PASS_HOST_optional_fifth_KV_only_step_and_unchanged_top2_output',
                 cases=cases, last_input_is_actual_fourth_primary_token=True,
                 no_fifth_LM_head_or_new_candidate=True,
                 no_probability_rounding=True, model_quantization=False,
                 NPU_used=False, actual_KV_neural_math_qualified=False,
                 runtime_installed=False, Raw=None, Goodput=None,
                 source_sha256=hashlib.sha256((P / 'spine364.py').read_bytes()).hexdigest())
    with output.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
