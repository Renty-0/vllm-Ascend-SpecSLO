"""Actual Native checker plus dependent CPU model double; no device use."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent
SOURCE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')


class ModelDouble:
    def __init__(self, torch, vocabulary):
        self.torch = torch
        self.vocabulary = vocabulary
        self.inputs = []

    def __call__(self, ids, positions, metadata):
        self.inputs.append(ids.clone())
        return (ids.float() + positions.float())[:, None]

    def compute_logits(self, hidden):
        # Every primary choice changes the next step, and scores include
        # nontrivial float32 softmax values that must survive byte-exactly.
        columns = self.torch.arange(self.vocabulary).float()[None, :]
        centers = (hidden + 3) % self.vocabulary
        return -(columns - centers).square() * .125


def main():
    output = P / 'HOST364_RESULT.json'
    assert not output.exists()
    import torch
    torch.set_num_threads(1)
    from spine364 import execute, decode
    tree = ast.parse(SOURCE.read_text())
    node = copy.deepcopy(next(node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == '_draft_outputs_match'))
    node.decorator_list = []
    namespace = dict(torch=torch)
    exec(compile('from __future__ import annotations\n' + ast.unparse(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))),
        str(SOURCE) + '::unchanged-Native-output-checker', 'exec'), namespace)
    matcher = namespace['_draft_outputs_match']
    cases = raw_probabilities = 0
    for n in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
        for vocabulary in (32, 151936):
            model = ModelDouble(torch, vocabulary)
            roots = torch.arange(n, dtype=torch.long) % vocabulary
            positions = [torch.full((n,), 127 + step, dtype=torch.long) for step in range(4)]
            metas = [types.SimpleNamespace(slot_mapping=torch.arange(n)) for _ in range(4)]
            packed = execute(model, roots, positions, metas, vocabulary)
            actual_tokens, actual_probabilities = decode(packed, vocabulary)
            reference_model = ModelDouble(torch, vocabulary)
            current = roots
            token_pairs, probability_pairs = [], []
            for position, metadata in zip(positions, metas, strict=True):
                logits = reference_model.compute_logits(reference_model(current, position, metadata))
                values, tokens = torch.softmax(logits.float(), dim=-1).topk(2, dim=-1)
                token_pairs.append(tokens)
                probability_pairs.append(values)
                current = tokens[:, 0].contiguous()
            reference_tokens = torch.cat((torch.stack(token_pairs, dim=1)[:, :, 0],
                                          torch.stack(token_pairs, dim=1)[:, :, 1]), dim=1)
            reference_probabilities = torch.cat((torch.stack(probability_pairs, dim=1)[:, :, 0],
                                                 torch.stack(probability_pairs, dim=1)[:, :, 1]), dim=1)
            assert torch.equal(actual_tokens.long(), reference_tokens)
            assert torch.equal(actual_probabilities.view(torch.int32), reference_probabilities.view(torch.int32))
            assert all(torch.equal(a, b) for a, b in zip(model.inputs, reference_model.inputs, strict=True))
            assert matcher(packed, packed.clone(), metas, n)
            altered = packed.clone()
            altered[0, 0, 1] ^= 1
            assert not matcher(packed, altered, metas, n)
            invalid = list(metas)
            invalid[0] = types.SimpleNamespace(slot_mapping=torch.full((n,), -1))
            assert not matcher(packed, packed.clone(), invalid, n)
            cases += 1
            raw_probabilities += actual_probabilities.numel()
    assert 'torch_npu' not in sys.modules
    value = dict(status='PASS_HOST_four_step_top2_chain_lossless_codec_and_original_Native_checker',
                 dependent_model_cases=cases, exact_float32_probabilities_checked=raw_probabilities,
                 original_primary_and_sibling_order_preserved=True,
                 subsequent_model_inputs_are_actual_preceding_primary_tokens=True,
                 original_Native_output_checker_unchanged=True,
                 one_changed_probability_bit_rejected=True, invalid_real_KV_slot_rejected=True,
                 no_probability_rounding=True, model_quantization=False,
                 NPU_used=False, runner_installed=False, real_draft_neural_math_qualified=False,
                 full463_qualified=False, Raw=None, Goodput=None,
                 source_sha256=hashlib.sha256((P / 'spine364.py').read_bytes()).hexdigest(),
                 Native_checker_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    with output.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
