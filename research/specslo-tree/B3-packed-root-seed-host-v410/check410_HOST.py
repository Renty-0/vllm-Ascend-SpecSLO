"""CPU contract comparison with actual FastV5 code and explicit HOST doubles.

The tiny arithmetic model is only a deterministic input generator. It cannot
qualify Ascend graphs, attention numerics, model outputs, or serving latency.
"""
import ast
import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
BASE = P.parent / 'tree-high-normal-cap4-native-families-v148'
sys.path.insert(0, str(BASE))


def main():
    import torch
    torch.set_num_threads(1)
    import root_seed410 as candidate
    import fast_tree_draft_v5 as fast

    @dataclass
    class Metadata:
        slot_mapping: object
        context_lens: object
        actual_seq_lengths_q: tuple
        sequence_lens: tuple
        use_fused_infer_attention: bool = False
        request_block_tables: object = None
        attention_mask: object = None

    def bucket(size, capacity):
        value = 1 << (size - 1).bit_length()
        return min(value, capacity)

    class ArithmeticModel:
        def __init__(self, engine):
            self.engine = engine
            self.weight = torch.sin(torch.arange(256).reshape(4, 64).float() * .07)

        def __call__(self, inputs, positions, metadata):
            hidden = torch.stack((inputs.float() * .015, positions.float() * .01,
                                  inputs.remainder(7).float() * .03,
                                  positions.remainder(5).float() * .02), dim=1)
            for slot, token, position in zip(metadata.slot_mapping.tolist(), inputs.tolist(), positions.tolist()):
                if slot >= 0:
                    self.engine.KV[slot] = (token, position)
            return hidden

        def compute_logits(self, hidden):
            return hidden @ self.weight

    class HostRunner:
        @staticmethod
        def _pad_inputs(inputs, positions, metadata, rows):
            extra = rows - len(inputs)
            assert extra >= 0
            if not extra:
                return inputs, positions, metadata
            return (torch.cat((inputs, inputs.new_zeros(extra))),
                    torch.cat((positions, positions.new_zeros(extra))),
                    replace(metadata,
                        slot_mapping=torch.cat((metadata.slot_mapping, metadata.slot_mapping.new_full((extra,), -1))),
                        context_lens=torch.cat((metadata.context_lens, metadata.context_lens.new_ones(extra))),
                        actual_seq_lengths_q=metadata.actual_seq_lengths_q + tuple(range(len(inputs)+1, rows+1)),
                        sequence_lens=metadata.sequence_lens + (1,) * extra))

        def __init__(self, engine):
            self.engine = engine
            self.model = engine.model
            self.max_graph_tokens = 128
            self.target_entries = {}
            self.hidden_calls = []
            self.chain_calls = []

        def run_tree_hidden(self, inputs, positions, metadata):
            self.hidden_calls.append(len(inputs))
            key = ('target-hidden:0|steps:1|paged', (len(inputs),))
            self.target_entries[key] = object()  # HOST object, never a Native entry.
            self.last_target_execution = SimpleNamespace(mode='replay', used_aclgraph=True)
            return self.model(inputs, positions, metadata)

        def run_draft_greedy(self, inputs, positions, metadata, vocabulary, **kwargs):
            self.chain_calls.append(len(positions))
            self.last_draft_execution = SimpleNamespace(mode='replay', used_aclgraph=True)
            self.last_draft_entry_key = ('HOST-double-only', len(positions))
            return fast.execute_top2(self, inputs, positions, metadata, vocabulary,
                                     return_confidence=kwargs['return_confidence'])

    class HostEngine:
        def __init__(self, n, repairs):
            self.is_draft = True
            self.device = 'cpu'
            self.draft_vocab_size = 64
            self.KV = {}
            self.config = SimpleNamespace(enable_prefix_caching=False, max_model_len=256,
                                          max_num_seqs=64, profile_host_decode_steps=0,
                                          kvcache_block_size=16)
            self.cache_allocation = SimpleNamespace(block_tables=[list(range(i*16, (i+1)*16)) for i in range(64)])
            self.prefix_cache = SimpleNamespace(num_blocks=1024)
            self._dynamic_tree_known_KV = {
                i: dict(safe_before=30+i-1, tokens={}) for i in range(n)}
            self.model = ArithmeticModel(self)
            self.graph_runner = HostRunner(self)
            self.reservations = []

        def _ensure_cache_capacity(self, ids, positions):
            self.reservations.append((list(ids), list(positions)))

        def _prepare_attention_metadata(self, ids, positions, use_fused_infer_attention):
            assert use_fused_infer_attention is False
            ends, lengths = [], []
            for i, (seq, position) in enumerate(zip(ids, positions, strict=True)):
                if i == len(ids)-1 or ids[i+1] != seq:
                    ends.append(i+1)
                    lengths.append(position+1)
            slots = [seq*256+position for seq, position in zip(ids, positions, strict=True)]
            return torch.tensor(positions), Metadata(torch.tensor(slots),
                torch.tensor([p+1 for p in positions]), tuple(ends), tuple(lengths))

    # Compile exactly the existing FastV5 nested callback without installing it.
    source = (BASE / 'fast_tree_draft_v5.py').read_text()
    install = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    original = next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'draft')
    namespace = dict(vars(fast), torch=torch, runner_cls=HostRunner,
                     native=SimpleNamespace(_next_linear_draft_graph_bucket=bucket,
                                            NativeACLGraphRunner=HostRunner),
                     original=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('unexpected reference fallback')))
    exec(compile(ast.get_source_segment(source, original), 'HOST410.actual-FastV5-body', 'exec'), namespace)
    reference = namespace['draft']
    native = SimpleNamespace(_next_linear_draft_graph_bucket=bucket, NativeACLGraphRunner=HostRunner)
    wrapper = candidate.build(native, reference)
    tests = 0
    for n in (1, 2, 3, 8, 15, 24, 25, 31, 32):
        for pattern in ('all3', 'mixed123'):
            for repair_mode in ('all', 'alternating', 'multi'):
                budgets = [3 if pattern == 'all3' else 1+i%3 for i in range(n)]
                budgets[-1] = 3
                plans = [SimpleNamespace(width=2, depth=4, prefix_len=30+i,
                         candidate_budget=b, parent_indices=torch.tensor([-1,0,1,2,3,0,1,2,3]))
                         for i,b in enumerate(budgets)]
                roots = [10+i%30 for i in range(n)]
                catchup = [([2+i%20+j for j in range(1+i%3)] + [roots[i]])
                           if repair_mode == 'multi' else
                           ([2+i%20, roots[i]] if repair_mode == 'all' or i%2 == 0 else [roots[i]])
                           for i in range(n)]
                left, right = HostEngine(n, None), HostEngine(n, None)
                for engine in (left, right):
                    for i, row in enumerate(catchup):
                        engine._dynamic_tree_known_KV[i]['safe_before'] = plans[i].prefix_len-len(row)+1
                kwargs = dict(committed_catchup_token_ids=catchup)
                expected = reference(left, plans, roots, list(range(n)), **kwargs)
                actual = wrapper(right, plans, roots, list(range(n)), **kwargs)
                assert actual['draft_token_ids'] == expected['draft_token_ids']
                assert all(len(row) == 8 and row[3] == row[7] == 0 for row in actual['draft_token_ids'])
                assert torch.equal(actual['draft_confidence'], expected['draft_confidence'])
                assert all(torch.equal(a,b) for a,b in zip(actual['node_confidences'], expected['node_confidences'], strict=True))
                assert right.KV == left.KV and right._dynamic_tree_known_KV == left._dynamic_tree_known_KV
                assert actual['num_draft_tokens'] == expected['num_draft_tokens'] == budgets
                assert actual['root_token_ids'] == expected['root_token_ids'] == roots
                assert torch.equal(actual['parent_indices'], expected['parent_indices'])
                assert left.graph_runner.chain_calls == [3] and right.graph_runner.chain_calls == [2]
                repair_stages = max(len(row)-1 for row in catchup)
                assert len(left.graph_runner.hidden_calls) == repair_stages
                assert len(right.graph_runner.hidden_calls) == 1
                assert actual['model_calls'] == 2 and expected['model_calls'] == 1+repair_stages
                assert actual['model_stages'] == 3
                assert right._dynamic_tree_draft_counts['generated_candidates'] == 6*n
                assert right.reservations == left.reservations
                tests += 1
    sentinel = object()
    delegated = candidate.build(native, lambda *args, **kwargs: sentinel)
    engine = HostEngine(1, None)
    plan = SimpleNamespace(width=2, depth=4, prefix_len=30, candidate_budget=4)
    assert delegated(engine, [plan], [10], [0]) is sentinel
    plan.candidate_budget = 3
    assert delegated(engine, [plan], [10], [0], committed_catchup_token_ids=[[10]]) is sentinel
    assert delegated(engine, [plan], [10], [0], eager_parent_sources={0: object()}) is sentinel
    plans = [SimpleNamespace(width=2, depth=4, prefix_len=30+i, candidate_budget=3) for i in range(32)]
    engine = HostEngine(32, None)
    for i in range(32):
        engine._dynamic_tree_known_KV[i]['safe_before'] = plans[i].prefix_len-5
    assert delegated(engine, plans, [10]*32, list(range(32)),
                     committed_catchup_token_ids=[[2,3,4,5,6,10]]*32) is sentinel
    assert engine._dynamic_tree_draft_counts['root_seed_capacity_fallback_calls'] == 1
    for roots, ids, catchup in [([10], [0], [[2,11]]), ([10,10], [0,0], [[2,10],[2,10]])]:
        try:
            delegated(HostEngine(2, None), [plan]*len(ids), roots, ids,
                      committed_catchup_token_ids=catchup)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError('HOST410 stale root or duplicate owner accepted')
    value = dict(status='PASS_HOST410_ACTUAL_FASTV5_CPU_CONTRACTS_ONLY', CPU_cases=tests,
                 actual_original_FastV5_callback_executed=True,
                 fake_arithmetic_model_explicit=True, actual_Native_engine_used=False,
                 padded_tokens_confidence_parent_indices_ledger_and_KV_contracts_equal=True,
                 repair_plus_three_steps_vs_packed_root_plus_two_steps=True,
                 reference_hidden_repair_steps_one_to_three=True,
                 candidate_two_graph_calls_in_HOST_double=True,
                 fallbacks_checked=4, invalid_inputs_rejected=2,
                 source_projection=candidate.descriptor(),
                 NPU_used=False, model_qualification=False, serving_installed=False,
                 Raw=None, Goodput=None)
    (P / 'HOST410_RESULT.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k:value[k] for k in ('status','CPU_cases','NPU_used')}))


if __name__ == '__main__':
    main()
