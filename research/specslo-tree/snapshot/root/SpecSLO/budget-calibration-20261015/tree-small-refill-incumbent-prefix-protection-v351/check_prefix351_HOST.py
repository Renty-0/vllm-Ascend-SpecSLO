"""Actual prefix hash function on CPU catches all protected-owner mutations."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B/'tree-high-normal-cap4-native-families-v148'),
               str(B/'tree-active-window32-16-local-v160')]
import runtime351
import subset351


def main():
    import torch
    key = torch.arange(32, dtype=torch.float32).reshape(8,4,1,1)
    value = key.clone()+100
    engine = SimpleNamespace(device=torch.device('cpu'),
        config=SimpleNamespace(kvcache_block_size=4),
        cache_allocation=SimpleNamespace(block_tables=[[i] for i in range(8)]),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(
            key_cache=key, value_cache=value))]))
    single = SimpleNamespace(n=1,contexts=[[1,2,3]],reference_contexts=[[1,2,3]],
                             incumbent=2,incumbent_tokens=[8,9])
    selected = SimpleNamespace(ids=[0])
    try: subset351.original.guard_rest(engine,single,selected)
    except (ValueError,RuntimeError) as error:
        assert 'non-empty' in str(error)
    else: raise AssertionError('old singleton empty-history regression absent')
    before = subset351.guard_rest(engine,single,selected)
    key[2,0].add_(1)
    assert subset351.guard_rest(engine,single,selected) != before
    key[2,0].sub_(1)
    assert subset351.guard_rest(engine,single,selected) == before
    multiple = SimpleNamespace(n=3,contexts=[[1,2,3]]*3,
        reference_contexts=[[1,2,3]]*3,incumbent=6,incumbent_tokens=[8,9])
    selected = SimpleNamespace(ids=[0,2])
    before = subset351.guard_rest(engine,multiple,selected)
    for owner in (1,4,6):
        value[owner,0].add_(1)
        assert subset351.guard_rest(engine,multiple,selected) != before
        value[owner,0].sub_(1)
        assert subset351.guard_rest(engine,multiple,selected) == before
    assert subset351.namespace()['finish'].__globals__['guard_rest'] is subset351.guard_rest
    assert 'torch_npu' not in sys.modules
    result = dict(status='passed_actual_prefix_protection_CPU',NPU_used=False,
        old_empty_history_error_reproduced=True, actual_frozen_prefix_hash_function_used=True,
        singleton_incumbent_KV_mutation_detected=True,
        nonselected_candidate_reference_and_incumbent_mutations_detected=True,
        all_restored_prefixes_equal=True, same_finish_guard_bound=True,
        model_executed=False, numerical_NPU_qualification=False)
    (P/'HOST351_PREFIX_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
