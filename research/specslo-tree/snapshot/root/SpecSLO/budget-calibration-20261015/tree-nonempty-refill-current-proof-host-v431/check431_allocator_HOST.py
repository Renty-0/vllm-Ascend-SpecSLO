"""Execute actual427 Bank construction with the real Native CPU allocator.

Only small HOST page-table tensors are allocated; there is no model, hidden
state or KV tensor and this does not qualify neural isolation.
"""
import gzip
import json
from pathlib import Path
import sys


def main():
    import torch, torch_npu
    def blocked(*a, **k):
        raise AssertionError('HOST431 allocator cannot construct models or initialize an NPU')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from vllm_ascend.spec_decode.pearl.native_cache import NativePrefixCache
    from config209 import make_config
    native.NativePearlEngine.__init__ = blocked
    P = Path(__file__).resolve().parent
    B = P.parent
    sys.path.insert(0,str(B/'tree-nonempty-prefill-incumbent-KV-protocol-host-v427'))
    from incumbents427 import Bank
    path = B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/normal24-global64-derived-controls-root-v405-full463-ownAR-v1/windows/window64/full463/rank1-full463-actual-trace.json.gz'
    trace = json.loads(gzip.decompress(path.read_bytes()))
    rows = [dict(token_ids=list(p),request_id=str(i)) for i,p in enumerate(trace['tokenized_inputs'][:65])]
    results = []
    for role in ('draft','target'):
        for n in (2,32):
            engine = object.__new__(native.NativePearlEngine)
            engine.device = torch.device('cpu')
            engine.rank = int(role == 'target')
            engine.is_draft = role == 'draft'
            engine.config = make_config()
            assert engine._cache_sequence_capacity == 463
            assert engine._cache_storage_sequence_capacity == 527
            engine.prefix_cache = NativePrefixCache(num_blocks=544, blocks_per_sequence=32, block_size=128)
            engine.cache_allocation = None
            engine.cache_block_tables = None
            bank = Bank(engine,rows,n)
            lease = bank.guard()
            assert lease['storage_rows'] == 527
            assert lease['real_active_owners'] == n+32 <= 64
            assert lease['planned_pages_with_scratch'] == lease['actual_unique_pages']+64 <= 544
            assert tuple(engine.cache_block_tables.shape) == (527,32)
            assert engine.cache_block_tables.cpu().tolist() == engine.cache_allocation.block_tables
            oldpages = [p for owner in bank.incumbent_ids427 for p in bank.tables[owner] if p>=0]
            newpages = [p for owner in [*bank.ids,*bank.reference_ids] for p in bank.tables[owner] if p>=0]
            assert len(oldpages) == len(set(oldpages)) and not set(oldpages).intersection(newpages)
            engine._release_cache()
            assert engine.cache_allocation is None and engine.cache_block_tables is None
            assert all(block.ref_count==0 for block in engine.prefix_cache._blocks)
            results.append(dict(role=role,new_N=n,actual_real_owners=n+32,
                allocated_pages=lease['actual_unique_pages'],scratch_reserved=64,
                old_owners=bank.incumbent_ids427,all_old_new_reference_pages_disjoint=True,
                actual_device_page_table_on_CPU_matches_original_allocator=True,
                actual_original_release_all_reference_counts_zero=True))
    value = dict(status='PASS_HOST431_ACTUAL_NATIVE_ALLOCATOR_AND_COMPLETE427_BANK_CONSTRUCTION',
        cases=results, only_small_CPU_page_tables_allocated=True,
        Native_models_constructed=0,NPU_used=False,neural_KV_tensors_allocated=0,
        actual427_NN_qualification=False,serving_installed=False,Raw=None,Goodput=None)
    (P/'HOST431_ALLOCATOR_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':
    main()
