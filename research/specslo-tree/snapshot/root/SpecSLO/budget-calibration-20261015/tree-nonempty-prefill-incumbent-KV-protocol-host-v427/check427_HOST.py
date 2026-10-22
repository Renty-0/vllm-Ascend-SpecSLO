"""Check proof source inverses and actual retained workload resource fixtures."""
import gzip
import hashlib
import inspect
import json
from pathlib import Path


def main():
    import torch
    import torch_npu
    def blocked(*a,**k):raise AssertionError('HOST427 cannot initialize a Native model or NPU')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    import incumbents427 as candidate
    ns,proof=candidate.namespace()
    assert ns['Bank']is candidate.Bank
    assert ns['run_case'].__globals__ is ns and ns['run'].__globals__ is ns
    assert ns['audit']is candidate.audit
    assert all(proof[k]['entire_source_inverse']is True for k in ('case','run'))
    assert '_root427_incumbent_prefill_actual' in ns['run'].__code__.co_names
    assert '_v209_actual_prefill' not in ns['run'].__code__.co_names
    B=Path(__file__).resolve().parent.parent
    root=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/normal24-global64-derived-controls-root-v405-full463-ownAR-v1'
    path=root/'windows/window64/full463/rank1-full463-actual-trace.json.gz'
    actual=json.loads(gzip.decompress(path.read_bytes()))
    rows=[dict(token_ids=list(p))for p in actual['tokenized_inputs'][:65]]
    old=root/'multiowner-eager-prefill-own-current-process/rank0-N32-pass0.json.gz'
    retained=json.loads(gzip.decompress(old.read_bytes()))
    values=retained['candidate_metadata']['inputs'];cursor=0
    for i,p in enumerate(retained['prompt_requests']):
        assert rows[i]['token_ids']==values[cursor:cursor+p['length']]
        cursor+=p['length']
    fixtures=[]
    for n in (2,32):
        f=candidate.fixture(rows,n)
        assert f['real_active_owners']==n+32<=64 and f['planned_pages_with_scratch']<=544
        assert len(set(f['old_ids']))==32
        assert not set(f['old_ids'])&set(range(2*n))
        assert len(set(f['old_source_indices']))==32
        assert all(p==rows[i]['token_ids']for i,p in zip(f['old_source_indices'],f['old_tokens'],strict=True))
        fixtures.append(dict(new_N=n,real_active=n+32,shadow_reference_owners=n,
            old_source_indices=f['old_source_indices'],old_prompt_lengths=list(map(len,f['old_tokens'])),
            planned_pages_with_scratch=f['planned_pages_with_scratch']))
    source=inspect.getsource(candidate.original.run_case)
    assert 'candidate,kv,record='in source and 'rh=lane('in source
    value=dict(status='PASS_HOST427_PROPOSED_MULTIINCUMBENT_PROOF_SOURCE_AND_ACTUAL_FIXTURE_BOUNDS',
        source_projections=proof,actual_workload_trace=str(path),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original209_N32_prompt_source_exactly_matched=True,finite_fixtures=fixtures,
        original_base209_proof_pointer_not_overwritten=True,
        original_new_prefix_reference_root_next_and_twozero_checks_retained_in_source=True,
        proposed_extra_guard='All allocated old KV pages, not only committed prefix; actual old draft ledger',
        Native_models_constructed=0,NPU_used=False,NN_executed=False,production_permission=False,
        serving_installed=False,Raw=None,Goodput=None)
    p=Path(__file__).resolve().parent
    (p/'HOST427_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
