"""Corrupt HOST doubles to check current proof, entry and callback provenance."""
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import host_paths383

P=Path(__file__).parent


def main():
    import scope383 as scope
    draft=object();mc2=object()
    native=ModuleType('native_engine');native.NativePearlEngine=SimpleNamespace(draft_tree_forward=draft)
    model=ModuleType('native_model');model.matmul_allreduce_add_rmsnorm_or_fallback=mc2
    pearl=ModuleType('vllm_ascend.spec_decode.pearl');pearl.native_engine=native;pearl.native_model=model
    saved={}
    for name,module in [('vllm_ascend',ModuleType('vllm_ascend')),('vllm_ascend.spec_decode',ModuleType('vllm_ascend.spec_decode')),('vllm_ascend.spec_decode.pearl',pearl)]:
        saved[name]=sys.modules.get(name);sys.modules[name]=module
    def engine():
        entries={}
        for m,s in [(80,16),(120,24)]:
            for cap in (10,11):
                shape=(((s,1,cap,4096),(s,32)),)
                key=(f'target-greedy:151936|steps:1|tree-fia-full:exact-length-task-update|buffers:{shape}',(m,))
                entries[key]=SimpleNamespace(runtime_validated=True)
        return SimpleNamespace(is_draft=False,draft_vocab_size=151936,
            cache_allocation=None,cache_block_tables=None,_best257_draft383=draft,
            _branch_tail_state=SimpleNamespace(adaptive377_enabled=True,uniform_KV_bucket377_enabled=True),
            model=SimpleNamespace(forward=object(),lm_head=SimpleNamespace(_project_local=object())),
            graph_runner=SimpleNamespace(target_entries=entries,disabled_entry_keys=set(),graph_cache_sealed=False),
            _hybrid383_actual_proof=dict(cases=693,same_model_instance=True,
                original227_FULL_math_preserved=True,banks=[{}, {}, {}]))
    def rejects(callback):
        try:callback()
        except (RuntimeError,AttributeError):return
        raise AssertionError('ROOT383 stale/incomplete HOST permission accepted')
    old_issue,old_current=scope.original.issue,scope.original.current
    try:
        inherited=SimpleNamespace(require=lambda e:frozenset())
        scope.original.issue=lambda e,component_proof:inherited
        scope.original.current=lambda e:dict(original=True)
        active=engine();permit=scope.issue(active,component_proof=object())
        assert permit.require(active)==frozenset()
        rejects(lambda:permit.require(engine()));rejects(lambda:scope.issue(active,component_proof=object()))
        for target,field in [(active.model,'forward'),(active.model.lm_head,'_project_local')]:
            before=getattr(target,field);setattr(target,field,object())
            rejects(lambda:permit.require(active));setattr(target,field,before)
        model.matmul_allreduce_add_rmsnorm_or_fallback=object()
        rejects(lambda:permit.require(active));model.matmul_allreduce_add_rmsnorm_or_fallback=mc2
        for field in ['adaptive377_enabled','uniform_KV_bucket377_enabled']:
            setattr(active._branch_tail_state,field,False);rejects(lambda:permit.require(active))
            setattr(active._branch_tail_state,field,True)
        proof=active._hybrid383_actual_proof;active._hybrid383_actual_proof=dict(proof)
        rejects(lambda:permit.require(active));active._hybrid383_actual_proof=proof
        key=next(iter(active.graph_runner.target_entries));entry=active.graph_runner.target_entries[key]
        active.graph_runner.target_entries[key]=SimpleNamespace(runtime_validated=True)
        rejects(lambda:permit.require(active));active.graph_runner.target_entries[key]=entry
        entry.runtime_validated=False;rejects(lambda:permit.require(active));entry.runtime_validated=True
        for field,value in [('cases',692),('same_model_instance',False),('original227_FULL_math_preserved',False),('banks',[])]:
            damaged=engine();damaged._hybrid383_actual_proof[field]=value
            rejects(lambda:scope.issue(damaged,component_proof=object()))
        damaged=engine();damaged.cache_allocation=object();rejects(lambda:scope.issue(damaged,component_proof=object()))
        scope.original.issue=lambda e,component_proof:(_ for _ in ()).throw(RuntimeError('original scope rejects'))
        rejects(lambda:scope.issue(engine(),component_proof=object()))
        assert permit.require(active)==frozenset()
    finally:
        scope.original.issue,scope.original.current=old_issue,old_current
        for name,value in saved.items():
            if value is None:del sys.modules[name]
            else:sys.modules[name]=value
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST383_SCOPE_PROVENANCE_FOUR_ENTRIES_AND_CALLBACK_CORRUPTIONS',
        NPU_used=False,uses_HOST_doubles=True,actual_Native_flags_assigned=False,
        original_scope_failure_retained=True,neural_qualification_inherited=False,production_qualified=False)
    (P/'HOST383_SCOPE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
