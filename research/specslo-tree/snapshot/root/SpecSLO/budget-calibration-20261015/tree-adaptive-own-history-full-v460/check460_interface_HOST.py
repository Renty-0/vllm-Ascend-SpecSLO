"""Check actual full worker callbacks and source projections before freezing."""
from pathlib import Path
from types import SimpleNamespace
import ast,importlib.abc,inspect,json,sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST460 interface check forbids model backends')
sys.meta_path.insert(0,NoModel())
from host_env460 import setup
setup()
import runtime460,extension460,extension438,scope460,trace460,audit460,audit_reference460,reference460,guard460
for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
source=runtime460.source();tree=ast.parse(source);compile(tree,'HOST460.actual-full-worker','exec')
attrs=sorted({n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='extension'})
assert len(attrs)==13 and all(callable(getattr(extension460,name,None))for name in attrs),attrs
calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='extension']
for call in calls:
    assert not any(isinstance(n,ast.Starred)for n in call.args)and all(n.arg is not None for n in call.keywords)
    inspect.signature(getattr(extension460,call.func.attr)).bind(*[object()for _ in call.args],**{n.arg:object()for n in call.keywords})
for name in('prepare_install','install_entry','after_install','preflight_snapshot','cache_snapshot','bucket_snapshot','metadata_snapshot','refill_snapshot'):
    assert getattr(extension460,name)is getattr(extension438,name)
sentinel=object();assert extension460.preflight_snapshot(SimpleNamespace(_root433_preflight_lifetime=sentinel))is sentinel
assert trace460.namespace()['standalone_report']is scope460.report
assert extension460.projected_qualifier().__name__=='qualify_component'
assert extension460.qualifier().__name__=='qualify_component'
assert reference460.constructed_run().__name__=='run'
assert audit460.base_audit().__name__=='audit'and audit460.constructed().__name__=='audit'
assert audit_reference460.constructed().__name__=='audit'
ns=guard460.execution_namespace();assert ns['execute_locked'].__globals__ is ns
assert(ns['MAX_SECONDS'],ns['MAX_TAG_BYTES'],ns['MAX_AGGREGATE_BYTES'])==(18000,256*1024**2,3584*1024**2)
# Invalid records and missing current-model proofs must never grant scope.
from bucket460 import valid_record
for m in(80,120):
    record=dict(M=m,uniform_KV_bucket425=dict(enabled=True,extent=512,original_sequence_lens=[17]),fixed_KV_extent=[512]*(m//5),attention_segments=m//5)
    assert valid_record(record)
    record['fixed_KV_extent'][0]=2560;assert not valid_record(record)
try:scope460.validate_proof(SimpleNamespace())
except RuntimeError:pass
else:raise AssertionError('HOST460 absent actual current-model proof accepted')
value=dict(status='PASS_HOST460_ACTUAL_FULL_WORKER_API_SOURCE_PROJECTIONS_AND_NEGATIVE_CONTROLS',
    referenced_attributes=attrs,all_actual_call_signatures_bound=len(calls),
    exact_original_preflight_snapshot_exported=True,all_full_source_projections_constructed=True,
    original_full_supervisor_limits_retained=True,invalid_physical_record_rejected=True,
    absent_current_model_proof_rejected=True,global_active_limit=64,
    current693_and_original227_681_75_6_and_full463_own_AR_still_required=True,
    Native_constructor_calls=0,NPU_used=False,NN_permission=False,Raw=None,Goodput=None)
with(P/'HOST460_COMPLETE_INTERFACE_RESULT.json').open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(value))
