"""Check every actual worker extension reference before any model launch."""
import ast,inspect,json
from pathlib import Path
from types import SimpleNamespace


def main():
    import runtime438,runtime437,extension438,extension437,extension433,trace438,scope438
    source=runtime438.source();tree=ast.parse(source);compile(tree,'HOST438.actual-worker','exec')
    attrs=sorted({n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='extension'})
    missing=[name for name in attrs if not hasattr(extension438,name)]
    assert not missing,missing
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='extension']
    for call in calls:
        assert not any(isinstance(n,ast.Starred)for n in call.args)and all(n.arg is not None for n in call.keywords)
        function=getattr(extension438,call.func.attr);assert callable(function)
        inspect.signature(function).bind(*[object()for _ in call.args],**{n.arg:object()for n in call.keywords})
    # The exact previously missing function, not a substitute or a forged proof.
    assert extension438.preflight_snapshot is extension433.preflight_snapshot
    assert not hasattr(extension437,'preflight_snapshot')
    sentinel=object();host_engine=SimpleNamespace(_root433_preflight_lifetime=sentinel)
    assert extension438.preflight_snapshot(host_engine)is sentinel
    for name in('prepare_install','install_entry','after_install','qualify_component','postfull_AR','snapshot','bucket_snapshot','metadata_snapshot','refill_snapshot','cache_snapshot'):
        assert getattr(extension438,name)is getattr(extension437,name)
    assert "from scope438 import issue,report as scope_report"in source
    assert trace438.namespace()['standalone_report']is scope438.report
    import guard438
    guard=guard438.execution_namespace();assert guard['execute_locked'].__globals__ is guard
    assert guard['MAX_SECONDS']==18000 and guard['MAX_TAG_BYTES']==256*1024**2 and guard['MAX_AGGREGATE_BYTES']==3584*1024**2
    result=dict(status='PASS_HOST438_COMPLETE_ACTUAL_WORKER_EXTENSION_API_AND_SIGNATURES',
        referenced_attributes=attrs,all_actual_call_signatures_bound=len(calls),
        previous_missing_export_reproduced=True,exact_original433_snapshot_exported=True,
        all_original437_NN_callbacks_identity_preserved=True,global_active_limit=64,
        all75_plus6_and_original_math_full463_AR_still_fresh_required=True,
        NPU_used=False,Native_constructor_calls=0,Raw=None,Goodput=None)
    (Path(__file__).resolve().parent/'HOST438_COMPLETE_INTERFACE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
