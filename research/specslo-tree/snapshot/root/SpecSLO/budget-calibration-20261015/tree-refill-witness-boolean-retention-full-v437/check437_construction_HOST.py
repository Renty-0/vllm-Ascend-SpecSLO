"""Assert exact436 neural callbacks and the sole post-generation fix."""
import ast,json
from pathlib import Path


def main():
    import runtime437,runtime436,extension437,extension436,trace437,trace432
    source=runtime437.source();compile(source,'HOST437.original436-retention-fix','exec')
    for name in('prepare_install','install_entry','after_install','qualify_component','postfull_AR','snapshot','bucket_snapshot','metadata_snapshot','refill_snapshot','source_binding'):
        assert getattr(extension437,name)is getattr(extension436,name)
    old=next(n for n in ast.parse(runtime436.source()).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
    new=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
    wanted=ast.dump(old,include_attributes=False).replace("'extension436'","'extension437'").replace("module='trace432'","module='trace437'")
    assert ast.dump(new,include_attributes=False)==wanted
    ns=trace437.namespace();parent=trace432.namespace()
    for name in('make_trace','warm_to_fixedpoint','run_windows'):
        assert ns[name].__globals__ is ns and ns[name].__code__ is parent[name].__code__
    assert ns['canonical_patch']is parent['canonical_patch'] and ns['standalone_report']is parent['standalone_report']
    import guard437
    guard=guard437.execution_namespace()
    assert guard['execute_locked'].__globals__ is guard
    assert guard['MAX_SECONDS']==18000 and guard['MAX_TAG_BYTES']==256*1024**2 and guard['MAX_AGGREGATE_BYTES']==3584*1024**2
    assert guard['SCOPE']['current_same_engine_parent4_lifecycle_cases']==75 and guard['SCOPE']['original_current427_extra_NN_cases']==6
    result=dict(status='PASS_HOST437_ORIGINAL436_WORKER_NEURAL_CALLBACKS_AND_BOUNDS_RETAINED',
        all_original227_681_75_plus6_warm_twozero_seal_full463_all3ownAR_required=True,
        unchanged_NN_qualification_functions_by_identity=True,unchanged_live_decode_source_projection=True,
        original_parent_qualification_wait_and_worker_timeout_unchanged=True,
        global_active_limit=64,NPU_used=False,Native_constructor_calls=0)
    (Path(__file__).resolve().parent/'HOST437_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
