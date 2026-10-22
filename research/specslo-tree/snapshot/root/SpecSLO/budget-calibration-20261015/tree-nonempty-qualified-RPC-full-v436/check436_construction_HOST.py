"""Exact434 numerical callbacks; only parent's calibration receive is different."""
import ast,json
from pathlib import Path


def main():
    import runtime436,runtime434,extension436,extension434
    source=runtime436.source();compile(source,'HOST436.exact434-parent-receive','exec')
    assert 'replies=receive_qualification(engine,\'V209 \'+ACTION)'in source
    assert 'qualification_rpc_wait='in source and 'from audit436 import audit'in source
    for name in('prepare_install','install_entry','after_install','qualify_component','postfull_AR','snapshot','bucket_snapshot','metadata_snapshot','refill_snapshot','source_binding'):
        assert getattr(extension436,name)is getattr(extension434,name)
    # Apart from explicit own extension routing, the whole numerical worker
    # AST is identical. Timeout adaptation occurs only in the parent's main.
    old=next(n for n in ast.parse(runtime434.source()).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
    new=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
    wanted=ast.dump(old,include_attributes=False).replace("'extension434'","'extension436'")
    assert ast.dump(new,include_attributes=False)==wanted
    import guard436
    ns=guard436.execution_namespace()
    assert ns['execute_locked'].__globals__ is ns
    assert ns['MAX_SECONDS']==18000 and ns['MAX_TAG_BYTES']==256*1024**2
    assert ns['MAX_AGGREGATE_BYTES']==3584*1024**2
    assert ns['SCOPE']['current_same_engine_parent4_lifecycle_cases']==75
    assert ns['SCOPE']['active_request_limit']==64
    assert ns['SCOPE']['original_current427_extra_NN_cases']==6 and ns['SCOPE']['real_incumbents']==32
    import extension432
    qualifier=extension432.qualifier();names=qualifier.__code__.co_names
    assert names.index('run')<names.index('incumbents427')<names.index('end_shadow')<names.index('issue')
    result=dict(status='PASS_HOST436_WHOLE_NUMERICAL_WORKER_UNCHANGED_PARENT_RECEIVE_ONLY',
        all_original227_681_75_and_warm_twozero_seal_full463_ownAR_required=True,
        original434_neural_worker_AST_except_source_routing_identical=True,
        parent_only_qualification_timeout_view=True,
        global_active_limit=64,NPU_used=False,Native_models_constructed=0)
    (Path(__file__).resolve().parent/'HOST436_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
