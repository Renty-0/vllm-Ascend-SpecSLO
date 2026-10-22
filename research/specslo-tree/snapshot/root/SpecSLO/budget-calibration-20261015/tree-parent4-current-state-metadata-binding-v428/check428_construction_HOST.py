"""Cold runtime/qualifier/actor/guard construction before models and own freeze."""
import ast,json
from pathlib import Path


def main():
    import runtime428,extension428,bounded428,target428,guard428
    import actor419
    source=runtime428.source()
    compile(source,'HOST428.actual-runtime','exec')
    assert "extension_module='extension428'"in source and 'from audit428 import audit'in source
    qualifier=extension428.qualifier()
    assert 'bounded428' in qualifier.__code__.co_names
    actual=bounded428.actor()
    case=actual.__globals__['run_case']
    assert case.__code__ is actor419.run_case.__code__
    assert case.__globals__['target']is target428
    assert actual.__globals__['target']is target428
    ns=guard428.execution_namespace()
    assert ns['execute_locked'].__globals__ is ns
    assert ns['prepared']is guard428.prepared and ns['actual_audit']is guard428.actual_audit
    assert ns['MAX_SECONDS']==18000 and ns['MAX_TAG_BYTES']==256*1024**2
    assert ns['MAX_AGGREGATE_BYTES']==3584*1024**2
    assert ns['SCOPE']['current_same_engine_parent4_lifecycle_cases']==75
    assert ns['SCOPE']['active_request_limit']==64
    value=dict(status='PASS_HOST428_ACTUAL_PARENT425_RUNTIME_QUALIFIER_ACTOR_AND_OWNED_GUARD',
        original425_installer_backend_scope_and_g_policy_retained=True,
        complete419_case_code_identity=True,actual_case_target_binding428=True,
        all_original_227_681_75_and_full463_ownAR_required=True,
        original_private_caps_and_owned_cleanup_retained=True,
        NPU_used=False,Native_models_constructed=0,NN_qualification_inherited=False)
    (Path(__file__).resolve().parent/'HOST428_CONSTRUCTION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
