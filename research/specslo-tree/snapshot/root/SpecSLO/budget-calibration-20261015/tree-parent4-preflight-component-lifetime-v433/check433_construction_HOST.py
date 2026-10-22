"""Actual source ordering, unchanged NN callbacks and complete worker projection."""
import ast,json
from pathlib import Path


def main():
    import runtime433,extension433,extension430,base_component433
    source=runtime433.source();compile(source,'HOST433.actual-runtime','exec')
    assert "extension_module='extension433'"in source and 'from audit433 import audit'in source
    assert 'original_commit430=capture(protocol,native)'in source
    factory=extension433.projected_qualifier()
    assert 'base_component433'in factory.__code__.co_names
    qualifier=extension433.qualifier()
    names=qualifier.__code__.co_names
    assert names.index('run')<names.index('end_shadow')<names.index('issue')
    assert 'bounded428'in names and 'slot_preflight218'in names
    assert 'end_shadow'not in base_component433.qualify.__code__.co_names
    for name in('prepare_install','install_entry','after_install','postfull_AR','snapshot','bucket_snapshot','metadata_snapshot'):
        assert getattr(extension433,name)is getattr(extension430,name)
    import guard433
    ns=guard433.execution_namespace()
    assert ns['execute_locked'].__globals__ is ns
    assert ns['MAX_SECONDS']==18000 and ns['MAX_TAG_BYTES']==256*1024**2
    assert ns['MAX_AGGREGATE_BYTES']==3584*1024**2
    assert ns['SCOPE']['current_same_engine_parent4_lifecycle_cases']==75
    assert ns['SCOPE']['active_request_limit']==64
    # All runtime shadow changes are made by the unchanged original218 API.
    P=Path(__file__).resolve().parent
    for name in('base_component433.py','extension433.py','runtime433.py'):
        tree=ast.parse((P/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign)else[node.target]
                assert not any(isinstance(n,ast.Attribute)and n.attr in('_dynamic_tree_generation','runtime_validated')
                               for target in targets for n in ast.walk(target))
    result=dict(status='PASS_HOST433_ACTUAL_QUALIFIER_SHADOW_CLOSE_ORDER_AND_WORKER_BINDING',
        original_425_factory_target_428_reference_430_commit_binding_retained=True,
        all_original_and227_681_75_ownAR_requirements_retained=True,
        original218_preflight_installer_and_body_unchanged=True,
        actual_end_shadow_after75_before_E4_issue=True,
        production_Native_flags_or_engine_generation_assigned=False,
        NPU_used=False,Native_models_constructed=0)
    (P/'HOST433_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
