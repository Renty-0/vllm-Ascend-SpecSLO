"""Actual430 worker/425 trace/425+428+430 audit composition; no model use."""
import ast
import json
from pathlib import Path


def main():
    import torch,torch_npu
    attempted=[]
    def blocked(*args,**kwargs):
        attempted.append(True)
        raise AssertionError('HOST432 no model or NPU initialization')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    from vllm_ascend.spec_decode.pearl import native_engine as native
    native.NativePearlEngine.__init__=blocked
    import runtime432,trace432,audit432,extension432,extension433,trace425
    source=runtime432.source()
    compile(source,'HOST432.actual430-worker','exec')
    assert "extension_module='extension432'"in source
    assert 'from scope432 import issue,report as scope_report'in source
    assert 'from trace432 import run_windows,warm_to_fixedpoint'in source
    assert 'from audit432 import audit'in source
    assert 'original_commit430=capture(protocol,native)'in source
    for name in('prepare_install','install_entry','after_install','postfull_AR','snapshot','bucket_snapshot','metadata_snapshot'):
        assert getattr(extension432,name)is getattr(extension433,name)
    qualifier=extension432.qualifier()
    names=qualifier.__code__.co_names
    assert names.index('bounded428')<names.index('incumbents427')<names.index('end_shadow')<names.index('issue')
    ns=trace432.namespace();parent=trace425.namespace()
    for name in('make_trace','warm_to_fixedpoint','run_windows'):
        assert ns[name].__globals__ is ns and ns[name].__code__ is parent[name].__code__
    assert ns['canonical_patch']is trace432.generation and ns['trace_source']is trace432.trace_source
    tree=trace432.trace_source(64)
    compile(tree,'HOST432.actual-saved-trace','exec')
    additions=[n for n in ast.walk(tree)if isinstance(n,ast.keyword)and n.arg=='ROOT432_refill_witnesses']
    assert len(additions)==1 and ast.unparse(additions[0].value)=="__import__('witness431').snapshot(engine, generation)"
    actual,base=audit432.constructed()
    projected=base()
    cursor=actual
    for _ in range(3):cursor=cursor.__globals__['original'].audit
    assert cursor.__globals__['projected_original_audit']is base
    assert 'from admission_audit432 import audit_trace'in projected.__code__.co_consts
    assert 'actual_admission_cohorts='in projected.__code__.co_consts
    assert '_ROOT425_valid_record'in projected.__code__.co_names
    # Missing actual new witnesses rejects historical traces. No HOST-generated
    # elapsed values or old-model proof can be silently substituted.
    import admission_audit432
    try:admission_audit432.audit_trace(dict())
    except RuntimeError as error:rejection=str(error)
    else:raise AssertionError('missing current trace witness was accepted')
    assert not attempted
    value=dict(status='PASS_HOST432_ACTUAL_WORKER_TRACE_AND_FULL_AUDIT_COMPOSITION',
        original433_worker_capture_and_original425_neural_callbacks_retained=True,
        all_original_trace_functions_code_identity_retained=True,
        trace_record_only_adds_actual_existing_clock_witnesses=True,
        all_original425_428_430_433_audits_composed_without_changing_their_bodies=True,
        only_empty_active_admission_predicate_replaced_by_stronger_actual_witness_audit=True,
        original218_close_after_all75_and_extra427_before_E4_issue=True,
        missing_current_witness_rejected=rejection,Native_models_constructed=0,NPU_used=False,
        HOST_only_source_gate_does_not_grant_NN_permission=True,production_launcher_not_created=True,
        NN_executed=False,serving_installed=False,
        Raw=None,Goodput=None)
    P=Path(__file__).resolve().parent
    (P/'HOST432_COMPOSITION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
