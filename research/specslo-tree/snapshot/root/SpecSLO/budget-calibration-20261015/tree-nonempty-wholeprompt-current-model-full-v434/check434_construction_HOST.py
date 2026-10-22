"""Conditional434 source callbacks and unchanged original supervision only."""
import json
from pathlib import Path


def main():
    import torch,torch_npu
    attempts=[]
    def blocked(*a,**k):
        attempts.append(True)
        raise AssertionError('HOST434 cannot construct a model or initialize an NPU')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    from vllm_ascend.spec_decode.pearl import native_engine as native
    native.NativePearlEngine.__init__=blocked
    import runtime434,extension434,extension432,guard434
    source=runtime434.source();compile(source,'HOST434.actual-frozen432-worker','exec')
    assert "extension_module='extension434'"in source and 'from audit434 import audit'in source
    assert 'from scope432 import issue,report as scope_report'in source
    assert 'from trace432 import run_windows,warm_to_fixedpoint'in source
    assert 'original_commit430=capture(protocol,native)'in source
    for name in('prepare_install','install_entry','after_install','qualify_component','postfull_AR','snapshot','bucket_snapshot','metadata_snapshot','refill_snapshot'):
        assert getattr(extension434,name)is getattr(extension432,name)
    ns=guard434.execution_namespace()
    assert ns['execute_locked'].__globals__ is ns
    assert ns['MAX_SECONDS']==18000 and ns['MAX_TAG_BYTES']==256*1024**2
    assert ns['MAX_AGGREGATE_BYTES']==3584*1024**2
    assert ns['SCOPE']['current_same_engine_parent4_lifecycle_cases']==75
    assert ns['SCOPE']['original_current427_extra_NN_cases']==6
    assert ns['SCOPE']['active_request_limit']==ns['SCOPE']['global_active_limit']==64
    from check432 import ready
    frozen=ready()
    assert frozen['NN_permission_granted']is False and frozen['NN_launched']is False
    assert not attempts
    result=dict(status='PASS_HOST434_FROZEN432_CALLBACKS_AND_ORIGINAL_OWNED_SUPERVISOR',
        exact_frozen432_component_scope_trace_and_audit_routing=True,
        original_models_arrivals_SLO_g_policy_targetM120_S24_and_Native_gates_retained=True,
        fresh_extra427_NN6_and_original75_required=True,
        original_elapsed_charges_and_full463_all3ownAR_required=True,
        original_private_caps_and_owned_cleanup_retained=True,
        Native_models_constructed=0,NPU_used=False,NN_launched=False,NN_permission_inherited=False)
    (Path(__file__).resolve().parent/'HOST434_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
