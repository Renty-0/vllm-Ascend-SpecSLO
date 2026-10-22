"""Frozen actual425 installation with own428 routing; no Native constructor."""
import json,tempfile
from pathlib import Path


def main():
    import torch,torch_npu
    attempted=[]
    def blocked(*a,**k):attempted.append(True);raise AssertionError('HOST428 no model or NPU init')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    from check428 import ready
    ready()
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    parent=require_ready();spec=dict(action=PARENT_ACTION,identity=parent['parent']['identity'])
    B=Path(__file__).resolve().parent.parent
    with activate_local_diagnostic(spec):
        import extension428
        output=Path(tempfile.mkdtemp(prefix='HOST428-preconstructor-',dir=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'))
        entry,controls=extension428.prepare_install(output)
        from vllm_ascend.spec_decode.pearl import native_engine as native,native_graph as graph
        native.NativePearlEngine.__init__=blocked
        extension428.install_entry(entry,native)
        import tail_hidden_observer,live_decode_patch
        tail_hidden_observer.install(graph)
        extension428.after_install(native,graph,live_decode_patch)
        origins=extension428.loaded_origins()
        from factory425 import current
        installed=current(native)
        import fast_tree_draft_v5 as canonical
        assert installed['fallback'].__globals__ is vars(canonical)
        assert canonical.execute_top2.__code__.co_filename.endswith('tail_hidden_observer.py')
        from bounded428 import actor
        import target428
        actual=actor()
        assert actual.__globals__['run_case'].__globals__['target'] is target428
        assert not attempted
        value=dict(status='PASS_HOST428_FROZEN_ACTUAL_INSTALL_REFERENCE_ROUTING_BEFORE_MODELS',
            actual_origins=origins,original425_installer_backend_and_tail_observer_retained=True,
            actual419_case_targets_own428_reference=True,Native_constructor_calls=0,
            NPU_init_calls=0,NPU_used=False,NN_qualification_inherited=False)
        f=B/'budget-diagnostics/ROOT428_REAL_PRECONSTRUCTOR_RESULT.json'
        with f.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
        print(json.dumps(dict(status=value['status'],NPU_used=False)))


if __name__=='__main__':main()
