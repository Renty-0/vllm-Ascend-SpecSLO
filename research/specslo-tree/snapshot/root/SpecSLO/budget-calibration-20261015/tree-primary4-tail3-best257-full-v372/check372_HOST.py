"""HOST only: exact runtime/environment deltas and original neural-auditor controls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
P257=BASE/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'
P371=BASE/'tree-primary4-tail3-eager-allN-device-v371'

def main():
    output=HERE/'HOST372_RESULT.json'
    assert not output.exists()
    sys.path[:0]=[str(HERE),str(P257),str(P371)]
    import runtime257, runtime372
    source=(BASE/'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text()
    old,_=runtime257.project_source(source)
    new,_=runtime372.project_source(source)
    expected=old.replace("extension_module='extension257'","extension_module='extension372'")
    expected=expected.replace("require(extension_module=='extension257'","require(extension_module=='extension372'")
    expected=expected.replace('from check257 import ready','from check372 import ready')
    expected=expected.replace('from scope257 import issue,report as scope_report','from scope372 import issue,report as scope_report')
    expected=expected.replace('from trace257 import run_windows,warm_to_fixedpoint','from trace372 import run_windows,warm_to_fixedpoint')
    expected=expected.replace('from audit257 import audit','from audit372 import audit')
    expected=expected.replace("     report['Native_final']=","     report['ROOT372_draft_tail']=extension.tail_snapshot(engine)\n     report['Native_final']=")
    assert new==expected
    oldtrace=(P257/'trace257.py').read_text()
    assert (HERE/'trace372.py').read_text()==oldtrace.replace('from scope257 import report as standalone_report','from scope372 import report as standalone_report')
    def model_env(s):
        return [line for line in s.splitlines() if line.startswith(('export VLLM_','unset VLLM_'))]
    assert model_env((P257/'startup257.sh').read_text())==model_env((HERE/'startup372.sh').read_text())
    for p in HERE.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    from qualify372 import project_body,SOURCE_SHA
    compile(project_body(),'ROOT372-HOST-original371-projection','exec')
    from probe371 import configurations
    # Re-use only fixture construction from the prior HOST auditor test.
    # No model/graph is created, no Native flag is assigned, and all temporary
    # fictional neural data is removed with this TemporaryDirectory.
    text=(P371/'check371_HOST.py').read_text()
    tree=ast.parse(text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='rows'for t in n.targets))
    end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.FunctionDef)and n.name=='run')
    namespace=dict(configurations=configurations,hashlib=hashlib,statistics=statistics)
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(fn.body[start:end]),type_ignores=[])),__file__+'::fictional-HOST-auditor-data','exec'),namespace)
    fixture=namespace['fixture']
    fixture.update(source_sha256=SOURCE_SHA,costs=[],median_seconds={})
    from audit_draft372 import audit as fresh_audit
    negatives=[]
    with tempfile.TemporaryDirectory(prefix='HOST-fixture372-',dir=HERE)as tmp:
        folder=Path(tmp)
        body=folder/'rank0-original371-neural-body.json'
        proofpath=folder/'rank0-tail3-same-instance-proof.json'
        def save(value,change=None):
            body.write_text(json.dumps(value))
            proof=dict(cases=306,fresh_same_model_instance=True,private_entries=64,
                all_new_private_entries_naturally_qualified=True,costs_measured=False,
                native_qualification_flag_assignments=False,
                original_warm_twozero_seal_required_for_all_entries=True,
                original_body_result_sha256=hashlib.sha256(body.read_bytes()).hexdigest())
            if change:proof.update(change)
            proofpath.write_text(json.dumps(proof))
        save(fixture)
        assert fresh_audit(folder)['cases']==306
        for label in ('probability','promotion','missing_N','invented_cost','stale_body_hash','unqualified_graphs'):
            value=copy.deepcopy(fixture);change=None
            if label=='probability':value['math_rows'][0]['all8_FP32_probabilities_bitwise']=False
            if label=='promotion':value['math_rows'][0]['eager_promotion_map_exact']=False
            if label=='missing_N':value['math_rows'].pop()
            if label=='invented_cost':value['costs']=[dict(seconds=.01)]
            if label=='stale_body_hash':change=dict(original_body_result_sha256='0'*64)
            if label=='unqualified_graphs':change=dict(private_entries=63)
            save(value,change)
            try:fresh_audit(folder)
            except RuntimeError:negatives.append(label)
            else:raise AssertionError('corrupt same-instance proof accepted: '+label)
    import guard372
    guard372.EXPECTED_FREEZE='f'*64
    argv=guard372.launch_argv('HOST-no-launch')
    assert 'SPECSLO_V257_ROOT_FREEZE_SHA=b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'in argv
    assert 'SPECSLO_V372_ROOT_FREEZE_SHA='+'f'*64 in argv
    ns=guard372.execution_namespace()
    assert ns['SCOPE']['fresh_same_instance_draft_cases']==306 and ns['SCOPE']['fresh_same_instance_canonical_math_cases']==227
    assert ns['execute_locked'].__globals__['actual_audit']is guard372.actual_audit
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST_full372_runtime_ENV_delta_and_fresh_neural_auditor_controls',
        original_worker_AST_bytes_preserved_outside_declared_extensions=True,
        original_VLLM_ENV_lines_equal=True,original_target227_and_all_original_components_required=True,
        fresh_draft306_required=True,rejected_corruptions=negatives,fictional_auditor_data_only=True,
        model_loaded=False,NPU_used=False,model_qualified=False,Raw=None,Goodput=None)
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
