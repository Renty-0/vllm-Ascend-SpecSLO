"""New source/CPU gate and separately explicit actual component adoption gate."""
import ast
import hashlib
import json
from pathlib import Path
from source_snapshot_v132 import P,sha,collect
CPU=P/'CPU_RESULT_V132.json';FREEZE=P/'PREPARED_SOURCE_HASHES_V132.json'
ACTIONS=('split192-candidate-full463',)
NEW_HOST_TEST_COUNT=10  # Only the new V146 full-process boundary tests; parent suites remain SHA-only.


def external_CPU_files():
    from source_fix_contract_v148 import parent_evidence
    fix,frozen,cpu=parent_evidence()
    fp,cp=Path(fix['freeze']),Path(fix['CPU'])
    rows={**frozen['files'],**frozen['external_files'],str(fp):sha(fp),str(cp):sha(cp)}
    import json
    cfg=json.loads((P/'CANDIDATE_V132.json').read_text())
    proof=cfg['component_actual_evidence']['variable_B4_local']
    if not isinstance(proof,dict):raise RuntimeError('V146 prior actual variable-B4 component is pending')
    rp,ap=Path(proof['receipt']),Path(proof['external_audit'])
    audit=json.loads(ap.read_text());parent=rp.parent
    pf=parent/'PREPARED_SOURCE_HASHES_V144.json';pc=parent/'CPU_RESULT_V144.json'
    inherited=json.loads(pf.read_text())
    rows.update(inherited['files']);rows.update(inherited['external_files'])
    rows.update(fix['retained_V143_mapping_evidence']);rows.update(fix['V146_aborted_before_model_evidence'])
    rows.update(audit['original_strict_audit']['raw_checkpoint_sha256'])
    rows.update(audit['original_strict_audit']['raw_report_sha256'])
    for path in (rp,ap,pf,pc,parent/'PREPARED_SOURCE_ARCHIVE_V144.tar.gz'):rows[str(path)]=sha(path)
    return rows


def source_contract():
    from source_fix_contract_v148 import require_parent_and_delta
    v146=require_parent_and_delta()
    parent=json.loads((P/'PARENT_V132.json').read_text());old=Path(parent['directory'])
    changes=v146['changed_cloned_files']
    from typed_source_contract_v132 import contract as typed_contract
    typed=typed_contract()
    from live_decode_patch import build_live_runtime
    from runtime_adapter_v132 import source_contract as runtime_contract
    source=build_live_runtime();compile(source,'<V81-bounded-live-mixed-prefill>','exec')
    from global_reads_v101 import unbound_global_reads
    added=[P/name for name in ('runtime_adapter_v132.py','split_shape_v132.py','candidate_controls_v132.py','candidate_phases_v132.py','candidate_runtime_v132.py','audit_candidate_v132.py','import_origins_v132.py','split_budget_v132.py','normal_admission_v140.py','gain_state_v140.py','exception_evidence_v140.py','no_mixed_prefill_v143.py','full_process_v146.py','variable_actual_gate_v146.py','scope_v146.py','dual_bank_v146.py','draft_v146.py','draft_observer_v146.py','target_v146.py','actor_v146.py','audit_v146.py')]
    unbound=unbound_global_reads(added)
    if unbound:raise RuntimeError('V132 new source has unbound global reads: '+repr(unbound))
    for name in ('launch_candidate_v132.sh','candidate-startup-body.sh'):
        text=(P/name).read_text()
        if text.count('task_root='+str(P)+'\n')!=1:raise RuntimeError('V132 actual launcher root differs')
    if (P/'candidate-startup-body.sh').read_text().split('export PYTHONPATH=',1)[1].split('cd "$task_root"',1)[0]!=(old/'combined-startup-body.sh').read_text().split('export PYTHONPATH=',1)[1].split('cd "$task_root"',1)[0]:raise RuntimeError('V132 model/runtime flags changed')
    from local_qualification_scope import SCOPES,fixture_for,load_fixture
    load_fixture()
    fixtures=[fixture_for(scope) for scope in SCOPES]
    phases=[phase for scope in SCOPES for phase in scope.phases()]
    if len(fixtures)!=21 or len(phases)!=26 or any(len(f['trees'])!=f['scope'].target_requests or len(f['prompt']['token_ids'])!=2301 or any(len(row['token_ids'])<2 for row in f['trees']) for f in fixtures):raise RuntimeError('V132 actual pre-model S35 fixture/phase domain differs')
    from factory_scope_v132 import collect as factory_scope
    factories=factory_scope()
    from production_probe_v101 import source_scope
    scopes={str(n):source_scope(n) for n in (65,463)}
    return dict(schema='V132.single-candidate-runtime-source-contract.v1',actual_component_qualification_inferred=False,
        preparation_epoch='V148',high_normal_cap4_full_process=v146,changed_ancestor_files=changes,typed_runtime=runtime_contract(),typed_allocator=typed,full_greedy_pipeline_factory_scope=factories,pre_model_actual_input_contract=dict(S35_scopes=21,S35_phases=26,source_workload=scopes,actual_fixture_helpers_executed=True),generated_live_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_fresh_decode_policy_Nativeclock_and_target_math_unchanged=True,original_ancestor_suite_repeated=False,
        component_device_qualification_inherited=False,global_factory_equivalence_proved=False)


def require_local_launch(action,expected_identity=None):
    if action not in ACTIONS:raise RuntimeError('V132 unknown local action')
    if not CPU.is_file() or not FREEZE.is_file():raise RuntimeError('V132 own actual HOST and freeze required')
    cpu=json.loads(CPU.read_text());frozen=json.loads(FREEZE.read_text());local=collect();external=external_CPU_files()
    if (cpu['schema']!='V148.actual-native-inventory-family-fix-HOST.v1' or cpu['preparation_epoch']!='V148' or cpu['status']!='passed' or cpu['input_hashes']!=cpu['input_hashes_after'] or cpu['input_hashes']!=local
        or frozen['files']!=local or frozen['external_files']!=external or frozen['CPU_sha256']!=sha(CPU)
        or cpu['external_input_hashes']!=cpu['external_input_hashes_after'] or cpu['external_input_hashes']!=external
        or cpu['parent9_actual_HOST_proof_verified'] is not True or cpu['tests_run']!=NEW_HOST_TEST_COUNT or cpu['parent7_actual_HOST_proof_verified'] is not True or cpu['parent13_actual_HOST_proof_verified'] is not True or cpu['parent8_actual_HOST_proof_verified'] is not True
        or cpu['parent23_actual_HOST_proof_verified'] is not True or cpu['NPU_initialized'] is not False):raise RuntimeError('V132 current source/CPU/freeze binding differs')
    from component_gates_v132 import require_actual_components
    component=require_actual_components()
    identity=dict(CPU_sha256=sha(CPU),freeze_sha256=sha(FREEZE))
    if expected_identity is not None and expected_identity!=identity:raise RuntimeError('V132 worker source identity differs')
    return dict(identity=identity,source_contract=source_contract(),component_actual_evidence=component,formal_Goodput=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--require-local',choices=ACTIONS,required=True);a=p.parse_args();print(json.dumps(require_local_launch(a.require_local)))
