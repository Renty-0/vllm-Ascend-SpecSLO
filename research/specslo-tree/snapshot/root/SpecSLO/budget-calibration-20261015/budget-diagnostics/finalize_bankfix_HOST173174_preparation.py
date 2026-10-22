"""Pure HOST source-state preparation, not freeze or device qualification."""
import importlib.util,json,pathlib,sys,time
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
new=int(sys.argv[1]);old=167 if new==173 else 170
name='tree-causal80-real16-bankfix-full-v173' if new==173 else 'tree-normal-cap5-real16-bankfix-full-v174'
P=B/name;G=D/f'v{new}-root-execution';sys.path.insert(0,str(P))
base=__import__(f'base{new}');check=__import__(f'check{new}');host=__import__(f'host{new}');sys.meta_path.insert(0,host.Firewall())
sha=base.sha;cpu=base.read(P/f'CPU_RESULT{new}.json')
prepared=dict(schema=f'V{new}.actual-one-bank-HOST-retained-source-preparation.v1',files=check.local_files(),external_files=check.external_files(),CPU_sha256=sha(P/f'CPU_RESULT{new}.json'),actual_HOST_attempts={str(p):sha(p) for p in sorted(P.glob(f'HOST_ATTEMPT{new}_*.json'))},old_frozen_parent=check.parent_frozen(),device_qualification=False,frozen=False)
(P/f'PREPARED_HOST{new}.json').write_text(json.dumps(prepared,indent=2,sort_keys=True)+'\n')
spec=importlib.util.spec_from_file_location(f'_V{new}_actual_guard_source_state_HOST',G/f'guard{new}.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
t0=time.perf_counter();state=guard.source_state(archive=False);inputs=guard.guard_inputs()
proof=dict(schema=f'V{new}.actual-bankfix-source-state-HOST.v1',status='passed',actual_source_state_called=True,elapsed_seconds=time.perf_counter()-t0,files=state['local'],selected_external=state['external'],retained_HOST_evidence=state['evidence'],CPU_sha256=state['CPU_sha256'],guard_inputs=inputs,original_parent_identity=state['parent_ready']['identity'],old_bank_parent=check.parent_frozen(),Torch_imported='torch' in sys.modules,NPU_used=False,freeze_or_helper_or_supervisor_executed=False,actual_interpreter=sys.executable,Python_version=sys.version)
assert proof['Torch_imported'] is False
proofpath=D/f'V{new}_ACTUAL_COLD_GUARD_SOURCE_STATE_HOST.json';proofpath.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
ready=dict(schema=f'V{new}.real16-bank-index-fix-source-ready.v1',status='ready-for-source-review-not-frozen',candidate=str(P),files=state['local'],selected_external=state['external'],guard_inputs=inputs,retained_HOST_evidence=state['evidence'],CPU=dict(path=str(P/f'CPU_RESULT{new}.json'),sha256=state['CPU_sha256'],tests_run=1,elapsed_seconds=cpu['elapsed_seconds']),cold_guard_source_state_HOST=dict(path=str(proofpath),sha256=sha(proofpath)),frozen_parent=check.parent_frozen(),projection=base.read(P/f'SOURCE_PORT_{new}.json'),
 fix='actual generated DualBank.prefix_ready i%32 -> i%16; both banks retain32 actual independent owner histories',
 actual_new_HOST=dict(original_NativePrefixCache=True,actual_build_bank_prefill_all32=True,actual_prefix_ready_guard_source=True,reference_owners_16_through31_checked=True,original_frozen_method_IndexError_reproduced=True,wrong_lease_rejected=True,wrong_source_history_rejected=True,prefix_evidence_and_hidden_are_HOST_stubs=True,model_KV_qualification=False),
 required_new_process_scope='unchanged parent own numerical/real protocol components; current Native natural gates; original warm/twozero/seal/full32',
 old_HOST_not_rerun=True,old_sources_modified=False,device_qualification_inherited=False,Torch_imported=False,NPU_used=False,freeze_created=False,formal_Goodput_measured=False,actual_interpreter=sys.executable,Python_version=sys.version)
path=D/f'V{new}_REAL16_BANKFIX_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json';path.write_text(json.dumps(ready,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(epoch=new,ready=str(path),sha256=sha(path),CPU_sha256=state['CPU_sha256'],local=len(state['local']),external=len(state['external']),guards=len(inputs),HOST_elapsed=cpu['elapsed_seconds'],cold_guard_HOST_sha256=sha(proofpath),Torch_imported=False)))
