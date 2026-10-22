"""Record audited controlled-history observations without numerical qualification."""
import json,hashlib,datetime,os
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'canonical-ar-prefix-history-isolation-diagnostic-v2'
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_bytes())
receipt=P/'receipt.json';audit=D/'CANONICAL_PREFIX_HISTORY_V2_INDEPENDENT_TERMINAL_AUDIT.json'
assert pin(receipt)['sha256']=='fabaca37de3156dd81e6c11886e3a085c81dd558411be2042c2c7a7d47de9673'
assert pin(audit)['sha256']=='bcceae437126f80d23421411297ff980ba86c3d2cef336647c1918a7384de993'
r=read(receipt);a=read(audit)
assert r['status']=='complete'and r['errors']==[]and r['child_exit_code']==0 and r['owned_remaining']==[]
assert r['sources_unchanged']is r['diagnostic_execution_complete']is True
assert r['devices_after']['all_idle']is True
assert a['status']=='passed_unscored_terminal_evidence_audit'and a['errors']==[]and len(a['case_audits'])==16
rows=[]
for case in read(P/'FIXTURE.json')['cases']:
 path=P/'actual'/('rank1-'+case['case_id']+'.json');d=read(path)
 pairs=d['controlled_natural_hidden80'];greedy=d['controlled_natural_greedy80'];eager=d['controlled_eager_FIA80'][0]['heads']
 tokens=[v['snapshot']['tokens'][0]for v in d['controlled_PA1']]
 rows.append(dict(case_id=case['case_id'],rank1_source=pin(path),original_reference_token=case['retained_reference_token'],
  historical_candidate_token=case['retained_candidate_token'],own_history_PA1_all16=tokens,
  eager_FIA80_roots=eager['full80_root_tokens'],same_computed_hidden_row1_root=eager['row1_root0_token'],
  same_hidden_full80_vs_row1_logits_exact=eager['root0_restricted_logits_exact'],
  same_hidden_full80_vs_row1_max_abs_logits=eager['root0_max_abs_logit_difference'],
  hidden80_FTF_root0=[v['heads']['full80_root_tokens'][0]for v in pairs],
  greedy80_FTF_root0=[v['physical_tokens'][0]for v in greedy],
  protected_all128_past_fingerprints_equal=d['past_before']==d['past_final']))
out=D/'ROOT_CANONICAL_PREFIX_HISTORY_V2_TERMINAL_RESULT_AND_COMPARISON.json'
result=dict(schema='ROOT.canonical-prefix-history-v2.audited-result.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 status='complete_independent_HOST_audit_passed_unscored',receipt=pin(receipt),independent_audit=pin(audit),
 actual_elapsed_seconds=r['elapsed_seconds'],session_id=11592,session_closed=True,child_exit_code=0,owned_remaining=[],all4_idle=True,
 source_unchanged=True,cases=rows,per_target_rank_own_AR_model_and_head_requests=9200,draft_model_and_head_requests=0,
 conclusions=[
 'All16 owners per case first reproduced their own original single-row AR history; controlled routes then shared that local protected past. No reference KV was copied.',
 'At P173-r1-y34 all16 PA1 roots choose7629 while eager FIA80 and restored hidden/greedy80 choose525 with the same protected past. Replacing only the head row count with1 on the same computed80-row hidden still chooses525.',
 'For the other three cases, PA1 and controlled eager/restored80 root observations agree. Historical candidate disagreements in those cases are not reproduced by this controlled own-history setup.',
 'All observed same-computed-hidden full80 versus row1 logits differ in some BF16 values, but their eager root token agrees in each of four cases. This alone does not explain the remaining PA1/FIA80 root mismatch.',
 'Changed-input and restored phases use actual Native graphs and preserve all128 past fingerprints. These tests do not reconstruct the original historical packed call or qualify whole463 own-AR equality.'
 ],numerical_root_cause_proven=False,production_qualified=False,whole463_AR_qualified=False,numerical_losslessness_proved=False,
 formal_Goodput=False,adopted=False)
with out.open('x')as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
summary=dict(result=pin(out),receipt=pin(receipt),independent_audit=pin(audit),status=result['status'],
 session_id=11592,model_running=False,whole463_AR_qualified=False,numerical_losslessness_proved=False,formal_Goodput=False,adopted=False)
session_keys=('active_model_session','active_model_session_id','active_session','active_session_id','current_model_session',
 'current_model_session_id','current_session_id','current_root_exec_session','root_exec_session')
for path in (B/'tree-v1/CURRENT_PROGRESS.json',Path('/root/SpecSLO/MIGRATION_STATUS.json')):
 doc=read(path)
 def update(d):
  d['canonical_prefix_history_v2_terminal']=summary
  d['canonical_prefix_history_v2_actual_complete']=True
  d['canonical_prefix_history_v2_independent_audit']=pin(audit)
  d['canonical_prefix_history_v2_future_HOST_auditor']['executed']=True
  d['canonical_prefix_history_v2_actual_started']['status']=result['status']
  for k in session_keys:
   if k in d:d[k]=None
  for k in ('model_running','current_model_running','supervisor_running','model_starting','current_starting','model_initialization_pending'):
   d[k]=False
  d['active_experiment']=None;d['active_directory']=None;d['active_trial_directory']=None;d['active_trial_tag']=None
  d['active_source_immutable']=False;d['source_files_must_not_be_edited_until_trial_finishes']=False
  d['current_stage']='canonical_history_diagnostic_complete_audited; V201_REV2_source_preparation_for_next_full'
  d['last_completed_model_session']=11592
 update(doc)
 if 'dynamic_tree_budget'in doc:update(doc['dynamic_tree_budget'])
 tmp=path.with_name(path.name+'.ROOT-update.tmp');tmp.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)
print(json.dumps(dict(result=pin(out),cases=len(rows),status=result['status'])))
