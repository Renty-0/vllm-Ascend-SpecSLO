"""Build only source/provenance manifests. Never freeze or launch a device job."""
import ast,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-parent3-repair-frontier-union3-active32-full-v181'
P=B/'tree-parent3-union3-oracle-device-indexfix-full-v187';G=D/'v187-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def normalized(text):return text.replace(str(OLD),str(P)).replace('181','187')
ORACLE_OLD=""" ix=torch.tensor(order,dtype=torch.long,device=e.device)
 m=replace(m,slot_mapping=m.slot_mapping.index_select(0,ix),context_lens=m.context_lens.index_select(0,ix),
  block_tables=m.block_tables.index_select(0,ix),actual_seq_lengths_q=tuple(range(1,F+R+1)),
  sequence_lens=tuple(m.sequence_lens[i] for i in order))
 return x.index_select(0,ix),p.index_select(0,ix),m"""
ORACLE_NEW=""" # Native context_lens may remain on CPU; every tensor owns its index device.
 # This oracle-only permutation leaves the original PA values and ABI intact.
 def permute(value):
  ix=torch.tensor(order,dtype=torch.long,device=value.device)
  return value.index_select(0,ix)
 m=replace(m,slot_mapping=permute(m.slot_mapping),context_lens=permute(m.context_lens),
  block_tables=permute(m.block_tables),actual_seq_lengths_q=tuple(range(1,F+R+1)),
  sequence_lens=tuple(m.sequence_lens[i] for i in order))
 return permute(x),permute(p),m"""
def main():
 assert not (P/'FROZEN_SOURCE187.json').exists() and not (P/'CPU_RESULT187.json').exists()
 frozen=read(OLD/'FROZEN_SOURCE181.json');cpu=read(OLD/'CPU_RESULT181.json')
 assert cpu['status']=='passed' and cpu['local_before']==cpu['local_after']==frozen['files']
 # Retain the original26 source inverses, adjusting only the same oracle block.
 rows=json.loads(normalized((OLD/'DERIVATIONS181.json').read_text()))
 for name,row in rows.items():
  text=Path(row['parent']).read_text();assert sha(row['parent'])==row['parent_sha256']
  for delta in row['changes']:
   assert text.count(delta['old'])==delta['count'];text=text.replace(delta['old'],delta['new'])
  text+=row.get('append_text','')
  if name=='pure187.py':
   assert text.count(ORACLE_OLD)==1
   # This function is inherited in the original body, before append_text.
   text=text.replace(ORACLE_OLD,ORACLE_NEW)
   row['changes'].append(dict(old=ORACLE_OLD,new=ORACLE_NEW,count=1))
  assert text==(P/name).read_text(),name
  row['sha256']=sha(P/name)
 save(P/'DERIVATIONS187.json',rows)
 source_rows={}
 for path,digest in frozen['files'].items():
  old=Path(path);assert sha(old)==digest
  if old.name=='INPUTS181.json':continue
  name=old.name.replace('181','187');actual=(P/name).read_text();text=old.read_text();changes=[]
  for before,after in ((str(OLD),str(P)),('181','187')):
   count=text.count(before)
   if count:changes.append(dict(old=before,new=after,count=count));text=text.replace(before,after)
  if name=='pure187.py':
   assert text.count(ORACLE_OLD)==1;changes.append(dict(old=ORACLE_OLD,new=ORACLE_NEW,count=1));text=text.replace(ORACLE_OLD,ORACLE_NEW)
  row=dict(parent=str(old),parent_sha256=digest,changes=changes,sha256=sha(P/name))
  if actual!=text:
   assert name in ('host187.py','check187.py','projection187.py','DERIVATIONS187.json'),name
   row['replacement_text']=actual
   row['replacement_scope']='HOST test/source provenance only; no model or production behavior'
  if (P/name).suffix=='.py':compile(ast.parse(actual),str(P/name),'exec')
  source_rows[name]=row
 archive=OLD/'PREPARED_SOURCE_ARCHIVE181.tar.gz'
 descriptor=dict(schema='V187.whole-frozen181-source-projection.v1',files=source_rows,
  only_model_side_change='independent_frame per-tensor device for permutation index',production_changes=False,
  parent181=dict(path=str(OLD),CPU_sha256=sha(OLD/'CPU_RESULT181.json'),freeze_sha256=sha(OLD/'FROZEN_SOURCE181.json'),
   archive=dict(path=str(archive),sha256=sha(archive))),device_qualification_inherited=False)
 save(P/'ORACLE_PROJECTION187.json',descriptor)
 inputs=read(OLD/'INPUTS181.json');external=dict(inputs['files'])
 for key in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):external.update(frozen[key])
 external.update({str(OLD/'CPU_RESULT181.json'):sha(OLD/'CPU_RESULT181.json'),str(OLD/'FROZEN_SOURCE181.json'):sha(OLD/'FROZEN_SOURCE181.json'),
  frozen['root_review']['path']:frozen['root_review']['sha256'],str(Path(__file__).resolve()):sha(__file__)})
 # Preserve the failed source/device evidence without treating it as a proof.
 receipt=OLD/'receipt-parent3-union3-active32-v181-full463-v1.json';r=read(receipt)
 assert r['status']=='failed' and r['sources_unchanged'] is True and r['owned_remaining']==[]
 assert r['child']['exit_code']==1
 evidence={str(receipt):sha(receipt)}
 errors={}
 for rank in range(4):
  error=OLD/f'actual/parent3-union3-active32-v181-full463-v1/rank{rank}-last-error.json'
  raw=read(error);assert 'index is on npu:0' in raw['error'] and 'on cpu' in raw['error']
  evidence[str(error)]=sha(error);errors[str(rank)]=raw['error'].splitlines()[-1]
 for label in ('pre_verifier','post_verifier'):
  row=r[label];assert row['exit_code']==0 and row['actual_result']['status']=='pass'
  assert (row['actual_result']['checked_frozen_files'],row['actual_result']['checked_external_files'],row['actual_result']['checked_packages'])==(10496,15,212)
  assert sha(row['path'])==row['sha256'];evidence[row['path']]=row['sha256']
 hardware=r['devices_after']['hardware'];assert hardware['exit_code']==0
 for card in (2,5,6,7):assert f'No running processes found in NPU {card}' in hardware['output']
 classification=D/'V181_FAILED_ORACLE_DEVICE_BOUNDARY_PRESERVED_FOR_V187.json'
 value=dict(schema='V187.frozen181-terminal-failure-preservation.v1',status='failed_parent_preserved',candidate=str(OLD),
  evidence=evidence,receipt_status=r['status'],sources_unchanged=r['sources_unchanged'],owned_remaining=r['owned_remaining'],
  all_four_idle_from_retained_receipt=True,original_baseline_pre_post_pass=True,limits=r['limits'],errors=errors,
  cause='original independent_frame shared NPU index applied to CPU context_lens',
  full_completed=False,numerical_qualification_claimed=False,formal_Goodput=False,new_device_action_performed=False)
 if classification.exists():assert read(classification)==value
 else:save(classification,value)
 external.update(evidence);external[str(classification)]=sha(classification)
 inputs.update(schema='V187.selected-direct-source-inputs.v1',files=external,old181_failed_device_attempt_inherited=False,
  oracle_per_tensor_device_fix_only=True,original_parent_suite_rerun=False)
 save(P/'INPUTS187.json',inputs)
 print(json.dumps(dict(local_source_files=len(source_rows)+2,external_files=len(external),projection_sha256=sha(P/'ORACLE_PROJECTION187.json'),
  parent_failure_classification=dict(path=str(classification),sha256=sha(classification)),freeze_created=False,NPU_used=False)))
if __name__=='__main__':main()
