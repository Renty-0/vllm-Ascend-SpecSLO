"""Prepare thin immutable-parent source projection; never freeze or launch."""
import ast,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-parent3-union3-oracle-device-indexfix-full-v187'
P=B/'tree-parent3-union3-padding-observer-devicefix-full-v188';G=D/'v188-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def normalize(text):return text.replace(str(OLD),str(P)).replace(OLD.name,P.name).replace('187','188')
BEFORE=""" order=real+dummy;index=torch.tensor(order,dtype=torch.long,device=engine.device)
 view=replace(metadata,slot_mapping=metadata.slot_mapping.index_select(0,index),
  context_lens=metadata.context_lens.index_select(0,index),block_tables=metadata.block_tables.index_select(0,index),
  sequence_lens=tuple(metadata.sequence_lens[i] for i in order))
 result=padding_reference.inspect(engine,inputs.index_select(0,index),positions.index_select(0,index),view,len(real))"""
AFTER=""" order=real+dummy
 # Diagnostic-only view: Native tensors need not share a device. In particular
 # padded context_lens is legally CPU while model inputs/slots are on NPU.
 def permute(value):
  index=torch.tensor(order,dtype=torch.long,device=value.device)
  return value.index_select(0,index)
 view=replace(metadata,slot_mapping=permute(metadata.slot_mapping),
  context_lens=permute(metadata.context_lens),block_tables=permute(metadata.block_tables),
  sequence_lens=tuple(metadata.sequence_lens[i] for i in order))
 result=padding_reference.inspect(engine,permute(inputs),permute(positions),view,len(real))"""
def main():
 assert not (P/'CPU_RESULT188.json').exists() and not (P/'FROZEN_SOURCE188.json').exists()
 f=read(OLD/'FROZEN_SOURCE187.json');c=read(OLD/'CPU_RESULT187.json')
 assert c['status']=='passed' and c['tests_run']==3 and c['local_before']==c['local_after']==f['files']
 omitted={'INPUTS187.json','DERIVATIONS187.json','PARENT_PROJECTION187.json','ORACLE_PROJECTION187.json','host187.py','projection187.py','check187.py'}
 rows={}
 for path,digest in f['files'].items():
  src=Path(path);assert sha(src)==digest
  if src.name in omitted:continue
  name=src.name.replace('187','188');text=normalize(src.read_text());deltas=[]
  if name=='observer188.py':
   assert text.count(BEFORE)==1;text=text.replace(BEFORE,AFTER);deltas=[dict(old=BEFORE,new=AFTER,count=1)]
  if name=='SCOPE188.txt':
   note='\nV188 supersedes failed187 observer diagnostics only: each diagnostic input/position/slot/context/table permutation index uses its tensor device. The corrected187 independent oracle is preserved. Original production math, timing, budgets and every current qualification gate are unchanged. Failed187 elapsed1089.732489s was not a7200s timeout and supplied no full result or new qualification. HOST12 mixed-device layouts exercise actual observer→original inspector with explicit carrier/cache boundaries, not NPU model mathematics. Frozen177 scoped storage helper is reused exactly.\n'
   deltas=[dict(old=text,new=text+note,count=1)];text+=note
  target=P/name
  if not target.exists() or target.read_text()!=text:target.write_text(text)
  if target.suffix=='.py':ast.parse(text)
  rows[name]=dict(parent=str(src),parent_sha256=digest,namespace=[str(OLD),str(P)],basename_namespace=[OLD.name,P.name],epoch_namespace=['187','188'],changes=deltas,sha256=sha(target))
 old_guard=D/'v187-root-execution'
 for name in ('guard187.py','launch187.sh','startup187.sh','GUARD_INPUTS187.json'):
  target=G/name.replace('187','188');text=normalize((old_guard/name).read_text())
  if name=='guard187.py':
   text=text.replace("c['tests_run']==3","c['tests_run']==2").replace('actual own3 HOST','actual own2 HOST')
   a=text.index('def safe_bytes_under(');b=text.index('\ndef source_state(',a)
   text=text[:a]+"_storage_path=checked(B/'budget-diagnostics/v177-root-safe-storage-overlay/safe_storage177.py')\n_storage_spec=importlib.util.spec_from_file_location('_V188_frozen_storage177',_storage_path)\n_storage=importlib.util.module_from_spec(_storage_spec);_storage_spec.loader.exec_module(_storage)\ndef bytes_under(path):return _storage.bytes_under(path,live_roots=(P/'actual',),require=require)\n"+text[b:]
   text=text.replace("P176=require_parent(archive=True)","P187=require_parent(archive=True)")
   old=""" return merge(frozen['files'],frozen['external_files'],frozen['retained_HOST_evidence'],frozen['execution_guard_inputs'],
  frozen['peer_reviews'],{str(CPU):frozen['CPU_sha256'],str(FREEZE):sha(FREEZE),
   frozen['root_review']['path']:frozen['root_review']['sha256']})"""
   new=""" sources=merge(frozen['files'],frozen['external_files'],frozen['retained_HOST_evidence'],frozen['execution_guard_inputs'],
  frozen['peer_reviews'],{str(CPU):frozen['CPU_sha256'],str(FREEZE):sha(FREEZE),
   frozen['root_review']['path']:frozen['root_review']['sha256']})
 # This sole inherited binary remains in external_files and is SHA-checked by
 # source_state/parent187; its bytes need not be nested in another source tar.
 dependency=str(B/'tree-natural-positive-union-discovery-local-v154/PREPARED_SOURCE_ARCHIVE_V154.tar.gz')
 require(dependency in frozen['external_files'] and sources[dependency]==sha(dependency),'one explicit old binary verified in place')
 require({p for p in sources if p.endswith('.tar.gz')}=={dependency},'no undeclared binary archive omissions')
 return {p:h for p,h in sources.items() if p!=dependency}"""
   assert text.count(old)==1;text=text.replace(old,new)
  if not target.exists() or target.read_text()!=text:target.write_text(text)
 external={}
 for key in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):external.update(f[key])
 external.update({str(OLD/'CPU_RESULT187.json'):sha(OLD/'CPU_RESULT187.json'),str(OLD/'FROZEN_SOURCE187.json'):sha(OLD/'FROZEN_SOURCE187.json'),f['root_review']['path']:f['root_review']['sha256']})
 evidence=read(D/'ROOT_V187_ACTUAL_PADDING_OBSERVER_MIXED_DEVICE_INDEX_FAILURE.json')
 assert evidence['timeout'] is False and evidence['full463_reached'] is False and evidence['sources_unchanged'] is True and evidence['owned_remaining']==[]
 for item in [evidence['receipt'],evidence['worker_exit'],*evidence['errors']]:
  assert sha(item['path'])==item['sha256'];external[item['path']]=item['sha256']
 for path in [D/'ROOT_V187_ACTUAL_PADDING_OBSERVER_MIXED_DEVICE_INDEX_FAILURE.json',D/'V187_INTERIOR_PADDING_OBSERVER_DEVICE_INDEX_ROOT_CAUSE_READONLY.json',D/'v177-root-safe-storage-overlay/safe_storage177.py',Path(__file__).resolve()]:external[str(path)]=sha(path)
 inputs=read(OLD/'INPUTS187.json');inputs.update(schema='V188.selected-readonly-parent187-inputs.v1',files=external,failed187_attempt_qualification_inherited=False,observer_per_tensor_device_fix_only=True,original_parent_suite_rerun=False)
 save(P/'INPUTS188.json',inputs)
 archive=OLD/'PREPARED_SOURCE_ARCHIVE187.tar.gz'
 save(P/'OBSERVER_PROJECTION188.json',dict(schema='V188.exact-parent187-thin-source-projection.v1',files=rows,parent187=dict(path=str(OLD),CPU_sha256=sha(OLD/'CPU_RESULT187.json'),freeze_sha256=sha(OLD/'FROZEN_SOURCE187.json'),archive=dict(path=str(archive),sha256=sha(archive))),only_model_adjacent_delta='diagnostic observer permutation index uses each tensor device; pure187 oracle correction preserved',device_qualification_inherited=False))
 print(json.dumps(dict(projected=len(rows),external=len(external),NPU_used=False,freeze_created=False)))
if __name__=='__main__':main()
