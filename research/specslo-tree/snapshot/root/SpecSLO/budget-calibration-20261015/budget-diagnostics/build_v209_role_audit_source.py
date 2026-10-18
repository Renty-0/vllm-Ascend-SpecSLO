"""One-time SOURCE-only projection. Never freezes, launches, or imports models."""
import difflib,hashlib,json,re,tokenize,io
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-causal80-small-cohort-single-writer-full-v205'
P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
OG=D/'v205-root-execution';G=D/'v209-root-execution'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def put(path,text):
 assert not path.exists(),str(path);path.write_text(text)
def edit(text,old,new,count=1):
 assert text.count(old)==count,(old,text.count(old),count)
 return text.replace(old,new)
def namespace(text,python=False):
 if not python:return text.replace('205','209')
 protected=[t for t in tokenize.generate_tokens(io.StringIO(text).readline)
  if t.type==tokenize.NUMBER or t.type==tokenize.STRING and re.fullmatch(r"['\"][0-9a-f]{64}['\"]",t.string)]
 lines=text.splitlines(keepends=True);offsets=[0]
 for line in lines:offsets.append(offsets[-1]+len(line))
 spans=[(offsets[t.start[0]-1]+t.start[1],offsets[t.end[0]-1]+t.end[1]) for t in protected]
 for start,end in reversed(spans):text=text[:start]+text[start:end].replace('205','\x00V205\x00')+text[end:]
 return text.replace('205','209').replace('\x00V209\x00','205')
def operations(original,result):
 return [dict(start=a,end=b,old=original[a:b],new=result[c:d])
  for op,a,b,c,d in difflib.SequenceMatcher(a=original,b=result,autojunk=False).get_opcodes() if op!='equal']
def projected(parent,name,text):
 original=namespace(parent.read_text(),parent.suffix=='.py')
 put(P/name,text)
 return dict(file=name,parent=str(parent),parent_sha256=sha(parent),namespace=['205','209'],
  namespace_preserves_python_NUMBER_and_64hex=True,deltas=operations(original,text))

assert not P.exists() and not G.exists(),'new private209 must be unused'
P.mkdir();G.mkdir()
projection=[];guard_projection=[]
for parent in sorted(OLD.glob('*.py')):
 if parent.name in ('check205.py','host_full205.py','test_full205.py'):continue
 name=parent.name.replace('205','209');text=namespace(parent.read_text(),True)
 if name=='audit_usage209.py':
  text=edit(text,"and r['rank']==rank and r['completed']", "and r['rank']==rank and r['role']==('draft'if rank==0 else'target')and r['completed']")
  text=edit(text,"and len(row['draft']['variable_num_draft_tokens'])==16",'')
  text=edit(text,"    if rank==0:\n     d=row['draft'];require(",
   "    if rank==0:\n     require(row['target']is row['reference']is row['crossM']is row['old_control']is None,'draft owns no target tensors or crossM proof')\n     d=row['draft'];require(type(d)is dict and len(d['variable_num_draft_tokens'])==16 and ")
  text=edit(text,"    else:\n     target=row['target'];ref=row['reference'];x=row['crossM'];require(",
   "    else:\n     require(row['draft']is None and all(type(row[k])is dict for k in('target','reference','crossM','old_control')),'target owns no draft proof and retains its own target/reference/crossM evidence')\n     target=row['target'];ref=row['reference'];x=row['crossM'];require(")
  anchor="   require({k:nxt[k]for k in('source','selected','rows','confidences')}==t['logical']['next'],'actual current own next retained')"
  text=edit(text,anchor,anchor+"\n   d=nxt['draft_proof']\n   if rank==0:require(type(d)is dict and len(d['variable_num_draft_tokens'])==16 and d['raw_encoded_shape']==[16,4,2]and d['all_physical_hidden_original_matcher']and d['all28_query_KV_original_matcher']and d['exact_full8_encoded']and d['source_KV_copied']is False,'draft next retains its own complete full16 numerical proof')\n   else:require(d is None,'target next must not fabricate a draft numerical proof')")
 if name=='base209.py':
  text+='''\n\ndef require_derivation205(*,archive=False):
 import tarfile
 pin=read(P/'INPUTS209.json')['source_donor205'];root=Path(pin['path']);f=read(root/'FROZEN_SOURCE205.json')
 require(sha(root/'FROZEN_SOURCE205.json')==pin['freeze_sha256']and sha(root/'CPU_RESULT205.json')==pin['CPU_sha256'],'immutable205 source/HOST donor only; failed model run grants no qualification')
 expected={**f['files'],**f['external_files'],**f['retained_HOST_evidence'],**f['execution_guard_inputs'],**f['peer_reviews'],str(root/'CPU_RESULT205.json'):f['CPU_sha256'],str(root/'FROZEN_SOURCE205.json'):pin['freeze_sha256'],f['root_review']['path']:f['root_review']['sha256']}
 for path,h in expected.items():require(sha(path)==h,'exact205 source donor '+path)
 if archive:
  ar=pin['archive'];require(sha(ar['path'])==ar['sha256'],'original205 donor archive SHA in place')
  with tarfile.open(ar['path'],'r:gz')as saved:
   members=saved.getmembers();require(len(members)==len(expected)==ar['members']and len({m.name for m in members})==len(members),'exact205 regular source archive set')
   for m in members:require(m.isfile()and '/'+m.name in expected and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==expected['/'+m.name],'actual205 regular source member '+m.name)
 return dict(source_donor205=str(root),members=len(expected),archive_verified_in_place=archive,production_parent_remains197=True,device_qualification_inherited=False)
'''
 projection.append(projected(parent,name,text))

text=namespace((OLD/'check205.py').read_text(),True)
text=edit(text," from base209 import require_production_parent,require_source_parent", " from base209 import require_production_parent,require_source_parent,require_derivation205\n require_derivation205(archive=archive)")
text=edit(text,"c['tests_run']==7","c['tests_run']==3")
put(P/'check209.py',text)
text=namespace((OLD/'host_full205.py').read_text(),True)
text=edit(text,'New seven HOST checks only','Three new retained-role/source HOST checks only')
put(P/'host_full209.py',text)
put(P/'SCOPE209.txt',"SOURCE ONLY. Sole production parent197 and exact frozen205 source donor. Only retained audit role/null handling changes; same actual constructor/B4/M80/multiowner/7N/current-engine protocols/Native/2zero/warm/seal/full463 must run again. Actual205 raw is a HOST regression fixture, never a new-engine qualification. No new math/profile/key/budget/admission/clock or performance claim.\n")

for parent in sorted(OG.iterdir()):
 if not parent.is_file():continue
 text=namespace(parent.read_text(),parent.suffix=='.py')
 if parent.name=='guard205.py':
  text=edit(text,"c['tests_run']==7","c['tests_run']==3")
  text=edit(text,'actual own7 HOST exact source pre/post','actual own3 retained-role/source HOST exact source pre/post')
 name=parent.name.replace('205','209');put(G/name,text)
 original=namespace(parent.read_text(),parent.suffix=='.py')
 guard_projection.append(dict(file=name,parent=str(parent),parent_sha256=sha(parent),namespace=['205','209'],
  namespace_preserves_python_NUMBER_and_64hex=True,deltas=operations(original,text)))

f=read(OLD/'FROZEN_SOURCE205.json')
donor={**f['files'],**f['external_files'],**f['retained_HOST_evidence'],**f['execution_guard_inputs'],**f['peer_reviews'],
 str(OLD/'CPU_RESULT205.json'):f['CPU_sha256'],str(OLD/'FROZEN_SOURCE205.json'):sha(OLD/'FROZEN_SOURCE205.json'),
 f['root_review']['path']:f['root_review']['sha256']}
A=OLD/'actual'/'small-tail-single-writer-v205-full463-v1'/'causal80-seven-real-N-current-process'
raw={str(p):sha(p) for p in sorted(A.glob('*.json.gz'))};assert len(raw)==424
receipt=OLD/'receipt-small-tail-single-writer-v205-full463-v1.json'
assert sha(receipt)=='a6f6845786219fb7fe5b6f5b70996541b8395b97f935719de240228507100621'
inputs=read(OLD/'INPUTS205.json');inputs['files'].update(donor);inputs['files'].update(raw)
inputs['files'][str(receipt)]=sha(receipt);inputs['files'][str(Path(__file__))]=sha(__file__)
inputs['files'][str(D/'V204_V206_FROZEN_ROLE_NULL_AUDITOR_SOURCE_SCAN.json')]=sha(D/'V204_V206_FROZEN_ROLE_NULL_AUDITOR_SOURCE_SCAN.json')
inputs['source_donor205']=dict(path=str(OLD),freeze_sha256=sha(OLD/'FROZEN_SOURCE205.json'),CPU_sha256=sha(OLD/'CPU_RESULT205.json'),
 archive=dict(path=str(OLD/'PREPARED_SOURCE_ARCHIVE205.tar.gz'),sha256=sha(OLD/'PREPARED_SOURCE_ARCHIVE205.tar.gz'),members=len(donor)),
 failed_actual_does_not_qualify_current_engine=True)
inputs['retained_actual205_HOST_fixture']=dict(directory=str(A),raw=raw,receipt=dict(path=str(receipt),sha256=sha(receipt)),
 terminal_failed=True,model_qualification_transferred=False)
put(P/'INPUTS209.json',json.dumps(inputs,indent=2,sort_keys=True)+'\n')
put(P/'PROJECTIONS209.json',json.dumps(dict(projection=projection,guard_projection=guard_projection,
 only_production_execution_delta='None: all actual math/model/policy/clock functions are namespace exact; only role-aware retained audit and source/HOST identity change.',
 source_donor205_failure_retained=True),indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(candidate=str(P),guard=str(G),whole_projection=len(projection),external=len(inputs['files']),retained205raw=len(raw))))
