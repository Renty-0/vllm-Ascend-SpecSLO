"""HOST-only bindings and exact original supervisor-helper reuse for P154."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import time

G=Path(__file__).resolve().parent
B=G.parent.parent
P=B/'tree-natural-positive-union-discovery-local-v154'
ACTION='natural-positive-union-discovery-local'
CPU=P/'CPU_RESULT154.json'
CPU_SHA='562ea284e40396faed0889a54efaa77efebfed4bbc8352ec7ee2551c11b58053'
FREEZE=P/'FROZEN_SOURCE154.json'
ARCHIVE=P/'PREPARED_SOURCE_ARCHIVE_V154.tar.gz'
LOCK=B/'.ROOT_NPU_EXECUTION.lock'
PARENT=B/'tree-large-mixed-prompt-local-v147'
MAX_SECONDS=1800
MAX_TAG_BYTES=32*1024**2
MAX_AGGREGATE_BYTES=64*1024**2
SOURCE_PEER=B/'budget-diagnostics/V154_INDEPENDENT_SOURCE_PEER_REVIEW.json'
SCOPE=dict(local_sequential_component_only=True,parent4_qualified=False,production_deferE_gate_opened=False,
 live_owner_alignment_proved=False,live_overlap_proved=False,full_workload_qualified=False,full_qualification=False,
 two_complete_zero_traces=False,cost_measured=False,dynamic_TTU_proved=False,formal_Goodput=False,adopted=False,
 will_finish_assumption='local max_tokens64 / initial_completion1 / parent delivery<=4 only; not a general dynamic prediction')

def require(value,message):
 if value is not True:raise RuntimeError('V154 ROOT execution: '+message)

def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
 return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def fresh(path):
 path=Path(path);require(not path.exists() and not path.is_symlink(),'preserve previous output '+str(path))

def checked_map(values):
 require(type(values) is dict and bool(values),'nonempty exact input map')
 for name,expected in values.items():
  path=Path(name)
  require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve()==path
   and re.fullmatch('[0-9a-f]{64}',expected) is not None and sha(path)==expected,'exact regular source '+name)
 return values

def merge(*maps):
 result={}
 for values in maps:
  for path,digest in values.items():
   require(path not in result or result[path]==digest,'contradictory input binding '+path)
   result[path]=digest
 return result

def source_state():
 """Bind own6 HOST/17local/195readonly and every retained HOST attempt."""
 require(P.resolve()==P and G.resolve()==G and sha(CPU)==CPU_SHA,'fixed private source/CPU identity')
 cpu=read(CPU);local=cpu['input_hashes'];external=cpu['external_input_hashes']
 require(cpu['status']=='passed' and cpu['tests_run']==6 and cpu['failures']==cpu['errors']==0
  and cpu['Torch_imported'] is cpu['NPU_initialized'] is cpu['model_initialized'] is False
  and local==cpu['input_hashes_after'] and external==cpu['external_input_hashes_after']
  and len(local)==17 and len(external)==195,'actual own CPU17/195 pre/post proof')
 current={str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('INPUTS154.json','FIXTURE_INPUT154.json','SCOPE154.txt'))}
 require(local==current and read(P/'INPUTS154.json')['files']==external,'exact original17/195 input maps')
 checked_map(local);checked_map(external)
 prepared=read(P/'PREPARED_HOST154.json')
 require(prepared['files']==local and prepared['external_files']==external and prepared['CPU_sha256']==CPU_SHA,
  'actual retained own HOST preparation identity')
 attempts=prepared['actual_attempts'];earlier=prepared['retained_earlier_CPU']
 require(set(attempts)=={str(p) for p in P.glob('HOST_ATTEMPT154_*.json')}
  and set(earlier)=={str(p) for p in P.glob('CPU_RESULT154_PRE_*.json')},'all retained actual HOST attempts and earlier CPU')
 evidence=merge({str(P/'PREPARED_HOST154.json'):sha(P/'PREPARED_HOST154.json')},attempts,earlier)
 checked_map(evidence)
 dependency=read(P/'INPUTS154.json')['P151_actual_conclusion'];actual=read(dependency['path'])
 require(sha(dependency['path'])==dependency['sha256'] and actual['ROOT_actual_conclusion_reviewed'] is True
  and actual['status']=='passed_subset' and actual['qualification_inherited'] is False,'actual negative P151 prerequisite only')
 return dict(local=local,external=external,evidence=evidence,CPU_sha256=CPU_SHA)

def guard_inputs():
 names=read(G/'GUARD_STATIC_INPUTS.json')['files']
 require(len(names)==len(set(names))==6 and all(Path(n).name==n for n in names),'exact six execution guard sources')
 own={str(G/name):sha(G/name) for name in names}
 original=read(G/'ORIGINAL_INPUTS.json')['files'];checked_map(own);checked_map(original)
 parent_freeze=read(PARENT/'PREPARED_SOURCE_HASHES_V147.json')
 require(parent_freeze['files'][str(PARENT/'supervise_v147.py')]==original[str(PARENT/'supervise_v147.py')],
  'actual frozen original P147 supervisor identity')
 # Launcher is the original P147 environment with only component identity
 # substitutions. The supervisor supplies a clean environment around it.
 template=(PARENT/'launch_v147.sh').read_text()
 wanted=template.replace('tree-large-mixed-prompt-local-v147','tree-natural-positive-union-discovery-local-v154').replace('V147 argument','V154 argument').replace('large-mixed-prompt-local-qualification',ACTION).replace('runtime_v147.py','runtime154.py')
 require((G/'launch_v154.sh').read_text()==wanted,'exact original launcher environment with only local identity substitutions')
 return merge(own,original)

def original_helpers():
 """Exact AST bodies, fresh namespace; never execute/import parent main."""
 guard_inputs();path=PARENT/'supervise_v147.py';tree=ast.parse(path.read_text())
 namespace=dict(Path=Path,P=P,B=B,require=require,sha=sha,json=json,importlib=__import__('importlib'),
  subprocess=subprocess,signal=signal,os=os,time=time)
 selected=[copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
  for name in ('save','helper','verify')]
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 original_try=next(n for n in main.body if isinstance(n,ast.Try))
 cleanup=copy.deepcopy(original_try.finalbody[0])
 require(isinstance(cleanup,ast.If) and ast.unparse(cleanup.test)=='child is not None and h is not None',
  'original owned-session cleanup anchor')
 function=ast.FunctionDef(name='owned_cleanup',args=ast.arguments(posonlyargs=[],
  args=[ast.arg(arg='child'),ast.arg(arg='h'),ast.arg(arg='row')],kwonlyargs=[],kw_defaults=[],defaults=[]),
  body=[cleanup],decorator_list=[])
 unit=ast.Module(body=[*selected,function],type_ignores=[])
 exec(compile(ast.fix_missing_locations(unit),str(path)+':V154-private-exact-helpers','exec'),namespace)
 return {name:namespace[name] for name in ('save','helper','verify','owned_cleanup')}

def review_binding(root_path,root_sha,peers,state=None,guards=None):
 state=source_state() if state is None else state;guards=guard_inputs() if guards is None else guards
 root_path=Path(root_path).resolve()
 require(re.fullmatch('[0-9a-f]{64}',root_sha) is not None and sha(root_path)==root_sha,'explicit ROOT review SHA')
 require(len(peers)==2 and len({str(Path(p).resolve()) for p,h in peers})==2,'two distinct exact peer reviews')
 peer_map={str(Path(p).resolve()):h for p,h in peers};checked_map(peer_map)
 for path in peer_map:
  status=read(path).get('status','').replace('-','_')
  require(status=='passed_source_review','peer passed source review, no device qualification inherited')
 require(str(SOURCE_PEER) in peer_map,'explicit V154 source peer among ROOT-reviewed exact peer SHAs')
 inputs=merge(state['local'],state['external'],state['evidence'],{str(CPU):CPU_SHA},guards,peer_map)
 root=read(root_path)
 require(root['status']=='passed_source_review' and root['reviewed_inputs']==inputs,
  'ROOT reviewed exact17+195+retained HOST+CPU+guard+two-peer flat map')
 return dict(path=str(root_path),sha256=root_sha),peer_map,inputs

def archive_sources(frozen):
 return merge(frozen['files'],frozen['external_files'],frozen['retained_HOST_evidence'],{str(CPU):frozen['CPU_sha256'],str(FREEZE):sha(FREEZE)},
  frozen['execution_guard_inputs'],{frozen['root_review']['path']:frozen['root_review']['sha256']},frozen['peer_reviews'])

def verify_archive(path,sources):
 with tarfile.open(path,'r:gz') as tar:
  members=tar.getmembers()
  require(len(members)==len(sources) and {m.name for m in members}=={p.lstrip('/') for p in sources},'unique exact archive input set')
  for member in members:
   require(member.isfile() and hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sources['/'+member.name],
    'archive member exact bytes '+member.name)
 return dict(path=str(path),sha256=sha(path),members=len(sources),selected_source_archive_only=True,
  complete_environment_import_closure_claim=False)

def write_archive(path,sources):
 fresh(path);checked_map(sources)
 with tarfile.open(path,'x:gz') as tar:
  for name in sorted(sources):tar.add(name,arcname=name.lstrip('/'),recursive=False)
 return verify_archive(path,sources)

def prepared():
 state=source_state();guards=guard_inputs();frozen=read(FREEZE)
 require(frozen['files']==state['local'] and frozen['external_files']==state['external']
  and frozen['CPU_sha256']==CPU_SHA and frozen['ROOT_source_reviewed'] is True and frozen['retained_HOST_evidence']==state['evidence'] and frozen['execution_guard_inputs']==guards,
  'prepared exact source and execution guard bytes, including this wrapper')
 root,peers,_=review_binding(frozen['root_review']['path'],frozen['root_review']['sha256'],
  list(frozen['peer_reviews'].items()),state,guards)
 require(root==frozen['root_review'] and peers==frozen['peer_reviews'],'prepared fixed ROOT/peer review bytes')
 sources=archive_sources(frozen);checked_map(sources);archive=verify_archive(ARCHIVE,sources)
 return frozen,sources,dict(CPU_sha256=CPU_SHA,freeze_sha256=sha(FREEZE),
  root_review=root,peers=peers,archive=archive,selected_local_files=17,selected_external_files=195)

def bytes_under(path):
 path=Path(path)
 if not path.exists():return 0
 require(path.resolve()==path and not path.is_symlink(),'private storage root no alias')
 total=0
 for base,dirs,files in os.walk(path,followlinks=False):
  for name in (*dirs,*files):require(not (Path(base)/name).is_symlink(),'private storage contains unexpected link')
  for name in files:total+=(Path(base)/name).stat().st_size
 return total

def storage(tag=None):
 actual=0 if tag is None else bytes_under(P/'actual'/tag)
 aggregate=bytes_under(P)+bytes_under(G)
 require(actual<=MAX_TAG_BYTES and aggregate<=MAX_AGGREGATE_BYTES,'32MiB actual tag /64MiB aggregate private cap')
 return dict(actual_tag_bytes=actual,aggregate_private_bytes=aggregate)
