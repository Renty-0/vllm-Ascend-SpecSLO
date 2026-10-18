"""ROOT-self-locked full execution; original baseline verifier and owned cleanup."""
import argparse,ast,copy,datetime,fcntl,gzip,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
G=Path(__file__).resolve().parent;B=G.parent.parent;P=B/'tree-active-window-full-v166'
if str(P) not in sys.path:sys.path.insert(0,str(P))
from base166 import BASE,ACTION,sha,read,require,parent_ready,parent_archive,checked
CPU=P/'CPU_RESULT166.json';FREEZE=P/'FROZEN_SOURCE166.json';ARCHIVE=P/'PREPARED_SOURCE_ARCHIVE166.tar.gz'
LOCK=B/'.ROOT_NPU_EXECUTION.lock'
MAX_SECONDS=7200;MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=512*1024**2
SOURCE_PEER=B/'budget-diagnostics/V166_INDEPENDENT_SOURCE_PEER_REVIEW.json'
SCOPE=dict(active_windows_requested=[32,24,16],configured_capacity=64,queued=463,storage=527,
 old_21scope26phase_claimed=False,cost_measured=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
PG_PATH=B/'budget-diagnostics/v154-root-execution/common_guard.py'
checked(PG_PATH)
spec=importlib.util.spec_from_file_location('_V166_original_guard_helpers',PG_PATH)
PG=importlib.util.module_from_spec(spec);spec.loader.exec_module(PG)
checked_map=PG.checked_map;merge=PG.merge;fresh=PG.fresh;verify_archive=PG.verify_archive;write_archive=PG.write_archive
bytes_under=PG.bytes_under

def source_state(*,archive=True):
 from check166 import local_files,external_files
 c=read(CPU);local=local_files();external=external_files()
 require(c['status']=='passed' and c['tests_run']==5 and c['failures']==c['errors']==0
  and c['Torch_imported'] is c['NPU_used'] is False
  and c['local_before']==c['local_after']==local and c['external_before']==c['external_after']==external,
  'actual own5 HOST exact source pre/post')
 checked_map(local);checked_map(external)
 prepared=read(P/'PREPARED_HOST166.json')
 require(prepared['files']==local and prepared['external_files']==external and prepared['CPU_sha256']==sha(CPU),'retained actual HOST preparation')
 attempts=prepared['actual_HOST_attempts'];require(set(attempts)=={str(p) for p in P.glob('HOST_ATTEMPT166_*.json')},'retain every HOST attempt')
 evidence=merge({str(P/'PREPARED_HOST166.json'):sha(P/'PREPARED_HOST166.json')},attempts);checked_map(evidence)
 return dict(local=local,external=external,evidence=evidence,CPU_sha256=sha(CPU),
  parent_ready=parent_ready(),parent_archive=parent_archive() if archive else None)

def guard_inputs():
 value=read(G/'GUARD_INPUTS166.json');names=value['files']
 require(names==['guard166.py','launch166.sh','startup166.sh','GUARD_INPUTS166.json'],'four fixed execution guard sources')
 values={str(G/n):sha(G/n) for n in names};checked_map(values)
 return values

def review_binding(root_path,root_sha,peers,state=None):
 state=source_state() if state is None else state
 root_path=Path(root_path).resolve();require(sha(root_path)==root_sha,'explicit ROOT source review SHA')
 peer_map={str(Path(p).resolve()):h for p,h in peers}
 require(len(peer_map)==len(peers)>=1 and str(SOURCE_PEER) in peer_map,'explicit current combined source/guard peer')
 checked_map(peer_map)
 require(all(read(p)['status'].replace('-','_')=='passed_source_review' and read(p)['candidate']==str(P) for p in peer_map),'actual current source peer status and candidate')
 inputs=merge(state['local'],state['external'],state['evidence'],guard_inputs(),{str(CPU):sha(CPU)},peer_map)
 root=read(root_path);require(root['status']=='passed_source_review' and root['reviewed_inputs']==inputs,'ROOT exact selected input/CPU/guard/peer flat map')
 return dict(path=str(root_path),sha256=root_sha),peer_map,inputs

def archive_sources(frozen):
 # The original73MB/6390-member binary remains an explicit verified dependency;
 # no old archive or complete377-source epoch is copied into the new tar.
 return merge(frozen['files'],frozen['external_files'],frozen['retained_HOST_evidence'],frozen['execution_guard_inputs'],
  frozen['peer_reviews'],{str(CPU):frozen['CPU_sha256'],str(FREEZE):sha(FREEZE),
   frozen['root_review']['path']:frozen['root_review']['sha256']})

def prepared():
 state=source_state();f=read(FREEZE)
 require(f['files']==state['local'] and f['external_files']==state['external']
  and f['CPU_sha256']==state['CPU_sha256'] and f['retained_HOST_evidence']==state['evidence']
  and f['execution_guard_inputs']==guard_inputs() and f['ROOT_source_reviewed'] is True
  and f['dependency_archives_verified_in_place']==state['parent_archive'],'current own freeze and original archive dependencies')
 root,peers,_=review_binding(f['root_review']['path'],f['root_review']['sha256'],list(f['peer_reviews'].items()),state)
 sources=archive_sources(f);checked_map(sources);archive=verify_archive(ARCHIVE,sources)
 return f,sources,dict(CPU_sha256=sha(CPU),freeze_sha256=sha(FREEZE),archive=archive,root_review=root,peers=peers,
  dependency_archives=state['parent_archive'],parent_ready=state['parent_ready'],device_qualification_inherited=False)

def storage(tag=None):
 actual=0 if tag is None else bytes_under(P/'actual'/tag);total=bytes_under(P)+bytes_under(G)
 require(actual<=MAX_TAG_BYTES and total<=MAX_AGGREGATE_BYTES,'private actual256MiB/aggregate512MiB limits')
 return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)

def original_helpers():
 namespace=dict(vars(PG),P=P,G=G,guard_inputs=guard_inputs,require=require,sha=sha)
 return type(PG.original_helpers)(PG.original_helpers.__code__,namespace,'original_helpers')()

def actual_audit(directory):
 from audit166 import audit
 completed=read(Path(directory)/'completed.json');exited=read(Path(directory)/'worker-exit.json')
 actual=audit(directory,(32,24,16))
 require(completed['action']==ACTION and completed['audit']==actual and actual['passed'] is True
  and completed['replies']==[['calibrated',i,dict(windows=[32,24,16],full463=True)] for i in range(4)]
  and exited==dict(closed=True,all_exited=True),'actual complete full32/24/16 workers and closed process pool')
 return actual,'passed_full',dict(raw_native_zero_subtractions=actual['native_zero_subtractions'],hidden_KV_recomputed_by_HOST=False)

def inherit_supervisor():
 path=B/'budget-diagnostics/v154-root-execution/supervise_v154.py'
 from base166 import checked
 text=checked(path).read_text()
 for old,new in (('launch_v154.sh','launch166.sh'),('V154.root-owned-local-component-supervisor','V166.root-owned-active-window-full-supervisor'),
  ('V154 supervisor received','V166 supervisor received'),('numeric_source_obligations_only=True','numeric_source_obligations_only=False'),
  ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
   'three full463 active windows; original own B4 numerical/protocol, real warm fixedpoint and Native zero/seal'),
  ('bounded1800second','bounded7200second'),('within1800seconds','within7200seconds'),
  ("'actual_component_audit'","'actual_full_audit'"),("'actual_local_coverage_status'","'actual_full_status'")):
  require(text.count(old)==1,'exact supervisor delta '+old);text=text.replace(old,new)
 wanted={'verifier_report','child_limits','launch_argv','execute_locked','main'}
 nodes=[copy.deepcopy(n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
 require({n.name for n in nodes}==wanted,'all original lock/verifier/owned-cleanup supervisor functions')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::V166-isolated','exec'),globals())
inherit_supervisor()

def prepare_only():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true',required=True)
 parser.add_argument('--root-review',required=True);parser.add_argument('--expected-root-review-sha256',required=True)
 parser.add_argument('--peer-review',nargs=2,action='append',required=True);a=parser.parse_args()
 state=source_state();root,peers,_=review_binding(a.root_review,a.expected_root_review_sha256,a.peer_review,state)
 fresh(FREEZE);fresh(ARCHIVE);storage()
 f=dict(schema='V166.ROOT-source-freeze.v1',files=state['local'],external_files=state['external'],CPU_sha256=sha(CPU),
  retained_HOST_evidence=state['evidence'],execution_guard_inputs=guard_inputs(),ROOT_source_reviewed=True,
  root_review=root,peer_reviews=peers,dependency_archives_verified_in_place=state['parent_archive'],
  selected_source_archive_only=True,**SCOPE)
 with FREEZE.open('x') as stream:json.dump(f,stream,indent=2,sort_keys=True);stream.write('\n')
 archive=write_archive(ARCHIVE,archive_sources(f));storage();prepared()
 print(json.dumps(dict(path=str(FREEZE),sha256=sha(FREEZE),archive=archive,NPU_used=False)))
if __name__=='__main__':
 if '--prepare-only' in sys.argv:prepare_only()
 else:main()
