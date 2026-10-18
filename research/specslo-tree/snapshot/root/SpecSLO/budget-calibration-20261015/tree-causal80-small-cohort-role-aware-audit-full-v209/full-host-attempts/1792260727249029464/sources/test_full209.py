"""Three current HOST regressions; retained205 tensors are not new model proof."""
import ast,copy,gzip,hashlib,importlib.util,json,re,sys,unittest
from pathlib import Path
from base209 import P,B,require,checked,read,sha
from scope_v146 import NATIVE_COUNTERS,FAILURES
from mapper209 import GREEDY,HIDDEN

DETAILS={};_CACHE={};_PINS={}
FIXTURE=read(P/'INPUTS209.json')['retained_actual205_HOST_fixture']
DIRECTORY=Path(FIXTURE['directory'])

def fixture_load(path):
 path=Path(path);key=str(path)
 require(key in FIXTURE['raw'],'only explicitly authorized immutable205 raw fixture')
 if key not in _CACHE:
  data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
  require(h==FIXTURE['raw'][key],'exact retained205 raw bytes before HOST replay')
  _CACHE[key]=json.loads(gzip.decompress(data));_PINS[key]=h
 return _CACHE[key]

def evaluator(path,mutations=None):
 path=checked(path) if path!=P/'audit_usage209.py' else path
 source=path.read_text();new=path.name=='audit_usage209.py'
 if new:
  require(source.count('V209.')==2,'only two literal schema namespace adaptations for HOST205 replay')
  source=source.replace('V209.','V205.')
  # Only producer schema namespace changes in this HOST-only evaluator.
  require(source.replace('V205.','V209.')==path.read_text(),'exact whole source schema inverse')
 tree=ast.parse(source);nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef)]
 ns=dict(gzip=gzip,json=json,Path=Path,require=require,sha=sha,GREEDY=GREEDY,HIDDEN=HIDDEN,
  NATIVE_COUNTERS=NATIVE_COUNTERS,FAILURES=FAILURES)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::HOST-retained205-schema-only','exec'),ns)
 old_load=ns['load']
 def load(p):
  key=str(Path(p));value=fixture_load(p)
  if mutations and key in mutations:
   value=copy.deepcopy(value);mutations[key](value)
  return value
 ns['load']=load
 return ns['audit'],dict(source_sha256=sha(path),HOST_schema_only_namespace_adaptation=new,
  retained_raw_JSON_reencoded=False,actual_original_load_function_AST_retained=True)

def cycle(rank=0):return str(DIRECTORY/f'rank{rank}-pass0-N9-cycle0.json.gz')
def next_file(rank=0):return str(DIRECTORY/f'rank{rank}-pass0-N9-next.json.gz')
def report(rank=0):return str(DIRECTORY/f'rank{rank}-component.json.gz')

class Tests(unittest.TestCase):
 def test01_actual_failure_and_complete_role_sensitive_replay(self):
  old=B/'tree-causal80-small-cohort-single-writer-full-v205/audit_usage205.py'
  fn,old_source=evaluator(old)
  with self.assertRaisesRegex(TypeError,"NoneType.*not subscriptable"):fn(DIRECTORY)
  fn,new_source=evaluator(P/'audit_usage209.py');result=fn(DIRECTORY)
  self.assertTrue(result['passed']);self.assertTrue(result['math_independent_qualified'])
  self.assertEqual(result['N'],list(range(9,16)));self.assertEqual(result['complete_protocol_commits'],84)
  self.assertEqual(result['Native_zero_subtractions'],1176);self.assertGreater(result['changed_host_length_TTU_replays'],0)
  self.assertEqual(len(result['raw_checkpoint_sha256']),420)
  for key in ('HOST_recomputes_unretained_hidden_or_KV','whole_AR','formal_Goodput','adopted'):self.assertIs(result[key],False)
  self.assertEqual(len(_CACHE),424)
  for rank in range(4):
   row=fixture_load(cycle(rank));nxt=fixture_load(next_file(rank))
   self.assertEqual(type(row['draft']).__name__,'dict'if rank==0 else'NoneType')
   self.assertEqual(type(nxt['draft_proof']).__name__,'dict'if rank==0 else'NoneType')
  DETAILS['retained_role_replay']=dict(original205_actual_failure_reproduced=True,new_role_sensitive_full_replay_passed=True,
   original_source=old_source,new_source=new_source,raw_files424=len(_CACHE),protocol_commits_per_rank=84,
   actual1176_zero_subtractions=result['Native_zero_subtractions'],actual_changed_TTU=result['changed_host_length_TTU_replays'],
   crossM_behavior_equal=result['crossM_behavior_equal'],numerical_variant_only=result['numerical_variant_only'],
   qualification_transferred_to_new_engine=False,new_model_or_full463_executed=False,fixture_raw_sha256=dict(_PINS))

 def test02_role_null_and_original_hard_gate_mutated_negatives(self):
  negatives=[
   ('draft-cycle-missing-own-proof',cycle(0),lambda v:v.update(draft=None)),
   ('target-cycle-fabricates-draft-proof',cycle(1),lambda v:v.update(draft=copy.deepcopy(fixture_load(cycle(0))['draft']))),
   ('draft-cycle-short-variable-list',cycle(0),lambda v:v['draft'].update(variable_num_draft_tokens=[4]*15)),
   ('draft-cycle-fabricates-target',cycle(0),lambda v:v.update(target={})),
   ('target-cycle-missing-target',cycle(1),lambda v:v.update(target=None)),
   ('target-cycle-missing-reference',cycle(1),lambda v:v.update(reference=None)),
   ('target-cycle-missing-physical-heads',cycle(1),lambda v:v['crossM'].update(new_all80=None)),
   ('target-cycle-missing-crossM',cycle(1),lambda v:v.update(crossM=None)),
   ('draft-next-missing-own-proof',next_file(0),lambda v:v.update(draft_proof=None)),
   ('target-next-fabricates-draft-proof',next_file(1),lambda v:v.update(draft_proof={})),
   ('wrong-report-role',report(1),lambda v:v.update(role='draft')),
   ('original-consensus-nonzero',cycle(0),lambda v:v['commit'].update(actual_numerical_failed=1)),
   ('original-Native-zero-corrupted',report(0),lambda v:v['traces'][7]['Native']['delta'].update(aclgraph_captures=1)),
   ('original-raw8-matcher-false',cycle(0),lambda v:v['draft'].update(exact_full8_encoded=False)),
   ('original-target-matcher-false',cycle(1),lambda v:v['reference'].update(all_physical_greedy_exact=False)),
  ]
  results=[]
  for label,path,mutate in negatives:
   fn,_=evaluator(P/'audit_usage209.py',{path:mutate})
   with self.assertRaises((RuntimeError,TypeError,KeyError,ValueError)) as error:fn(DIRECTORY)
   results.append(dict(case=label,rejected=True,error_type=type(error.exception).__name__,error=str(error.exception)))
  DETAILS['mutated_negatives']=dict(actual_audit_function_executed=True,cases=results,
   old_math_consensus_Native_and_matcher_gates_not_relaxed=True,fixture_files_modified=False)

 def test03_exact_source_derivation_launch_and_guard_identity(self):
  from check209 import require_parent
  parent=require_parent(archive=False)
  desc=read(P/'PROJECTIONS209.json');records=[]
  for row in desc['projection']:
   source=Path(row['parent']);self.assertEqual(sha(source),row['parent_sha256'])
   # Use the exact public projection algorithm used by the actual source check.
   text=source.read_text().replace(*row['namespace']);parts=[];offset=0
   for op in row['deltas']:
    self.assertEqual(text[op['start']:op['end']],op['old']);parts.extend((text[offset:op['start']],op['new']));offset=op['end']
   self.assertEqual(''.join(parts)+text[offset:]+row.get('append_text',''),(P/row['file']).read_text())
   if row['file'] not in ('base209.py','audit_usage209.py'):self.assertEqual(row['deltas'],[])
   records.append(dict(file=row['file'],deltas=len(row['deltas'])))
  G=B/'budget-diagnostics/v209-root-execution'
  guard_records=[]
  for row in desc['guard_projection']:
   donor=Path(row['parent']);self.assertEqual(sha(donor),row['parent_sha256'])
   text=donor.read_text().replace(*row['namespace']);parts=[];offset=0
   for op in row['deltas']:
    self.assertEqual(text[op['start']:op['end']],op['old']);parts.extend((text[offset:op['start']],op['new']));offset=op['end']
   self.assertEqual(''.join(parts)+text[offset:],(G/row['file']).read_text())
   guard_records.append(dict(file=row['file'],deltas=len(row['deltas'])))
  spec=importlib.util.spec_from_file_location('_V209_HOST_actual_guard',G/'guard209.py');guard=importlib.util.module_from_spec(spec)
  sys.modules[spec.name]=guard;spec.loader.exec_module(guard)
  argv=guard.launch_argv('HOST-only-source-no-launch')
  self.assertIn(str(G/'launch209.sh'),argv);self.assertNotIn(str(G/'launch205.sh'),argv)
  self.assertEqual(guard.P,P);self.assertEqual(guard.CPU,P/'CPU_RESULT209.json');self.assertEqual(guard.FREEZE,P/'FROZEN_SOURCE209.json')
  self.assertEqual(guard.ARCHIVE,P/'PREPARED_SOURCE_ARCHIVE209.tar.gz')
  self.assertEqual(guard.MAX_SECONDS,7200);self.assertEqual(guard.MAX_TAG_BYTES,256*1024**2);self.assertEqual(guard.MAX_AGGREGATE_BYTES,512*1024**2)
  self.assertEqual(guard.guard_inputs(),{str(G/n):sha(G/n) for n in ('guard209.py','launch209.sh','startup209.sh','GUARD_INPUTS209.json')})
  self.assertFalse((P/'FROZEN_SOURCE209.json').exists());self.assertFalse((P/'actual').exists())
  for n in ('launch209.sh','startup209.sh'):
   text=(G/n).read_text();self.assertIn('20261015',text);self.assertIn(P.name,text);self.assertNotIn('tree-causal80-small-cohort-single-writer-full-v205',text)
  from base209 import PARENT_ACTION
  self.assertEqual(PARENT_ACTION,'split192-candidate-full463')
  runtime=ast.parse((P/'runtime209.py').read_text())
  statements=[n for n in ast.walk(runtime)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='spec'for t in n.targets)]
  self.assertEqual(len(statements),1)
  parent_identity={'HOST':'actualP148bound-source'};ns=dict(PARENT_ACTION=PARENT_ACTION,readiness={'parent':{'identity':parent_identity}})
  exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),'V209-actual-runtime-permit-binding','exec'),ns)
  self.assertEqual(ns['spec']['action'],'split192-candidate-full463');self.assertIs(ns['spec']['identity'],parent_identity)
  DETAILS['source_and_guard']=dict(parent=parent,projection=records,guard_projection=guard_records,actual_launch_argv_only_constructed=argv,
   prospective_freeze_and_archive_exact_paths=True,ROOT_missing_no_launch=True,model_called=False,
   self_ROOT_lock_original_supervisor_retained=True,original177_scoped_storage_retained=True)
