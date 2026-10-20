"""Raw original parent4 verdict/finish/consume audit; no hidden/KV reconstruction."""
import ast,copy,json,gzip,types,struct
from pathlib import Path
from base179 import require,checked
from scope179 import contract
from metadata179 import semantic

class Value:
 def __init__(self,values):self.values=values
 @property
 def ndim(self):return 1
 def numel(self):return len(self.values)
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return copy.deepcopy(self.values)
 def __getitem__(self,key):
  if isinstance(key,tuple):
   rows,cols=key;return Value([row[cols] for row in self.values[rows]])
  return Value(self.values[key])

def original_verdict(rows,targets):
 path=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py'))
 node=copy.deepcopy(next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_verify_tree_outputs_host'));node.decorator_list=[]
 ns=dict(torch=types.SimpleNamespace(tensor=lambda values,**kw:Value(values),long='long'),TreeVerificationOutput=lambda emitted,accepted:(emitted.values,accepted.values))
 unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
 exec(compile(ast.fix_missing_locations(unit),str(path)+'::V179-original-verifier','exec'),ns)
 plans=[types.SimpleNamespace(candidate_budget=4,parent_indices=Value([-1,0,1,2])) for _ in rows]
 return ns['_verify_tree_outputs_host'](rows,plans,Value(targets[:5*len(rows)]),4,'explicit-HOST')

def audit_reports(reports):
 require(len(reports)==4 and [r['rank'] for r in reports]==list(range(4)),'actual four current reports')
 results=[r['result'] for r in reports];first=results[0];gates=first['discovery']['gates']
 from scope179 import NATIVE_COUNTERS,FAILURES
 for report,result in zip(reports,results,strict=True):
  require(report['cache_released'] is True and report['Native_flags_assigned'] is False,'actual fixture cache released without assigning flags')
  require(set(result['Native_before'])==set(result['Native_after'])==set(result['Native_delta'])==set(NATIVE_COUNTERS), 'original21 Native fields retained')
  require(all(result['Native_delta'][k]==result['Native_after'][k]-result['Native_before'][k]>=0 for k in NATIVE_COUNTERS)
   and not any(result['Native_after'][k] for k in FAILURES),'actual21 monotonic differences and zero failures')

 require(first['contract']==contract() and [r['owner'] for r in gates]==list(range(16,32)),'exact finite parent4 current input domain')
 heads=[r['phases'][0]['target']['full_physical_tokens'] for r in results[1:]]
 require(all(v==heads[0] for v in heads),'actual independent TP replica whole target heads')
 emitted,accepted=original_verdict([r['parent_selected_row'] for r in gates],heads[0]);positive=[];negative=[]
 for j,row in enumerate(gates):
  require(row['original_parent_primary_path']==[0,1,2,3] and row['staged_dependency_length']==4
   and len(row['staged_dependency'])==4,'actual parent4 dependency, never inherited parent3 scope')
  out=emitted[j];nodes=[v for v in accepted[j] if v>=0];length=4
  checks=dict(fully_accepted=len(nodes)>=4,parent_proposal_matches=row['staged_parent_proposal_id']==row['parent_ticket']['proposal_id'],
   dependency_length_positive=True,observed_path_equal=out[:4]==row['staged_dependency'],frontier_present=row['staged_frontier'] is not None,
   frontier_equal=row['staged_frontier'] is not None and out[4]==int(row['staged_frontier']),not_will_finish=True)
  require(row['actual_output_row']==out and row['accepted_nodes']==nodes and row['accepted_count']==len(nodes)
   and row['actual_observed_dependency']==out[:4] and row['actual_observed_frontier']==out[4]
   and row['gates']==checks and row['predicted_original_finish_eager_valid']==all(checks.values()),'raw original target verdict and every actual finish gate independently reconstructed')
  (positive if all(checks.values()) else negative).append(row['owner'])
 for result in results:
  require(result['discovery']==first['discovery'] and result['discovery']['promoted_owners']==positive
   and result['discovery']['invalidated_owners']==negative,'all4 exact genuine promotion/invalid partition')
  if positive:
   proof=result['continuation'];require(proof['consumed_owners']==positive and proof['original_protocol_closed'] is True,'original positive child actual consume')
   require([v['owner'] for v in proof['child_commit_tickets']]==positive and all(v['eager'] is True
    and v['proposal_id']==gates[v['owner']-16]['child_ticket']['proposal_id'] for v in proof['child_commit_tickets']),
    'child consumed exactly the actual naturally promoted ticket')
  else:require(result['continuation'] is None,'no positive means no false child consume')
 if positive:
  numeric=first['union_sameM'];require(all(numeric[k] is True for k in ('sameM_pass','parent4_actual_independent_match','allphysical_union_hidden','allphysical_fourstep_hidden','full_encoded','all56_written_KV','exact_canonical8_scores_and_ledger')),'actual allphysical own same-M new parent4 usage gates')
  meta=numeric['independent_metadata'];require(semantic(meta['actual'])==semantic(meta['reference']) and meta['logical_metadata_exact'] is True,'own allocation metadata exact, cross-epoch logical metadata exact')
  from metadata179 import validate_own
  for proof in (meta['actual'],meta['reference']):
   tables={int(k):v for k,v in proof['authoritative_owner_tables'].items()}
   rebuilt=validate_own(proof['actual'],proof['independent_original_PA'],tables,proof['owners'],proof['logical_positions'],proof['block_size'])
   require(rebuilt['field_equal']==proof['field_equal'] and rebuilt['own_lease']==proof['own_lease'] and rebuilt['all_fields_exact'] is rebuilt['own_physical_mapping_exact'] is True,'raw all10 fields and actual owner slots/tables/padding independently rederived')
  encoded=numeric['full_physical_encoded'];require(encoded==numeric['reference_full_physical_encoded'] and len(encoded)==32 and all(len(row)==4 and all(len(pair)==2 for pair in row) for row in encoded),'actual whole M32x4x2 encoded retained exact to own reference')
  decoded=[[int(pair[0])//1000001 for pair in row]+[int(pair[1])//1000001 for pair in row] for row in encoded]
  scores=[[int(pair[0])%1000001/1000000 for pair in row]+[int(pair[1])%1000001/1000000 for pair in row] for row in encoded]
  f32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
  require(numeric['actual_canonical8']==decoded and numeric['reference_score8']==scores and numeric['actual_score8']==[[f32(v) for v in row] for row in scores]
   and numeric['actual_confidence']==[f32(sum(row)/8) for row in scores] and numeric['actual_root_tokens']==numeric['reference_root_tokens'],'raw canonical8/score8/confidence and actual roots independently redecoded')

  require(numeric['actual_parent_B']==4 and numeric['source_frontier_position']==132 and numeric['child_prefix']==133,'new parent4 positions explicitly qualified')
 return dict(status='passed-component-evidence' if positive and negative else 'positive-only-subset' if positive else 'negative-only-subset',
  positive_count=len(positive),negative_count=len(negative),positive_owners=positive,negative_owners=negative,
  original_raw_verdict_reexecuted=True,all_original_finish_gates_recomputed=True,true_positive_child_consumed=bool(positive),
  own_parent4_sameM_qualified=bool(positive),not_will_finish_scope='max_tokens64; initial completion1; actual parent emits at most5',
  device_hidden_KV_recomputed_by_HOST=False,actual_numeric_gate_markers_only=True,full_qualification=False,
  two_complete_zero_traces=False,production_gates_opened=False,formal_Goodput=False)

def audit_saved(directory):
 return audit_reports([json.loads(gzip.decompress((Path(directory)/f'rank{i}-component.json.gz').read_bytes())) for i in range(4)])

def numeric_raw(numeric):
 """Independently reconstruct retained encodings/metadata; tensor hashes only."""
 from metadata179 import validate_own
 require(all(numeric[k] is True for k in ('sameM_pass','parent4_actual_independent_match','allphysical_union_hidden','allphysical_fourstep_hidden','full_encoded','all56_written_KV','exact_canonical8_scores_and_ledger')),'actual new P4 complete physical numeric gates')
 require(numeric['M']==32 and numeric['F']==16 and numeric['R'] in (0,1,16) and numeric['U']=={0:16,1:20,16:32}[numeric['R']], 'finite P4 R0/R1/R16 geometry')
 meta=numeric['independent_metadata']
 for proof in (meta['actual'],meta['reference']):
  rebuilt=validate_own(proof['actual'],proof['independent_original_PA'],{int(k):v for k,v in proof['authoritative_owner_tables'].items()},proof['owners'],proof['logical_positions'],proof['block_size'])
  require(rebuilt['field_equal']==proof['field_equal'] and rebuilt['own_lease']==proof['own_lease'] and rebuilt['all_fields_exact'] is rebuilt['own_physical_mapping_exact'] is True,'actual own epoch original PA ten fields and authoritative physical pages/slots')
 require(semantic(meta['actual'])==semantic(meta['reference']),'independent fresh logical metadata exact')
 encoded=numeric['full_physical_encoded'];require(encoded==numeric['reference_full_physical_encoded'] and len(encoded)==32,'own whole shared4 M32 encoding equality')
 require(all(len(row)==4 and all(len(p)==2 for p in row) for row in encoded),'allphysical32x4x2')
 rows=[[v[0]//1000001 for v in row]+[v[1]//1000001 for v in row] for row in encoded]
 scores=[[v[0]%1000001/1000000 for v in row]+[v[1]%1000001/1000000 for v in row] for row in encoded]
 f32=lambda v:struct.unpack('f',struct.pack('f',v))[0]
 require(numeric['actual_canonical8']==rows and numeric['reference_score8']==scores and numeric['actual_score8']==[[f32(v) for v in row] for row in scores]
  and numeric['actual_confidence']==[f32(sum(row)/8) for row in scores] and numeric['actual_root_tokens']==numeric['reference_root_tokens'],'raw canonical8/score8/denominator8 and roots')
 parent=numeric['actual_parent_full_encoded'];require(parent==numeric['reference_parent_full_encoded'] and len(parent)==16 and all(len(r)==4 for r in parent),'independent actual whole parent4 M16 encoded')
 require(numeric['actual_parent_canonical8']==[[v[0]//1000001 for v in row]+[v[1]//1000001 for v in row] for row in parent],'genuine parent4 canonical8 independently redecoded')
 require(len(numeric['candidate_KV_sha256'])==len(numeric['reference_KV_sha256'])==len(numeric['parent_candidate_KV_sha256'])==len(numeric['parent_reference_KV_sha256'])==56,'all56 KV witness hashes retained')
 require(all(len(v)==64 and set(v)<=set('0123456789abcdef') for name in ('candidate_KV_sha256','reference_KV_sha256','parent_candidate_KV_sha256','parent_reference_KV_sha256') for v in numeric[name]),'actual all56 KV hash witnesses well formed; original device matcher is the numerical gate')
 return dict(R=numeric['R'],U=numeric['U'],M=32,F=16,metadata10_and_own_leases_recomputed=True,physical_encoding_and_scores_redecoded=True,all56_KV_hash_markers_retained=True,KV_hash_equal_observed=numeric['candidate_KV_sha256']==numeric['reference_KV_sha256'],parent_KV_hash_equal_observed=numeric['parent_candidate_KV_sha256']==numeric['parent_reference_KV_sha256'],original_device_matcher_gate_passed=True,hidden_KV_recomputed_by_HOST=False)
