"""Reconstruct retained component metadata/verdict/encoding on HOST only.

Device hidden/KV tensors are not retained: their mathematical comparison is
an actual source gate, with raw shape/hash markers retained for attribution.
"""
import ast,copy,types,struct
from pathlib import Path
from base165 import require,checked

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
 """Execute actual Native host verifier; explicit HOST tensor/plan carriers."""
 path=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py'))
 method=copy.deepcopy(next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_verify_tree_outputs_host'))
 method.decorator_list=[]
 unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
 torch=types.SimpleNamespace(tensor=lambda values,**kw:Value(values),long='long')
 ns=dict(torch=torch,TreeVerificationOutput=lambda emitted,accepted:(emitted.values,accepted.values))
 exec(compile(ast.fix_missing_locations(unit),str(path)+'::V165-original-HOST-verdict','exec'),ns)
 plans=[types.SimpleNamespace(candidate_budget=3,parent_indices=Value([-1,0,1])) for _ in rows]
 return ns['_verify_tree_outputs_host'](rows,plans,Value(targets[:4*len(rows)]),4,'explicit-HOST')

def natural_raw(reports):
 rows=reports[0]['result']['discovery']['gates'];r0=reports[0]['result']
 require([r['owner'] for r in rows]==list(range(32,64)),'all natural parent rows')
 physical=[r['result']['phases'][0]['target']['full_physical_tokens'] for r in reports[1:]]
 require(all(v==physical[0] for v in physical),'actual target TP replicas whole physical tokens')
 emitted,accepted=original_verdict([r['parent_selected_row'] for r in rows],physical[0])
 positive=[];negative=[]
 for i,row in enumerate(rows):
  path=row['original_parent_primary_path'];dependency=row['staged_dependency'];out=row['actual_output_row'];nodes=row['accepted_nodes'];length=row['staged_dependency_length']
  require(path==[0,1,2] and length==len(dependency)==3,'actual qualified parent3 linear fixture')
  require(out==emitted[i] and nodes==[v for v in accepted[i] if v>=0]
   and row['accepted_count']==len(nodes),'raw original host-verdict reexecution from actual target head')
  observed=out[:length];frontier=out[length] if length<len(out) else -1
  gates=dict(fully_accepted=len(nodes)>=len(path),parent_proposal_matches=row['staged_parent_proposal_id']==row['parent_ticket']['proposal_id'],
   dependency_length_positive=length>0,observed_path_equal=observed==dependency,
   frontier_present=row['staged_frontier'] is not None,
   frontier_equal=row['staged_frontier'] is not None and frontier==int(row['staged_frontier']),not_will_finish=True)
  require(row['gates']==gates and row['actual_observed_dependency']==observed and row['actual_observed_frontier']==frontier
   and row['predicted_original_finish_eager_valid']==all(gates.values()),'every original finish predicate rederived from raw fields')
  (positive if all(gates.values()) else negative).append(row['owner'])
 require(r0['discovery']['promoted_owners']==positive and r0['discovery']['invalidated_owners']==negative,'actual promotion/invalid partition')
 for report in reports:
  r=report['result'];require(r['discovery']==r0['discovery'],'cross-rank raw original verdict/ticket gates')
  c=r['continuation'];tickets=c['child_commit_tickets']
  require(c['consumed_owners']==positive and [t['owner'] for t in tickets]==positive
   and all(t['eager'] is True and t['proposal_id']==rows[t['owner']-32]['child_ticket']['proposal_id'] for t in tickets),
   'original natural child ticket actually consumed exactly once')
 return dict(positive_owners=positive,negative_owners=negative,positive_count=len(positive),negative_count=len(negative),
  original_host_verdict_reexecuted=True,all_finish_predicates_recomputed=True,
  not_will_finish_scope='max_tokens64; initial completion1; this parent emits at most4',child_tickets_rechecked=True)

def representative_raw(item):
 c=item['case'];N,E=c['normal'],c['eager'];n=item['numerical'];m=item['metadata'];F,U,M=n['F'],n['U'],n['M']
 from metadata161 import FIELDS
 require(set(m['actual'])==set(m['independent_original_PA_reordered'])==set(FIELDS),'ten exact metadata fields retained')
 equality={k:m['actual'][k]==m['independent_original_PA_reordered'][k] for k in FIELDS}
 require(m['field_equal']==equality and all(equality.values()),'metadata equality independently redone')
 actual=m['actual'];tables=item['lease']['actual_leased_block_tables'];owners=list(range(N,N+E))+list(range(N));positions=[131]*E+[128]*N
 real=list(range(E))+list(range(F,F+N));dummy=sorted(set(range(U))-set(real));expected_tables=[tables[str(owner)] for owner in owners]
 own=dict(real_slots=[actual['slot_mapping'][i] for i in real]==[t[p//128]*128+p%128 for t,p in zip(expected_tables,positions,strict=True)],
  real_tables=[actual['block_tables'][i] for i in real]==expected_tables,dummy_slots=[actual['slot_mapping'][i] for i in dummy]==[-1]*len(dummy),
  logical_Q=actual['host_Q']==list(range(1,F+N+1)),logical_KV=actual['host_KV']==[132]*E+[1]*(F-E)+[129]*N+[1]*(U-F-N))
 require(m['own_lease']==own and all(own.values()),'actual real owner physical pages/slots/dummy/Q/KV independently redone')
 pad=m['padding_diagnostic']
 if dummy:
  require(pad['actual_physical_real_indices']==real and pad['actual_physical_dummy_indices']==dummy
   and pad['diagnostic_row_permutation']==real+dummy and pad['model_metadata_reordered'] is False
   and pad['diagnostic_view_only'] is True and len(pad['prefix0_KV_sha256'])==56,
   'component observer retains actual indices and all56 prefix0 witness hashes; model remains original order')
  require(pad['metadata']['rows']==list(range(len(real),U)) and pad['metadata']['tokens']==pad['metadata']['positions']==[0]*len(dummy)
   and pad['metadata']['slot_mapping']==[-1]*len(dummy) and pad['metadata']['context_lens']==[1]*len(dummy),
   'all original inspector dummy fields independently redone on explicitly labeled view')
 else:require(pad is None,'no unnecessary dummy witness for fully real physical rows')
 encoded=n['full_physical_encoded'];reference=n['reference_full_physical_encoded']
 require(encoded==reference and len(encoded)==M and all(len(row)==4 and all(len(v)==2 for v in row) for row in encoded),
  'whole physical encoded shape/exact reference independently redone')
 decoded=[[int(v[0])//1_000_001 for v in row]+[int(v[1])//1_000_001 for v in row] for row in encoded[:N+E]]
 scores=[[int(v[0])%1_000_001/1_000_000 for v in row]+[int(v[1])%1_000_001/1_000_000 for v in row] for row in encoded[:N+E]]
 require(n['canonical8']==n['reference_canonical8']==decoded and n['actual_full_head_tokens']==n['reference_full_head_tokens']
  and len(n['actual_full_head_tokens'])==F,'canonical8 and exact whole F head independently redone')
 # Scores on the wire are float32; raw exact bitwise matching with tensor
 # reference ran on device. HOST rederives integer-quantized score values;
 # it does not compare Python float64 arithmetic to rounded float32 equality.
 f32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
 require(n['reference_score8']==scores and n['actual_score8']==[[f32(v) for v in row] for row in scores]
  and n['actual_confidence']==[f32(sum(row)/8) for row in scores],'actual/reference score8 and confidence denominator8 independently redone')
 require(n['actual_root_tokens']==n['reference_root_tokens'],'actual new roots retained exact')
 return dict(name=c['name'],N=N,E=E,F=F,R=N,U=U,M=M,metadata10_recomputed=True,
  actual_owner_slots_tables_padding_recomputed=True,full_encoded_and_canonical8_recomputed=True,
  whole_F_head_tokens_recompared=True,quantized_reference_scores_redecoded=True,
  actual_hidden_KV_gate_markers_only=True,hidden_KV_recomputed_by_HOST=False)
