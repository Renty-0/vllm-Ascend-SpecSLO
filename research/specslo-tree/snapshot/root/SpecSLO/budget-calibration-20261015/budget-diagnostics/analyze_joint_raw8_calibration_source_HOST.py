"""Read-only finite topology/Native integer bounds. No model or gain curve."""
import ast,collections,copy,hashlib,itertools,json,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-high-normal-cap4-native-families-v148';F=B/'tree-b4-joint-subtree-single-writer-diagnostic-v206'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def defs(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[copy.deepcopy(n)for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::READONLY-SOURCE-HOST','exec'),ns)
parents=ast.literal_eval(next(n.value for n in ast.parse((F/'selector206.py').read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='PARENTS'for t in n.targets)))
ns=dict(copy=copy);defs(F/'audit206.py',{'Vector','Matrix'},ns)
engine=ast.parse((N/'native_engine.py').read_text());cls=next(n for n in engine.body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine');node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_verify_tree_outputs_host'))
node.decorator_list=[];node.returns=None
for a in(*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs):a.annotation=None
ns.update(torch=types.SimpleNamespace(long='HOST-integer',tensor=lambda v,**kw:ns['Matrix'](v)),TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b))
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual-Native-integer-verifier::READONLY','exec'),ns)
layouts=[]
for budget in range(1,6):
 records=[]
 for chosen in itertools.combinations(range(8),budget):
  if any(parents[i]>=0 and parents[i]not in chosen for i in chosen):continue
  remap={v:i for i,v in enumerate(chosen)};packed=[-1 if parents[i]<0 else remap[parents[i]]for i in chosen];depth=[]
  for p in packed:depth.append(1+(depth[p]if p>=0 else 0))
  leaf=max(range(budget),key=lambda i:depth[i]);path=[];i=leaf
  while i>=0:path.append(i);i=packed[i]
  path.reverse();tokens=[100+i for i in chosen];heads=[1000+i for i in range(budget+1)]
  parent=-1
  for i in path:heads[parent+1]=tokens[i];parent=i
  outcome=ns['_verify_tree_outputs_host']([tokens],[types.SimpleNamespace(candidate_budget=budget,parent_indices=ns['Vector'](packed))],ns['Vector'](heads),4,'explicit-HOST-only')
  out=[v for v in outcome.token_ids.values[0]if v>=0];accepted=[v for v in outcome.accepted_node_indices.values[0]if v>=0]
  assert accepted==path and len(out)==max(depth)+1
  isfork=any(packed.count(i)>1 for i in set(packed))
  assert (not isfork)==(packed==list(range(-1,budget-1)))
  records.append(dict(nodes=list(chosen),packed_parents=packed,longest_path=max(depth),max_possible_delivered=len(out),family='full9'if isfork else'causal',
   proof_only_synthetic_unique_integer_verdict=True,measured_g=False,accept_all_selected_structurally_possible=not isfork))
 layouts.append(dict(B=budget,ancestor_closed_layouts=len(records),linear=sum(r['family']=='causal'for r in records),fork=sum(r['family']=='full9'for r in records),
  possible_longest_paths=sorted({r['longest_path']for r in records}),possible_max_delivered=sorted({r['max_possible_delivered']for r in records}),layouts=records))
# Run exact pure exported original envelope functions, no Native modules imported.
layoutns=dict(STATIC_NATIVE_KEYS=frozenset(tuple(x)for x in read(P/'BOUNDED_BUCKET_WHITELIST.json')['static_native_keys']))
tree=ast.parse((P/'target_layout.py').read_text())
for name in('TOKENS','SEGMENTS'):
 layoutns[name]=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)))
defs(P/'target_layout.py',{'_lengths','legacy_envelope','envelope'},layoutns)
compns=dict(QUALIFIED_KEYS=tuple(sorted(tuple(v)for v in read(P/'EXACT_QUALIFIED_KEY_INVENTORY.json')['geometries'])))
defs(P/'complete_search.py',{'_lengths','envelope'},compns)
shapes=[]
for Bv in range(1,6):
 for family in sorted({r['family']for r in layouts[Bv-1]['layouts']}):
  n=16;q=n*(Bv+1)
  try:
   m,s,dummy=(80,16,[])if(q,family)==(80,'causal')else compns['envelope'](n,q,family)if(q,family)==(64,'causal')else layoutns['envelope'](n,q,family)
   assert len(dummy)==s-n and sum(dummy)==m-q and all(2<=x<=9 for x in dummy)
   shapes.append(dict(B=Bv,N=n,C=n*Bv,Q=q,family=family,M=m,S=s,dummy=dummy,greedy_geometry_in_original44=(m,s,family)in compns['QUALIFIED_KEYS'],
    actual_existing_key_is_not_usage_qualification=True,production_173_cap_allows_allB=not(Bv==5 and n>8)))
  except ValueError as e:shapes.append(dict(B=Bv,N=n,Q=q,family=family,original_envelope_error=str(e)))
contexts=read(B/'tree-adaptive-side-prototype-v15/inputs.json')['contexts'];cohorts={}
for category in('coding','chat','summarization'):
 rows=[c for c in contexts if c['category']==category];ids=sorted({c['request_id']for c in rows});assert len(ids)==4 and len(rows)==12
 grouping={i:sorted(c['output_offset']for c in rows if c['request_id']==i)for i in ids};assert all(v==[32,96,192]for v in grouping.values())
 cohorts[category]=dict(source_ids=ids,offsets_by_identity=grouping,actual_prefix_lengths=[len(c['token_ids'])for c in rows],source_rows=12,effective_independent_identity_clusters=4,filler_observations_counted=0)
files=[F/n for n in('selector206.py','actor206.py','bank206.py','draft206.py','target206.py','audit206.py','FROZEN_SOURCE206.json')]
files += [P/n for n in('tree_entry.py','budget_policy.py','split_budget_v132.py','split_shape_v132.py','backbone_selection.py','runtime_tree_decode.py','packed_window.py','root_seed.py','fast_tree_draft_v5.py','target_layout.py','complete_search.py','EXACT_QUALIFIED_KEY_INVENTORY.json','BOUNDED_BUCKET_WHITELIST.json','tree-gain-model.json')]
files += [N/n for n in('native_engine.py','spec_rhythm.py','tree_budget.py')]+[B/'tree-adaptive-side-prototype-v15/inputs.json',B/'tree-causal80-real16-bankfix-full-v173/mapper173.py',D/'JOINT_CLOSED_B4_EXACT16_FROM197_SOURCE_ONLY_PRODUCTION_INTEGRATION_PLAN.json']
value=dict(schema='SOURCE_ONLY.raw8-closure-delivery-geometries.v1',status='readonly_source_integer_algebra',parents=list(parents),layouts=layouts,controlled_N16_shape_candidates=shapes,cohorts=cohorts,source_pins={str(p):sha(p)for p in files},
 exact_Native_integer_bound_witnesses=sum(x['ancestor_closed_layouts']for x in layouts),all_raw8_depth4=True,synthetic_integer_only_not_model_output=True,measured_gain_or_performance=False,model_or_device_used=False,current206actual_or_receipt_read=False)
out=D/'JOINT_RAW8_B1_TO5_SOURCE_GEOMETRY_INTEGER_BOUND_HOST.json';assert not out.exists();out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),summary=[{k:v for k,v in r.items()if k!='layouts'}for r in layouts],shapes=shapes)))
