"""HOST-only inclusive timings of real Native metadata, CPU arithmetic double.

No measured serving performance, model neural math or NPU latency is inferred.
The fixture source is read unchanged; no frozen file is written.
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,collections,copy,hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
SOURCE=B/'tree-static-q5-M120-full463-own-AR-v256/check_full256_HOST.py'
ACTUAL=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/static-M120-root-v257-LIVE-target-SLO-first-full463-ownAR-v1'
for p in(ACTUAL/'generation-progress').glob('rank1*.json'):
    v=json.loads(p.read_text())
    if v['sealed']and v['status']=='running':raise RuntimeError('do not profile HOST during measured full generation')
spec=importlib.util.spec_from_file_location('_metadata_HOST_original_fixture',SOURCE)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
node=copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
cut=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='forward'for t in n.targets))
node.name='_construct_metadata_CPU_fixture';node.body=node.body[:cut]+[ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))]
namespace=dict(vars(fixture));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE)+'::unchanged-fixture-prefix-only','exec'),namespace)
env=namespace[node.name]();engine=env['engine'];native=env['native'];tree=env['tree'];pad=env['pad'];host=env['host'];torch=env['torch']
import branch_tail255,projection255,projection247,fixed_mask_extent
times=collections.defaultdict(list)
def wrapped(owner,name,label):
    original=getattr(owner,name)
    def measured(*args,**kwargs):
        started=time.perf_counter_ns()
        try:return original(*args,**kwargs)
        finally:times[label].append((time.perf_counter_ns()-started)/1e6)
    setattr(owner,name,measured)
for owner,name,label in[(host,'make_host_tree_attention_metadata','original_HOST_metadata'),
    (engine._branch_tail_state,'prepare','static_prepare_inclusive'),(branch_tail255,'layout','branch_layout'),
    (fixed_mask_extent,'padded','mask_capacity_padding'),(projection255,'plan','Python_projection_geometry'),
    (projection247.CACHE,'get','immutable_index_cache'),(engine,'_cache_slot_mapping','cache_slot_mapping')]:wrapped(owner,name,label)
results=[]
for label,budgets in [('N24-allB4',[4]*24),('N16-allB4',[4]*16),('N4-B5',[5]*4),('N8-mixed',[1,2,3,4,5,4,3,2])]:
    raw=tree.cached_cpu_tree_speculation_plan(2,4,127,4096,candidate_budget=8)
    plans=[tree.pack_selected_tree_plan(raw,list(range(min(b,4)))+([7]if b==5 else[]))for b in budgets]
    roots=list(range(41,41+len(budgets)));rows=[[103+17*i+j for j in range(b)]for i,b in enumerate(budgets)]
    def one():
        pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(len(plans))),[],[])
        return native.NativePearlEngine.target_tree_forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
    one();times.clear();total=[]
    for _ in range(80):
        started=time.perf_counter_ns();result=one();total.append((time.perf_counter_ns()-started)/1e6)
        assert result['query_count']==120 and result['num_draft_tokens']==budgets
    results.append(dict(case=label,iterations=80,total_CPU_forward_median_ms=statistics.median(total),
        inclusive_component_median_ms={k:statistics.median(v)for k,v in times.items()},
        calls_by_component={k:len(v)for k,v in times.items()}))
assert 'torch_npu'not in sys.modules
value=dict(schema='ROOT.HOST-only-real-M120-metadata-inclusive-timings.v1',status='HOST_components_completed',
    fixture_source=str(SOURCE),fixture_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),results=results,
    components_overlap_do_not_sum=True,model='CPU arithmetic double; no attention/neural computation or actual graph',
    actual_NPU_used=False,NPU_latency_or_Goodput_inferred=False,production_qualified=False)
output=D/'ROOT_M120_METADATA_HOST_COMPONENT_TIMING.json'
with output.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
