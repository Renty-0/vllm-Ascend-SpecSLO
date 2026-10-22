"""CPU mathematical proof using unchanged original AST functions, no Triton import."""
import ast,copy,dataclasses,hashlib,importlib.util,itertools,json,sys,types
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import torch
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
ROOT=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust')
FIXED=ROOT/'vllm_ascend/ops/triton/spec_decode/fixed_greedy_verdict.py'
NATIVE=ROOT/'vllm_ascend/spec_decode/pearl/native_engine.py';TREE=ROOT/'vllm_ascend/spec_decode/pearl/tree.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise RuntimeError(message)
require('torch_npu' not in sys.modules and not torch.cuda.is_initialized(),'CPU-only import boundary')
fixedns=dict(torch=torch,FIXED_GREEDY_FULL_WINDOW_WIDTH=4)
fixednodes=[copy.deepcopy(n) for n in ast.parse(FIXED.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_fixed_greedy_full_window_verdict_reference','fixed_greedy_full_window_verdict')]
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*fixednodes],type_ignores=[])),str(FIXED)+'::original-CPU-AST','exec'),fixedns)
fixed=fixedns['fixed_greedy_full_window_verdict']
spec=importlib.util.spec_from_file_location('_HOST_real_original_tree_math',TREE);tree=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tree;spec.loader.exec_module(tree)
source=ast.parse(NATIVE.read_text());engine=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='NativePearlEngine')
state=copy.deepcopy(next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='PearlPipelineState'))
verifier=copy.deepcopy(next(n for n in engine.body if isinstance(n,ast.FunctionDef) and n.name=='_verify_tree_outputs_host'))
# Unwrap staticmethod only; the copied method body remains byte-for-byte AST.
verifier.decorator_list=[n for n in verifier.decorator_list if not isinstance(n,ast.Name) or n.id!='staticmethod']
truncate=copy.deepcopy(next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_truncate_completion'))
finished=copy.deepcopy(next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_finished'))
ns=dict(torch=torch,dataclass=dataclasses.dataclass,copy=copy,Sequence=Sequence,TreeVerificationOutput=tree.TreeVerificationOutput)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,state,verifier,truncate,finished],type_ignores=[])),str(NATIVE)+'::original-HOST-verifier-and-state-AST','exec'),ns)
verify=ns['_verify_tree_outputs_host'];truncate=ns['_truncate_completion'];State=ns['PearlPipelineState']
V=151936

def expand(verdict,draft):
 """Test-only mathematical expansion; never installed into any runtime."""
 n=len(draft);tokens=torch.full((n,5),-1,dtype=torch.long);paths=torch.full((n,4),-1,dtype=torch.long)
 for i,(accepted,correction) in enumerate(verdict.tolist()):
  require(0<=accepted<=3 and 0<=correction<V,'valid sentinel verdict, never a fourth candidate')
  tokens[i,:accepted]=torch.tensor(draft[i][:accepted],dtype=torch.long)
  tokens[i,accepted]=correction;paths[i,:accepted]=torch.arange(accepted)
 return tree.TreeVerificationOutput(tokens,paths)

def compare(draft,heads):
 n=len(draft);plans=[tree.pack_selected_tree_plan(tree.cached_cpu_tree_speculation_plan(2,4,128+i%4,4096,candidate_budget=3),[0,1,2]) for i in range(n)]
 require(all(p.candidate_budget==3 and p.depth==4 and p.parent_indices.tolist()==[-1,0,1] for p in plans),'real production2x4 packed primary3 plan')
 h=torch.tensor(heads,dtype=torch.long).flatten();d=torch.tensor([row+[-1] for row in draft],dtype=torch.long).flatten()
 actual=fixed(h,d);old=verify(draft,plans,h,4,torch.device('cpu'));new=expand(actual,draft)
 require(torch.equal(old.token_ids,new.token_ids) and torch.equal(old.accepted_node_indices,new.accepted_node_indices),'original raw TreeVerificationOutput exact')
 return actual,old,new

alphabet=(0,1,V-1);examples=[(list(d),list(h)) for d in itertools.product(alphabet,repeat=3) for h in itertools.product(alphabet,repeat=4)]
batches=(1,2,4,9,16,17,20,24,32);offset=0;counts=Counter();boundaries=0;batch_seen=set();prefix_state_checks=0
while offset<len(examples):
 n=min(batches[len(batch_seen)%len(batches)],len(examples)-offset)
 # Cycle through all requested batch sizes, then keep32 rather than dropping coverage.
 if len(batch_seen)==len(batches):n=min(32,len(examples)-offset)
 selected=examples[offset:offset+n];draft=[v[0] for v in selected];heads=[v[1] for v in selected]
 verdict,old,new=compare(draft,heads);batch_seen.add(n)
 for i,(k,correction) in enumerate(verdict.tolist()):
  counts[k]+=1;oldtokens=[t for t in old.token_ids[i].tolist() if t>=0];newtokens=[t for t in new.token_ids[i].tolist() if t>=0]
  for remain in range(6):
   for eos in (frozenset(),frozenset({0}),frozenset({V-1})):
    for ignore in (False,True):
     a=truncate(oldtokens,eos,remain,ignore);b=truncate(newtokens,eos,remain,ignore);require(a==b,'original truncation and EOS exact');boundaries+=1
     # Remaining0 is an already-finished boundary: no empty tree commit is legal.
     if not remain:require(a==[],'remaining0 produces no commit');continue
     state_a=State([7],prompt_length=1,max_tokens=remain,ignore_eos=ignore)
     state_b=State([7],prompt_length=1,max_tokens=remain,ignore_eos=ignore)
     state_a.apply_tree_verification(output_token_ids=a,accepted=sum(v>=0 for v in old.accepted_node_indices[i].tolist()),proposed_tokens=3)
     state_b.apply_tree_verification(output_token_ids=b,accepted=k,proposed_tokens=3)
     require(dataclasses.asdict(state_a)==dataclasses.asdict(state_b) and ns['_finished'](state_a,eos)==ns['_finished'](state_b,eos),'original tree state/accounting/finish exact')
     prefix_state_checks+=1
 offset+=n
require(dict(counts)=={0:1458,1:486,2:162,3:81},'all four first-mismatch positions covered')
require(all(n in batch_seen for n in batches),'multiple production batch sizes')
# Counterexample1: invalid real head equals sentinel, violating vocabulary invariant.
bad=fixed(torch.tensor([5,6,7,-1]),torch.tensor([5,6,7,-1])).tolist()
require(bad==[[4,-1]],'explicit invalid-head sentinel collision counterexample')
# Counterexample2: same B3 but branched plan must never take the linear adapter.
branch=tree.pack_selected_tree_plan(tree.cached_cpu_tree_speculation_plan(2,4,128,4096,candidate_budget=3),[0,1,4])
rows=[[5,6,7]];heads=torch.tensor([7,11,12,13],dtype=torch.long)
old_branch=verify(rows,[branch],heads,4,torch.device('cpu'));fused_branch=expand(fixed(heads,torch.tensor([5,6,7,-1])),rows)
require(not torch.equal(old_branch.token_ids,fused_branch.token_ids),'branched counterexample must differ')
# CPU public-wrapper validation boundaries. No NPU branch or Triton symbol invoked.
empty=fixed(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long));require(tuple(empty.shape)==(0,2),'empty fixed kernel wrapper contract')
for h,d in [(torch.tensor([1,2,3]),torch.tensor([1,2,3])),(torch.tensor([1,2,3,4],dtype=torch.int32),torch.tensor([1,2,3,4],dtype=torch.int32))]:
 try:fixed(h,d)
 except ValueError:pass
 else:raise RuntimeError('original input validation negative failed')
try:verify([],[],torch.empty(0,dtype=torch.long),4,torch.device('cpu'))
except ValueError:pass
else:raise RuntimeError('original empty tree verifier boundary lost')
require('triton' not in sys.modules and 'torch_npu' not in sys.modules and not torch.cuda.is_initialized(),'no Triton or accelerator import/init')
result=dict(schema='uniformB3-sentinel4-original-math.actualCPU.v1',passed=True,vocabulary=V,sentinel=-1,
 actual_original_functions=['fixed_greedy_full_window_verdict CPU path','_fixed_greedy_full_window_verdict_reference','NativePearlEngine._verify_tree_outputs_host','PearlPipelineState.apply_tree_verification','_truncate_completion','_finished','actual original2x4 plan + pack_selected_tree_plan'],
 valid_rows_checked=len(examples),first_mismatch_histogram=dict(sorted(counts.items())),batch_sizes=sorted(batch_seen),EOS_and_remaining_cases=boundaries,actual_tree_state_checks=prefix_state_checks,
 target_and_draft_domain='three real nonnegative token candidates and four real greedy heads in[0,V); fourth INTERNAL draft lane=-1 only; real plan packed parents[-1,0,1]; max_depth4',
 exact_raw_token_shape='N x5, final placeholder -1; effective output length k+1<=4',exact_raw_path_shape='N x4, [0,...,k-1] then -1',
 remaining_range=[0,5],EOS_sets=[[],[0],[V-1]],ignore_EOS_both_modes=True,accepted_count_before_truncation_preserved=True,
 invalid_head_counterexample=dict(heads=[5,6,7,-1],kernel=bad,valid_domain=False),
 branch_counterexample=dict(parents=branch.parent_indices.tolist(),row=rows[0],heads=heads.tolist(),tree_tokens=old_branch.token_ids.tolist(),sentinel_tokens=fused_branch.token_ids.tolist()),
 empty_fixed_wrapper_allowed_but_tree_requires_nonempty=True,NPU_kernel_executed=False,Triton_imported=False,Torch_CPU_only=True,NPU_initialized=False,model_constructed=False,protocol_qualification_claimed=False,performance_measured=False,current180_raw_read=False,
 source_pins={str(p):sha(p) for p in (FIXED,NATIVE,TREE,Path(__file__))})
p=D/'UNIFORM_B3_FIXED4_SENTINEL_ORIGINAL_MATH_CPU_EVIDENCE.json'
with p.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),rows=len(examples),first_mismatch_histogram=dict(counts),boundary_checks=boundaries,state_checks=prefix_state_checks)))
