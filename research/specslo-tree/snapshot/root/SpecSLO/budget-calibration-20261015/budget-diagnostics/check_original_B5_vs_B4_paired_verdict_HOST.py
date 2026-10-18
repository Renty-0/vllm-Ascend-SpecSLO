"""Pure original greedy-verdict proof for a four-spine plus one side leaf.

This proves a combinatorial counterfactual only. It does not measure neural
accuracy, branch probabilities, serving latency or class-specific gains.
"""
import ast,collections,copy,hashlib,itertools,json,sys,types
from pathlib import Path
SOURCE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
OUTPUT=Path(__file__).with_name('ROOT_ORIGINAL_B5_B4_PAIRED_VERDICT_HOST_RESULT.json')
def main():
 import torch
 torch.set_num_threads(1)
 text=SOURCE.read_text();nodes=[]
 for name in ('_verify_tree_outputs_host','_tree_row_requires_kv_compaction'):
  node=copy.deepcopy(next(n for n in ast.walk(ast.parse(text))if isinstance(n,ast.FunctionDef)and n.name==name))
  node.decorator_list=[];nodes.append(node)
 ns=dict(torch=torch,TreeVerificationOutput=collections.namedtuple('CPUOriginalVerdict','outputs accepted'))
 exec(compile('from __future__ import annotations\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))),str(SOURCE)+'::unchanged-original-pure-greedy-bodies','exec'),ns)
 verify=ns['_verify_tree_outputs_host'];move=ns['_tree_row_requires_kv_compaction'];counts=collections.Counter();examples={}
 for parent in (-1,0,1,2):
  parents=[-1,0,1,2,parent];tokens=[10,20,30,40,50]
  options=[[tokens[i]for i,p in enumerate(parents)if p==owner]+[99,100]for owner in (-1,0,1,2,3,4)]
  for predictions in itertools.product(*options):
   tree=types.SimpleNamespace(candidate_budget=5,parent_indices=torch.tensor(parents))
   chain=types.SimpleNamespace(candidate_budget=4,parent_indices=torch.tensor(parents[:4]))
   a=verify([tokens],[tree],torch.tensor(predictions),4,torch.device('cpu'))
   b=verify([tokens[:4]],[chain],torch.tensor(predictions[:5]),4,torch.device('cpu'))
   output_a=[x for x in a.outputs[0].tolist()if x>=0];output_b=[x for x in b.outputs[0].tolist()if x>=0]
   accepted=a.accepted[0].tolist();side=int(4 in accepted)
   assert len(output_a)-len(output_b)==side
   assert output_a[:len(output_b)]==output_b
   assert int(move(accepted,5))==side and not move(b.accepted[0].tolist(),4)
   counts['cases']+=1;counts['side_selected'if side else'main_or_rejection']+=1
   examples.setdefault(str((parent,side)),dict(parent=parent,predictions=predictions,
       tree_output=output_a,chain_output=output_b,tree_path=accepted))
 assert counts['side_selected']>0 and counts['main_or_rejection']>0 and 'torch_npu'not in sys.modules
 result=dict(status='HOST_original_greedy_B5_vs_B4_counterfactual_proved',
   original_source=str(SOURCE),original_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
   counts=dict(counts),examples=examples,all_original_function_bodies_unchanged=True,
   result='For this exact spine4+side-leaf tree, untruncated B5 delivered = same-prefix B4 delivered +1 iff side node4 is accepted; otherwise equal. B4 output is always a prefix of B5 output.',
   limitations=['Not a measurement of neural branch acceptance probability',
     'Same main-node target predictions are assumed; actual neural equality needs independent qualification',
     'Terminal clipping can remove the extra delivered token; exclude terminal horizon from g estimation',
     'No serving gain or throughput prediction'],NPU_used=False,neural_qualified=False,Goodput=None)
 with OUTPUT.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],counts=dict(counts),NPU_used=False)))
if __name__=='__main__':main()
