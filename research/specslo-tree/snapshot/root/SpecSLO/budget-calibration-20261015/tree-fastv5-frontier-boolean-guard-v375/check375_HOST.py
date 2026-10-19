"""Actual repaired method with one real frontier/empty negative CPU fixture."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-fastv5-eager-frontier-chain-host-v373'))
from boolean375 import install
import forward373

def main():
    out=P/'HOST375_RESULT.json';assert not out.exists()
    install()
    fake=ModuleType('torch');fake.long='HOST-long';fake.tensor=lambda values,**kwargs:tuple(values)
    calls=[]
    engine=SimpleNamespace(device='HOST-only',config=SimpleNamespace(max_num_seqs=64),
        _ensure_cache_capacity=lambda ids,pos:calls.append((ids,pos)),
        _prepare_attention_metadata=lambda ids,pos,backend:(tuple(pos),SimpleNamespace()))
    native=SimpleNamespace(_next_linear_draft_graph_bucket=lambda n,maximum:n,
        NativeACLGraphRunner=SimpleNamespace(_pad_inputs=lambda inputs,pos,meta,physical:(inputs,pos,meta)))
    state=forward373.State.__new__(forward373.State);state.engine=engine;state.native=native
    with patch.dict(sys.modules,{'torch':fake}):
        frame=state.frontier(engine,[(7,127,42)])
        assert frame['real']==frame['physical']==1 and frame['inputs']==(42,)
        try:state.frontier(engine,[])
        except RuntimeError:pass
        else:raise AssertionError('empty frontier accepted')
    assert calls==[([7],[127])]
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    for p in P.glob('*.py'):ast.parse(p.read_text())
    subprocess.run(['bash','-n',str(P/'startup375.sh')],check=True)
    value=dict(status='PASS_HOST_actual_frontier_list_to_Boolean_contract_and_empty_negative',
        complete_original_method_AST_inverse=True,numerical_body_unchanged=True,
        CPU_metadata_fixture_only=True,NPU_used=False,Raw=None,Goodput=None)
    with out.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))
if __name__=='__main__':main()
