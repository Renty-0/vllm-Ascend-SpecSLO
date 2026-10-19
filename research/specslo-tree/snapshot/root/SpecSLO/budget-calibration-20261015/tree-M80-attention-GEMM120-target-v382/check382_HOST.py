"""Exact native plain signatures and bounded geometry predicate; no device math."""
import ast
import __future__
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(P))


def main():
    from hybrid382 import eligible,project_linear
    class Tokens:
        ndim=1
        def __init__(self,n):self.n=n
        def numel(self):return self.n
    def metadata(*,tree=True,s=16,q=10,ends=None):
        return SimpleNamespace(tree_attention=tree,sequence_lens=(512,)*s,
            actual_seq_lengths_q=tuple(range(5,81,5))if ends is None else ends,
            tree_attention_mask=SimpleNamespace(shape=(s,1,q,4096)))
    assert eligible(Tokens(80),metadata(q=10))and eligible(Tokens(80),metadata(q=11))
    negatives=[(Tokens(120),metadata(s=24)),(Tokens(80),metadata(tree=False)),
               (Tokens(80),metadata(s=24)),(Tokens(80),metadata(q=5)),
               (Tokens(80),metadata(ends=(80,))),(Tokens(79),metadata()),(Tokens(80),None)]
    assert all(eligible(x,m)is False for x,m in negatives)
    source=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_model.py').read_text()
    root=ast.parse(source);checked=[]
    for classname,method in [('NativeColumnLinear','forward'),('NativeLMHead','_project_local')]:
        cls=next(n for n in root.body if isinstance(n,ast.ClassDef)and n.name==classname)
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==method)
        namespace={};flags=__future__.annotations.compiler_flag
        exec(compile(ast.Module(body=[node],type_ignores=[]),'HOST382.original-native-function','exec',flags=flags,dont_inherit=True),namespace)
        original=namespace[method]
        with patch('inspect.getsource',return_value=ast.unparse(node)):
            projected=project_linear(original,lambda *args:None)
        assert inspect.signature(projected)==inspect.signature(original)
        assert projected.__code__.co_flags & flags==original.__code__.co_flags & flags
        checked.append(classname+'.'+method)
    import guard382,runtime382
    ns=guard382.execution_namespace();source=runtime382.worker_source();compile(source,'HOST382.actual-worker','exec')
    assert ns['SCOPE']['finite_cases']==693 and ns['SCOPE']['independent_prefixes']==216
    assert ns['ACTION']=='hybrid-M80-attention-GEMM120-693-and-paired-cost'
    assert source.count('from probe382 import run')==1 and source.count('numeric_cases=693')==2
    try:guard382.actual_audit('/nonexistent-HOST382-data')
    except FileNotFoundError:pass
    else:raise AssertionError('expected nonexistent actual output')
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST382_NATIVE_PLAIN_SIGNATURES_GEOMETRY_AND_ACTUAL_LIFECYCLE',
        exact_native_functions=checked,eligibility_cases=9,NPU_used=False,
        actual_guard_audit_imports_checked=True,models_qualified=False,Raw=None,Goodput=None)
    (P/'HOST382_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
