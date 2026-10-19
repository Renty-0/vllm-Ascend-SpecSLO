"""Use actual indented source lines, matching inspect.getsource on native methods."""
import ast
import __future__
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

P=Path(__file__).parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-M80-attention-GEMM120-target-v382')]


def main():
    import hybrid382
    source=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_model.py').read_text()
    root=ast.parse(source);lines=source.splitlines(keepends=True);samples=[]
    for clsname,name in [('NativeColumnLinear','forward'),('NativeLMHead','_project_local')]:
        cls=next(n for n in root.body if isinstance(n,ast.ClassDef)and n.name==clsname)
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==name)
        raw=''.join(lines[node.lineno-1:node.end_lineno])
        assert raw.startswith('    def ')
        namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'HOST384.native-annotations','exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),namespace)
        samples.append((namespace[name],raw))
    with patch('inspect.getsource',return_value=samples[0][1]):
        try:hybrid382.project_linear(samples[0][0],lambda *args:None)
        except IndentationError:pass
        else:raise AssertionError('must reproduce original indented-source error')
    from repair384 import install
    install()
    for original,raw in samples:
        with patch('inspect.getsource',return_value=raw):
            projected=hybrid382.project_linear(original,lambda *args:None)
        assert inspect.signature(projected)==inspect.signature(original)
    import guard384,runtime384
    ns=guard384.execution_namespace();worker=runtime384.worker_source();compile(worker,'HOST384.actual-worker','exec')
    assert ns['SCOPE']['finite_cases']==693 and ns['SCOPE']['independent_prefixes']==216
    assert ns['ACTION']=='hybrid-M80-GEMM120-693-cost-native-method-dedent'
    assert worker.count('from repair384 import run')==1 and worker.count('numeric_cases=693')==2
    try:guard384.actual_audit('/nonexistent-HOST384')
    except FileNotFoundError:pass
    else:raise AssertionError('expected nonexistent output after actual import')
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    v=dict(status='PASS_HOST384_ACTUAL_INDENTED_NATIVE_METHODS_AND_LIFECYCLE',
        NPU_used=False,raw_method_source_indentation_failure_reproduced=True,
        actual_native_method_signatures_preserved_after_dedent=True,
        model_qualified=False,Raw=None,Goodput=None)
    (P/'HOST384_CONSTRUCTION_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))


if __name__=='__main__':main()
