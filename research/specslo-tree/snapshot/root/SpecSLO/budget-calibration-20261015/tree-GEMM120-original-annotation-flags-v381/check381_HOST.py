"""Rebuild the exact native signature without importing torch or touching NPUs."""
import ast
import __future__
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch
from types import SimpleNamespace

P=Path(__file__).parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-M80-original-GEMM120-controls-v380')]


def main():
    import control380
    source=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/mc2.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
              and n.name=='matmul_allreduce_add_rmsnorm_or_fallback')
    namespace={}
    flag=__future__.annotations.compiler_flag
    exec(compile(ast.Module(body=[node],type_ignores=[]),'HOST381.exact-native-function','exec',flags=flag,dont_inherit=True),namespace)
    original=namespace[node.name]
    with patch('inspect.getsource',return_value=ast.unparse(node)):
        try:control380.project_mc2(original,lambda *args:None)
        except NameError as exc:assert "torch"in str(exc)
        else:raise AssertionError('HOST must reproduce missing original annotation flags')
    from repair381 import install
    install()
    with patch('inspect.getsource',return_value=ast.unparse(node)):
        projected=control380.project_mc2(original,lambda *args:None)
    assert inspect.signature(projected)==inspect.signature(original)
    assert projected.__code__.co_flags & flag==original.__code__.co_flags & flag
    from repair381 import projected_probe
    import probe380
    # Execute the real outer branch schedule with all device callbacks skipped.
    # This checks collective stage order, never numerical model behavior.
    def schedule(function,target):
        function_source=inspect.getsource(function)if function is probe380.run else None
        if function_source is None:
            # The projected source is compiled dynamically; recover its exact
            # import-free AST by observing the original-to-repaired call below.
            original_compile=compile
            captured=[]
            def retain(source,*args,**kwargs):
                captured.append(source);return original_compile(source,*args,**kwargs)
            with patch('builtins.compile',side_effect=retain):projected_probe()
            function_source=next(s for s in captured if isinstance(s,str)and s.startswith('def run('))
        tree=ast.parse(function_source)
        class NoImports(ast.NodeTransformer):
            def visit_Import(self,node):return ast.copy_location(ast.Pass(),node)
            def visit_ImportFrom(self,node):return ast.copy_location(ast.Pass(),node)
        tree=NoImports().visit(tree)
        namespace=dict(function.__globals__)
        namespace.update(build=lambda *a,**kw:{},cases=lambda:[dict(name='N16-all4-anchor0')])
        exec(compile(ast.fix_missing_locations(tree),'HOST381.outer-protocol-only','exec'),namespace)
        labels=[]
        def stage(label,callback):
            labels.append(label)
            if label.endswith('register-observers'):return SimpleNamespace(close=lambda:None)
            if label.endswith('actual-controlled-eager-observations'):return {}
            if label.endswith('all72-committed-prefixes'):return True
            return None
        engine=SimpleNamespace(is_draft=not target,rank=1 if target else 0,_branch_tail_state=SimpleNamespace())
        namespace['run'](engine,None,stage,None,dict(prefixes=[]))
        return labels
    assert schedule(probe380.run,False)!=schedule(probe380.run,True)
    fixed=projected_probe()
    assert schedule(fixed,False)==schedule(fixed,True)
    import guard381,runtime381
    ns=guard381.execution_namespace();source=runtime381.worker_source();compile(source,'HOST381.actual-worker','exec')
    assert ns['SCOPE']['finite_cases']==3 and ns['SCOPE']['independent_prefixes']==288
    assert ns['ACTION']=='M80-GEMM120-controls-original-annotation-flags'
    assert source.count('from repair381 import run')==1 and source.count('from check381 import ready')==2
    try:guard381.actual_audit('/nonexistent-HOST381-data')
    except FileNotFoundError:pass
    else:raise AssertionError('expected nonexistent output after actual audit import')
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    value=dict(status='PASS_HOST381_EXACT_NATIVE_POSTPONED_ANNOTATIONS_AND_WORKER',
        NPU_used=False,original_failure_reproduced=True,exact_native_signature_preserved=True,
        exact_native_annotation_compiler_flag_preserved=True,
        original_collective_stage_mismatch_reproduced=True,all_role_stage_order_repair_checked=True,
        models_qualified=False,Raw=None,Goodput=None)
    (P/'HOST381_CONSTRUCTION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
