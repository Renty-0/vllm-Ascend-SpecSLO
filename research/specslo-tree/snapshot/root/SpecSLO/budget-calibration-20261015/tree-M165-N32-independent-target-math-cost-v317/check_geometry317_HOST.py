"""Run original complete Native HOST ancestry checks on all546 declared cases."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast
import copy
import json
import sys
from pathlib import Path
P=Path(__file__).resolve().parent
B=P.parent
sys.path.extend([str(B/n)for n in ('tree-M165-single-prepare-host-v313',
    'tree-uniform-q5-M165-S33-C128-host-v285','tree-single-validated-prefix-descriptors-host-v304',
    'tree-single-prepare-target-math-cost-v311','tree-alternate-chain-target-host-v269',
    'tree-static-q5-M120-node-math-v255','tree-canonical-contiguous-full463-own-AR-v235',
    'tree-contiguous-branch-tail-final-metadata-math-v234')])
import check313_HOST
import check285_HOST as original
from oracle317 import cases

def main():
    tree=ast.parse(Path(original.__file__).read_text())
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main'))
    for node in ast.walk(function):
        if isinstance(node,ast.ImportFrom)and node.module=='fixture285_HOST':node.module='fixture313_HOST'
    outer=next(n for n in function.body if isinstance(n,ast.For))
    mode_loop=next(n for n in ast.walk(outer)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='mode')
    start=next(i for i,n in enumerate(mode_loop.body)if isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='plans'for v in n.targets))
    body=copy.deepcopy(mode_loop.body[start:])
    make_calls=[n for n in ast.walk(body[0])if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='make']
    assert len(make_calls)==1 and len(make_calls[0].args)==3 and make_calls[0].args[2].value==0
    make_calls[0].args[2]=ast.parse("case['anchor']",mode='eval').body
    body+=ast.parse('verify_descriptors(e)').body
    loop=ast.For(target=ast.Name(id='case',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='declared_cases',ctx=ast.Load()),args=[],keywords=[]),
        body=ast.parse("n=case['N'];budgets=case['budgets'];mode=case['name'];ps=[len(prefixes[i])for i in range(n)]").body+body,orelse=[])
    index=function.body.index(outer);function.body[index]=loop
    for node in ast.walk(function):
        if isinstance(node,ast.Constant)and node.value=='HOST285_ANCESTRY_RESULT.json':node.value='HOST317_GEOMETRY_AND_SPLIT_RESULT.json'
    prefixes=json.loads((P/'FIXTURE317.json').read_text())['prefixes']
    namespace=dict(vars(original),P=P,prefixes=prefixes,declared_cases=cases,verify_descriptors=check313_HOST.verify_descriptors)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),__file__+'::original-Native-ancestry','exec'),namespace)
    namespace['main']()
    result=json.loads((P/'HOST317_GEOMETRY_AND_SPLIT_RESULT.json').read_text())
    assert result['cases']==546 and result['NPU_used']is False and 'torch_npu'not in sys.modules

if __name__=='__main__':main()
