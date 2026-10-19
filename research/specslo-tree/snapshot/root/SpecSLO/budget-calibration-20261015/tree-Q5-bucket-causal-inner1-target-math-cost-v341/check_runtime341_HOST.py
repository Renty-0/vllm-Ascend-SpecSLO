"""Compile actual worker lifecycle and inspect strict finite diagnostic wiring."""
import ast
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import runtime341
import guard341
import geometry341

P=Path(__file__).resolve().parent


def main():
    source=runtime341.worker_source()
    assert source.count('from check341 import ready')==2
    assert source.count('numeric_cases=42')==2
    assert 'from probe341 import run' in source
    previous=sys.modules.get('check341')
    sys.modules['check341']=SimpleNamespace(ready=lambda:None)
    try:runtime=runtime341.build_runtime()
    finally:
        if previous is None:sys.modules.pop('check341')
        else:sys.modules['check341']=previous
    assert inspect.signature(runtime['worker'])==inspect.signature(runtime341.numerical.original.worker)
    assert runtime['main'].__globals__['DONOR']==guard341.DONOR
    assert runtime['main'].__globals__['P']==runtime341.numerical.OLD
    assert len(geometry341.cases())==42 and len(geometry341.arms())==6
    fixture=json.loads((runtime341.numerical.OLD/'FIXTURE326.json').read_text())
    owners=set()
    for case in geometry341.cases():
        owners.update(case['owners'])
        assert case['query_count']==5*case['N']<=120 and sum(case['budgets'])<=96
        assert geometry341.extent(case,fixture['prefixes'])>=max(len(fixture['prefixes'][i])+5 for i in case['owners'])
    assert owners==set(range(32))
    ns=guard341.execution_namespace()
    assert ns['P']==guard341.DONOR and ns['SCOPE']['finite_cases']==42 and ns['SCOPE']['models_requested'] is True
    tree=ast.parse((P/'probe341.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_inner')
    calls=[ast.unparse(n)for n in ast.walk(run)if isinstance(n,ast.Call)]
    statements=[ast.unparse(n)for n in run.body]
    bank=next(i for i,s in enumerate(statements)if s.startswith('bank = build('))
    precision=next(i for i,s in enumerate(statements)if s.startswith('causal_precision = stage('))
    assert bank<precision
    assert 'range(24)' in (P/'probe341.py').read_text()
    assert not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules)
    result=dict(status='passed_retained_runtime_and_scope_HOST',NPU_used=False,
        actual_worker_signature_preserved=True,original209_output_root=True,
        original326_prefix_fixture=True,all32_origins_covered=True,
        cases=42,cost_arms=6,cost_rotations=24,
        original_prefills_before_causal_precision_patch=True,
        full_service_or_dynamic_B_qualified=False)
    (P/'HOST341_RUNTIME_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
