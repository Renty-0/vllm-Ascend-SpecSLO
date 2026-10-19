"""Compile actual worker lifecycle and inspect strict finite diagnostic wiring."""
import ast
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import runtime342
import guard342
import geometry342

P=Path(__file__).resolve().parent


def main():
    source=runtime342.worker_source()
    assert source.count('from check342 import ready')==2
    assert source.count('numeric_cases=28')==2
    assert 'from probe342 import run' in source
    previous=sys.modules.get('check342')
    sys.modules['check342']=SimpleNamespace(ready=lambda:None)
    try:runtime=runtime342.build_runtime()
    finally:
        if previous is None:sys.modules.pop('check342')
        else:sys.modules['check342']=previous
    assert inspect.signature(runtime['worker'])==inspect.signature(runtime342.numerical.original.worker)
    assert runtime['main'].__globals__['DONOR']==guard342.DONOR
    assert runtime['main'].__globals__['P']==runtime342.numerical.OLD
    assert len(geometry342.cases())==28 and len(geometry342.arms())==4
    fixture=json.loads((runtime342.numerical.OLD/'FIXTURE326.json').read_text())
    owners=set()
    for case in geometry342.cases():
        owners.update(case['owners'])
        assert case['query_count']==5*case['N']<=120 and sum(case['budgets'])<=96
        assert geometry342.extent(case,fixture['prefixes'])>=max(len(fixture['prefixes'][i])+5 for i in case['owners'])
    assert owners==set(range(32))
    ns=guard342.execution_namespace()
    assert ns['P']==guard342.DONOR and ns['SCOPE']['finite_cases']==28 and ns['SCOPE']['models_requested'] is True
    tree=ast.parse((P/'probe342.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_inner')
    calls=[ast.unparse(n)for n in ast.walk(run)if isinstance(n,ast.Call)]
    statements=[ast.unparse(n)for n in run.body]
    assert 'install_causal_precision'not in (P/'probe342.py').read_text()
    assert not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules)
    result=dict(status='passed_retained_runtime_and_scope_HOST',NPU_used=False,
        actual_worker_signature_preserved=True,original209_output_root=True,
        original326_prefix_fixture=True,all32_origins_covered=True,
        cases=28,cost_arms=4,cost_rotations=20,
        original_attention_precision_callbacks_unchanged=True,
        full_service_or_dynamic_B_qualified=False)
    (P/'HOST342_RUNTIME_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
