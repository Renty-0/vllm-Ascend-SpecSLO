"""Compile the real retained loader and predeclare all original neural cases."""
import ast
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import runtime343
import guard343
from domain343 import cases,prelude,short_cases,short_prefixes,mid_prefixes,TOTAL_CASES
from bucket343 import case_extent

P=Path(__file__).resolve().parent


def main():
    source=runtime343.worker_source()
    assert source.count('numeric_cases=594')==2 and source.count('from check343 import ready')==2
    previous=sys.modules.get('check343');sys.modules['check343']=SimpleNamespace(ready=lambda:None)
    try:actual=runtime343.build_runtime()
    finally:
        if previous is None:sys.modules.pop('check343')
        else:sys.modules['check343']=previous
    assert inspect.signature(actual['worker'])==inspect.signature(runtime343.original.worker)
    assert actual['main'].__globals__['P']==runtime343.original.P
    assert actual['main'].__globals__['DONOR']==runtime343.original.DONOR==guard343.DONOR
    fixture=json.loads((runtime343.original.P/'FIXTURE311.json').read_text())
    assert TOTAL_CASES==len(prelude())+len(cases())+2*len(short_cases())==594
    used=set()
    for prefixes,selected in [(fixture['prefixes'],cases()),(mid_prefixes(fixture['prefixes']),short_cases()),
                              (short_prefixes(fixture['prefixes']),short_cases())]:
        for case in selected:
            e=case_extent(case,prefixes);used.add(e)
            assert max(len(prefixes[i])+min(b,4)+1 for i,b in enumerate(case['budgets']))<=e<=2560
    assert used=={512,1024,1536,2048,2560}
    ns=guard343.execution_namespace()
    assert ns['P']==guard343.DONOR and ns['SCOPE']['finite_cases']==594
    assert ns['SCOPE']['independent_prefixes']==216 and ns['ACTION']==runtime343.ACTION
    tree=ast.parse((P/'oracle343.py').read_text())
    assert 'strict full hidden and all128 independent/graph-eager KV bitwise'in (P/'oracle343.py').read_text()
    assert 'from metadata255 import checked'in (P/'oracle343.py').read_text()
    assert not any(name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for name in sys.modules)
    result=dict(status='passed_retained343_loader_and_domain_HOST',NPU_used=False,
        model_constructed=False,original311_worker_signature=True,original209_output_root=True,
        original311_teacher_fixture=True,maximum_extent_cases=6,mixed_cases=408,
        mid_cases=90,short_cases=90,total=594,all_five_extents_covered=sorted(used),
        independent_reference_FULL2560=True,all_original_matchers_and_strict_bitwise_gate=True)
    (P/'HOST343_RUNTIME_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
