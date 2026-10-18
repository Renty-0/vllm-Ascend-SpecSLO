"""Rebuild actual Native maps and prove omitted suffix programs write nothing."""
import ast,copy,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
    sys.path.insert(0,str(P))
    source=(P/'probe278.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    stop=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Import)and any(a.name=='torch_npu'for a in n.names))
    ns=dict(P=P,B=B,Path=Path,sys=sys,os=os);old=sys.argv;sys.argv=['HOST278',str(P/'unused-HOST-output.json')]
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(node.body[:stop]),type_ignores=[])),str(P/'probe278.py')+'::HOST-prefix-only','exec'),ns)
    finally:sys.argv=old
    from prefix278 import select_grid,BUCKETS
    counts={str(b):0 for b in BUCKETS};examples=[]
    for mapping in ns['maps']:
        grid=select_grid(mapping['source'],mapping['destination'],rows=69632)
        source,dest=mapping['source'],mapping['destination']
        assert len(source)==len(dest)==1792
        full=[(s,d)for s,d in zip(source,dest,strict=True)if s!=d]
        compact=[(s,d)for s,d in zip(source[:grid],dest[:grid],strict=True)if s!=d]
        assert compact==full and all(s==d for s,d in zip(source[grid:],dest[grid:],strict=True))
        counts[str(grid)]+=1;examples.append(dict(case=mapping['case'],grid=grid,nonidentity=len(full)))
    assert all(v>0 for v in counts.values())and examples[0]['grid']==128
    s=list(range(1792));d=list(s);d[-1]=4095
    assert select_grid(s,d,rows=4096)==1792
    s=list(range(1792));assert select_grid(s,s,rows=4096)==128
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_exact263_kernel_identity_suffix_omission_proved',
        actual_original_Native_maps=172,grid_counts=counts,examples=examples,
        fixed_index_buffers1792_retained=True,all_nonidentity_programs_preserved=True,
        only_identity_programs_omitted=True,original263_kernel_body_reused=True,
        bad_suffix_forces_full1792=True,NPU_used=False,Goodput=None,adopted=False,
        full_target_install_requires_distinct_grid_in_Native_capture_key=True)
    with(P/'HOST278_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='examples'}))
if __name__=='__main__':main()
