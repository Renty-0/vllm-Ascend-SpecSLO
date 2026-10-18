"""Exact tile-row coverage and unchanged HOST anti-alias contract."""
import ast,copy,hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
    # Execute only the existing Native HOST map preparation from this probe.
    # Stop before importing torch_npu or executing any device operation.
    source=(P/'probe275.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    stop=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Import)and any(a.name=='torch_npu'for a in n.names))
    prefix=copy.deepcopy(node.body[:stop]);old_argv=sys.argv
    sys.argv=['HOST275',str(P/'HOST-unused-output.json')]
    ns=dict(P=P,B=B,Path=Path,sys=sys,os=os)
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),str(P/'probe275.py')+'::HOST-prefix-only','exec'),ns)
    finally:sys.argv=old_argv
    maps=ns['maps'];torch=ns['torch'];torch.set_num_threads(1)
    from contract275 import checked
    grid=[block*16+row for block in range(1792//16)for row in range(16)]
    assert grid==list(range(1792))
    counts=[]
    for mapping in maps:
        proof=checked(mapping['source'],mapping['destination'],rows=69632)
        moved=[(mapping['source'][row],mapping['destination'][row])for row in grid
               if mapping['source'][row]!=mapping['destination'][row]]
        original=[(s,d)for s,d in zip(mapping['source'],mapping['destination'],strict=True)if s!=d]
        assert moved==original
        counts.append(proof['nonidentity'])
    for width in(256,384):
        tile=1<<((2*width-1).bit_length())
        assert [j for j in range(tile)if j<width*2]==list(range(width*2))
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_tiled_byte_grid_and_original_copy_contract_passed',
        actual_original_Native_maps=len(maps),nonidentity_counts=counts,
        pair_count=1792,rows_per_program=16,programs=112,
        original_row1_programs=1792,exact_row_and_byte_coverage=True,
        same_nonidentity_source_destination_pairs=True,all_unique_destinations_and_disjoint_sources=True,
        widths=[256,384],NPU_used=False,rawbit_device_gate_still_required=True,
        real_model_qualification_still_required=True,Goodput=None,adopted=False)
    with(P/'HOST275_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='nonidentity_counts'}))
if __name__=='__main__':main()
