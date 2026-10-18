"""Actual Native maps and exhaustive per-page prefix span inverse."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-prefix-precopy-dual-cachewrite-probe-v289'),str(B/'tree-fused-branch-KV-row-copy-v262')]
def main():
    from spans292 import pack
    from probe289 import maps_HOST
    actual=[]
    for mapping in maps_HOST():
        d=mapping['split'];spans=pack(d['prefix_source'],d['prefix_destination'])
        assert spans['copied_rows']==d['prefix_copies']
        actual.append(dict(name=mapping['name'],**spans))
    count=0
    for owners in range(13):
        for residue in range(128):
            pairs=[]
            for owner in range(owners):
                size=(residue+owner*17)%128
                pairs.extend((owner*256+i,40000+owner*256+i)for i in range(size))
            assert len(pairs)<=1792
            pairs.extend((60000+i,60000+i)for i in range(1792-len(pairs)))
            packed=pack([a for a,b in pairs],[b for a,b in pairs])
            assert packed['span_count']<=owners
            assert packed['copied_rows']==sum(a!=b for a,b in pairs)
            count+=1
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_actual_Native10maps_and1664_page_residue_span_inverses_passed',actual_maps=actual,
        exhaustive_cases=count,max_spans=12,all_nonidentity_rows_exactly_reconstructed=True,
        program_count_hypothesis='atmost768 span-layer programs rather than1792x64 row-layer programs',
        raw_bit_device_test_still_required=True,NPU_used=False,kernel_compiled=False,Goodput=None)
    with(P/'HOST292_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='actual_maps'}))
if __name__=='__main__':main()
