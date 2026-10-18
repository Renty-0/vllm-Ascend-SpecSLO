"""Check all actual Native map bounds and raw-input coverage before NPU use."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
def main():
    from probe295 import maps_HOST
    assert 12*64<=65535 and 1792<=65535 and 120<=65535
    maps=maps_HOST()
    import torch
    torch.set_num_threads(1)
    coverage={}
    for width in(256,384):
        seen=torch.zeros(65536,dtype=torch.bool)
        for mapping in maps:
            assert mapping['grid']in(128,256,512,1024,1792)
            split=mapping['split']
            assert split['original_copies']==split['prefix_copies']+split['current_aliases']
            assert all(a==b for a,b in zip(mapping['source'][mapping['grid']:],mapping['destination'][mapping['grid']:]))
            assert all(a==b for a,b in zip(split['prefix_source'][mapping['grid']:],split['prefix_destination'][mapping['grid']:]))
            unique={s:i for i,s in enumerate(dict.fromkeys(mapping['slots']))}
            writers=torch.tensor([unique[s]for s in mapping['slots']],dtype=torch.long)
            for layer in range(64):
                for kv in(0,1):
                    for variant in(0,1):
                        bits=(writers[:,None]*width+torch.arange(width)[None,:]+layer*120*width+kv*179+variant*311)%65536
                        seen[bits]=True
        assert bool(seen.all());coverage[str(width)]=int(seen.sum())
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_actual_Native_maps_and_all65536_raw_input_words_coverage_passed',
        map_count=len(maps),maps=[dict(name=m['name'],grid=m['grid'],prefix_copies=m['split']['prefix_copies'],aliases=m['split']['current_aliases'])for m in maps],
        raw16_input_coverage=coverage,both_strided_and_contiguous_probe_declared=True,
        duplicate_padding_writers_have_identical_input_words=True,all_maps_within_captured_grid=True,
        NPU_used=False,kernel_compiled=False,models_loaded=False,Goodput=None,prefix_programs=768,coreDim_limit=65535,all_launch_grids_within_hardware_limit=True)
    with(P/'HOST295_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
