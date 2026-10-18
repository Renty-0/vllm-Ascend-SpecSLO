"""CPU-only exact visibility and fixed-extent boundary checks."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
from pathlib import Path
from dataclasses import dataclass
import sys,json,hashlib
import torch
torch.set_num_threads(1)
P=Path(__file__).resolve().parent;B=P.parent
sys.path[:0]=[str(B/'tree-contiguous-branch-tail-final-metadata-math-v234'),str(B/'tree-high-normal-cap4-native-families-v148')]
from static242 import fixed_query_metadata
@dataclass
class Meta:
    slot_mapping:object; context_lens:object; block_tables:object
    request_block_tables:object; tree_attention_mask:object
    actual_seq_lengths_q:object; sequence_lens:object
    tree_attention:bool=True; use_fused_infer_attention:bool=True

anchors=[0,127,128,1505,2555]
for anchor in anchors:
    masks=torch.ones((16,1,9,4096),dtype=torch.bool)
    tables=torch.full((16,32),-1,dtype=torch.int32);lengths=[]
    for owner in range(16):
        p=max(0,anchor-owner*7);live=(p+5+127)//128
        tables[owner,:live]=(torch.arange(live)+owner*32)%500
        for q in range(5):masks[owner,0,q,:p+q+1]=False;lengths.append(p+q+1)
    meta=Meta(torch.arange(80),torch.tensor(lengths,dtype=torch.int32),tables,tables,masks,
              tuple(range(5,81,5)),tuple(lengths[4::5]))
    result=fixed_query_metadata(meta,543)
    assert result.slot_mapping is meta.slot_mapping and result.context_lens is meta.context_lens
    assert result.actual_seq_lengths_q==meta.actual_seq_lengths_q and result.sequence_lens==(2560,)*16
    assert torch.equal(result.tree_attention_mask[:,:,:9],masks)and result.tree_attention_mask[:,:,9].all()
    assert torch.equal(result.request_block_tables,torch.where(tables>=0,tables,543))
    for owner in range(16):
        for q in range(5):
            visible=(~result.tree_attention_mask[owner,0,q]).nonzero().flatten()
            assert (tables[owner].index_select(0,visible//128)>=0).all()and (visible<2560).all()
for extent in [2561,4096]:
    meta.sequence_lens=(extent,)+(2560,)*15
    try:fixed_query_metadata(meta,543)
    except ValueError:pass
    else:raise RuntimeError('visible keys silently truncated')
meta.sequence_lens=(2560,)*16
for page in [-1,544,None]:
    try:fixed_query_metadata(meta,page)
    except ValueError:pass
    else:raise RuntimeError('invalid zero page accepted')
result=dict(status='HOST_passed',cases=5,checked_query_rows=400,anchors=anchors,
            extent2560_boundary_passed=True,unsupported_live_extents_rejected=[2561,4096],
            invalid_zero_pages_rejected=3,NPU_initialized=False,model_calls=0,
            source_sha256=hashlib.sha256((P/'static242.py').read_bytes()).hexdigest())
(P/'HOST_STATIC_METADATA242_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
