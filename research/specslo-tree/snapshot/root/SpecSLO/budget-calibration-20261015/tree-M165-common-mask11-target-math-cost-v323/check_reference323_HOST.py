"""Independent complete-mask check for the prospective165 own-AR frame."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
import check_geometry323_HOST # HOST dependency paths only; no model created on import.
def main():
 from fixture323_HOST import build
 env=build();e=env['engine'];torch=env['torch']
 from static_base323 import ensure_zero
 from metadata323 import pack_q5,checked
 ensure_zero(e);results=[]
 for position in(0,127,128,1505,2555,2559):
  owners=[24]+list(range(463,495));positions=list(range(position,position+5))+list(range(5))*32
  frame=dict(tokens=[41,0,0,0,0]+[0]*160,selected=owners,ids=[owner for owner in owners for _ in range(5)],positions=positions)
  e._prepare_attention_metadata(frame['ids'],frame['positions'],True)
  x,p,m=pack_q5(e,frame,full=True);checked(m)
  expected=torch.ones((33,1,11,4096),dtype=torch.bool)
  expected[0,0,0,:position+1]=False;expected[:,0,1:5,0]=False;expected[1:,0,0,0]=False
  assert torch.equal(m.tree_attention_mask,expected)
  assert x.numel()==p.numel()==165 and m.actual_seq_lengths_q==tuple(range(5,166,5))and m.sequence_lens==(2560,)*33
  results.append(position)
 assert 'torch_npu'not in sys.modules
 result=dict(status='HOST_complete_own_reference165_masks_and_boundaries_passed',boundaries=results,full_mask_bytes_checked=True,NPU_used=False,neural_qualified=False,Goodput=None)
 with(P/'HOST_REFERENCE323_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
