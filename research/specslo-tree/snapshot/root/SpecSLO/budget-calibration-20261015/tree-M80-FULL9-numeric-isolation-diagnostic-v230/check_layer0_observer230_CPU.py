import json,sys
from pathlib import Path
def main():
    import torch
    from layer0_observer230 import observe
    torch.set_num_threads(1);torch.manual_seed(0)
    layer=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.ReLU(),torch.nn.Linear(16,8));x=torch.randn(40,8)
    expected=layer(x);actual,taps=observe(layer,lambda:layer(x),rows=40,query_row=0)
    assert torch.equal(actual,expected)and len(taps)==4 and all(t.shape[0]in(8,16)for t in taps.values())
    assert all(not m._forward_hooks for m in layer.modules())
    try:observe(layer,lambda:(_ for _ in()).throw(RuntimeError('intentional CPU fixture error')),rows=40,query_row=0)
    except RuntimeError:pass
    else:raise AssertionError('body exception must propagate')
    assert all(not m._forward_hooks for m in layer.modules())and 'torch_npu'not in sys.modules
    out=Path(__file__).with_name('CPU230_LAYER0_OBSERVER_RESULT.json')
    with out.open('x')as f:json.dump(dict(status='passed',read_only_output_bitwise_equal=True,all_hooks_removed_success_and_exception=True,Torch_CPU=True,NPU_used=False,neural_qualified=False),f,indent=2);f.write('\n')
    print(json.dumps(dict(status='passed',NPU_used=False)))
if __name__=='__main__':main()
