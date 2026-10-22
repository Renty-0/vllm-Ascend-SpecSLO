"""Check actual retained tensor/graph APIs without NPU initialization."""
import ast,hashlib,inspect,json
from pathlib import Path
import torch,torch_npu
from vllm_ascend.spec_decode.pearl.native_model import ACL_FORMAT_FRACTAL_NZ,NativeColumnLinear,NativeMergedColumnLinear
import linear440
P=Path(__file__).resolve().parent
assert ACL_FORMAT_FRACTAL_NZ==29
assert callable(torch_npu.npu_format_cast)and callable(torch.npu.graph)
for name in('record','synchronize','elapsed_time'):assert callable(getattr(torch.npu.Event,name))
assert callable(torch.npu.NPUGraph)and callable(torch.npu.NPUGraph.replay)
assert issubclass(NativeMergedColumnLinear,NativeColumnLinear)
assert 'self.bias'in inspect.getsource(NativeColumnLinear.__init__)
assert len(linear440.VARIANTS)==4
assert callable(torch.nn.functional.linear)
result=dict(status='PASS_HOST440_RETAINED_ACTUAL_BF16_GRAPH_EVENT_AND_MODEL_APIS',
    actual_torch_version=str(torch.__version__),actual_torch_npu_version=str(torch_npu.__version__),
    actual_NZ_format=ACL_FORMAT_FRACTAL_NZ,original_gate_up_bias_attribute_exists=True,
    actual_graph_and_event_methods_present=True,NPU_used=False,model_constructors=0,
    production_permission=False,Raw=None,Goodput=None)
with(P/'HOST440_API_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
