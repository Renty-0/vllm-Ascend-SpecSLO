"""Initialize the original draft runtime, then execute immutable probe365."""
import runpy
import torch
import torch_npu
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

torch.npu.set_device(0)
init_device_properties_triton()
runpy.run_path('/root/SpecSLO/budget-calibration-20261015/tree-primary4-top2-chain-device-v365/probe365.py', run_name='__main__')
