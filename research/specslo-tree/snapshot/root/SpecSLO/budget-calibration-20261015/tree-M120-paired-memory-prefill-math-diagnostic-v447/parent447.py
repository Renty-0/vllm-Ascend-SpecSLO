"""Unchanged445 NN/timing body, explicit graph teardown after saved evidence."""
from projection425 import project_function
import forward445 as original

def release(graphs,frames,outputs):
    import torch
    if len(graphs)!=6 or set(graphs)!=set(frames)or set(graphs)!=set(outputs):
        raise RuntimeError('ROOT447 complete actual445 six graph frames before teardown')
    torch.npu.synchronize()
    for g in graphs.values():g.reset()
    graphs.clear();frames.clear();outputs.clear()

measure=project_function(original.measure,(
    ("    return dict(report=str(path),all5_lengths_math_exact=local['all_math_exact'],",
     "    _release_parent447(graphs,frames,hidden_outputs)\n"
     "    return dict(report=str(path),all5_lengths_math_exact=local['all_math_exact'],"),
),dict(_release_parent447=release))
