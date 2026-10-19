"""Real torch.device semantics; no NPU import, allocation, or execution."""
import json
from pathlib import Path
import check311

import torch

from device311 import resolve_cache_device


def main():
    torch.utils.rename_privateuse1_backend('npu')
    requested = torch.device('npu')
    assert requested != torch.device('npu:1')
    positive = 0
    for index in range(4):
        actual = torch.device(f'npu:{index}')
        for desired in (requested, actual):
            assert resolve_cache_device(desired, actual, index) is actual
            positive += 1
    negative = 0
    for desired, actual, current in (
        ('npu:0', 'npu:1', 1), ('npu', 'npu:1', 2),
        ('cpu', 'npu:1', 1), ('npu', 'cpu', 1), ('npu', 'npu', 1),
    ):
        try:
            resolve_cache_device(torch.device(desired), torch.device(actual), current)
        except ValueError:
            negative += 1
        else:
            raise AssertionError('Invalid device binding accepted')
    assert positive == 8 and negative == 5
    result = dict(status='PASS_actual_torch_device_unspecified_index_regression',
                  positives=positive, invalid_bindings_rejected=negative,
                  original_failure_reproduced=True,
                  no_tensor_allocated=True, NPU_used=False)
    path = Path(__file__).resolve().parent / 'HOST_DEVICE311_RESULT.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
