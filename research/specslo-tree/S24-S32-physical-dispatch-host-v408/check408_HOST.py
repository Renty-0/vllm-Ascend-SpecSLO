"""CPU tensors compare both routes to unchanged original projectors."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import random
import sys

P = Path(__file__).resolve().parent
B = P.parent


def equal(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if is_dataclass(a):
        return equal(asdict(a), asdict(b), torch)
    if isinstance(a, dict):
        return set(a) == set(b) and all(equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y, torch) for x, y in zip(a, b))
    return a == b


def main():
    import torch
    import torch_npu
    def blocked(*args, **kwargs):
        raise AssertionError('HOST408 may not initialize an NPU')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    torch.set_num_threads(1)
    sys.path.insert(0, str(B / 'tree-M120-S32-frugal-budget-projection-host-v397'))
    from check397_HOST import Metadata, partition
    import projection408
    import static408
    rng = random.Random(0)
    narrow = wide = 0
    for trial in range(96):
        n = 25 + (trial // 2) % 8 if trial % 2 else 1 + (trial // 2) % 24
        budgets = [rng.randrange(1, 4 if n > 24 else 5) for _ in range(n)]
        while budgets.count(3) > 24 and n > 24:
            budgets[budgets.index(3)] = 2
        ends, q = partition(budgets)
        groups = len(ends)
        x = torch.arange(120, device='cpu', dtype=torch.long)
        pos = x + 512
        slots = x + 2048
        slots[q:] = -1
        tables = torch.arange(groups * 32, device='cpu', dtype=torch.int32).reshape(groups, 32) % 500
        mask = torch.ones((groups, 1, 10, 4096), device='cpu', dtype=torch.bool)
        for owner in range(groups):
            for query in range(10):
                mask[owner, 0, query, (owner * 11 + query) % 128] = False
        meta = Metadata(slots, (x % 128 + 1).to(torch.int32), tables, tables,
                        mask, ends, (128,) * groups)
        got = projection408.transform(x, pos, meta, q, 543, extent=512,
                                      real_N=n, copy_enabled=False)
        expected = (projection408.narrow if n <= 24 else projection408.wide)(
            x, pos, meta, q, 543, extent=512)
        assert equal(got, expected, torch)
        assert len(got[2].sequence_lens) == (24 if n <= 24 else 32)
        assert torch.equal(got[0].index_select(0, got[3]), x)
        assert torch.equal(got[2].slot_mapping.index_select(0, got[3]), slots)
        narrow += n <= 24
        wide += n > 24
    try:
        projection408.transform(x, pos, meta, q, 543, real_N=n, copy_enabled=True)
    except ValueError:
        pass
    else:
        raise AssertionError('wide branch-copy route accepted')
    original_install = static408.original.install
    projected = static408.construct()
    assert static408.original.install is original_install
    assert projected.__globals__['ensure_zero'] is static408.original.ensure_zero
    result = dict(status='PASS_HOST408_CPU_DUAL_PROJECTOR_EQUIVALENCE_AND_STATE_INVERSE',
                  exact_narrow_tensor_cases=narrow, exact_wide_tensor_cases=wide,
                  complete120_row_input_position_KV_mask_equivalence=True,
                  unqualified_wide_branch_copy_rejected=True,
                  static_state=static408.descriptor(), NPU_used=False,
                  model_constructors=0, Native_state_installed=False,
                  uniform_bucket_and_FULL_runtime_integration_pending=True,
                  production_permission=False, Raw=None, Goodput=None)
    (P / 'HOST408_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
