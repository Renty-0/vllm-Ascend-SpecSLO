"""Fresh copy of291; resolve the concrete cache device before strict checks."""
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
OLD = B / 'tree-all-layer-prefix-dual-cache-target-cost-v291'
NEW = B / 'tree-concrete-device-span-dual-cache-target-cost-v296'


def main():
    NEW.mkdir(exist_ok=False)
    copied = []
    for source in OLD.iterdir():
        if not source.is_file():
            continue
        if source.suffix not in {'.py', '.sh'} and source.name not in {'FIXTURE291.json', 'DECLARED291_CASES.json'}:
            continue
        target = NEW / source.name.replace('291', '296')
        if source.suffix == '.json':
            target.write_bytes(source.read_bytes())
        else:
            text = source.read_text().replace(OLD.name, NEW.name).replace('291', '296')
            target.write_text(text)
        copied.append(target.name)
    path = NEW / 'static_state296.py'
    source = path.read_text()
    before = 'state.prefix296_copies=SpanCopies(caches,engine.device)'
    after = ('from device296 import resolve_cache_device\n'
             '    actual_device=resolve_cache_device(engine.device,caches[0][0].device,torch.npu.current_device())\n'
             '    state.prefix296_copies=SpanCopies(caches,actual_device)')
    assert source.count(before) == 1
    path.write_text(source.replace(before, after))
    review = dict(schema='ROOT296.source-only-concrete-cache-device-fix.v1',
        copied_from=str(OLD), renamed_files=copied,
        old_failure='engine.device is torch.device(npu) without index; actual cache tensors have indexed devices',
        change='Validate requested device against actual cache device and current NPU; pass concrete cache device to unchanged strict128-cache validator',
        all_kernel_bytes_unchanged=True, geometry_and_model_math_unchanged=True,
        previous_frozen_sources_unchanged=True, model_qualification_inherited=False,
        NPU_used=False, Goodput=None, adopted=False)
    (NEW / 'SOURCE_ONLY296_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review))


if __name__ == '__main__':
    main()
