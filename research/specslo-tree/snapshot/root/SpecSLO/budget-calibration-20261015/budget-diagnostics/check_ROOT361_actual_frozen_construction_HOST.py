"""Actual frozen guard/runtime construction under a model-import firewall."""
import importlib.abc
import inspect
import json
import os
from pathlib import Path
import sys


B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-Q5-bucket-completion-and-paired-mode-cost-v361'
D = B / 'budget-diagnostics'
FREEZE = 'a1dbe700b6fa857631be31878df58fd7b129de38491ae027a8823799ce81bf3f'
ARCHIVE = '3e96cc84744c376ef5855bc57cda8627323c4ba841bc3843420c906d0618c20a'


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'triton'):
            raise RuntimeError('ROOT361 HOST construction attempted model import: ' + fullname)


def main():
    output = D / 'HOST_ROOT361_ACTUAL_FROZEN_CONSTRUCTION.json'
    assert not output.exists()
    sys.meta_path.insert(0, NoModelImports())
    sys.path.insert(0, str(P))
    import guard361
    guard361.EXPECTED_FREEZE = FREEZE
    guard361.EXPECTED_ARCHIVE = ARCHIVE
    own, sources, evidence = guard361.prepared()
    assert own['target_cases_each'] == 594 and own['independent_prefixes_each_target'] == 216
    namespace = guard361.execution_namespace()
    assert namespace['MAX_SECONDS'] == 3600 and namespace['P'] == guard361.DONOR
    assert namespace['MAX_TAG_BYTES'] == 192 * 1024 ** 2
    assert namespace['MAX_AGGREGATE_BYTES'] == 3584 * 1024 ** 2
    assert callable(namespace['execute_locked'])
    os.environ['SPECSLO_V361_ROOT_FREEZE_SHA'] = FREEZE
    os.environ['SPECSLO_V344_ROOT_FREEZE_SHA'] = own['parent_freeze_sha256']
    import runtime361
    import runtime344
    actual = runtime361.build_runtime()
    before = runtime344.build_runtime()
    assert set(actual) == {'worker', 'main'}
    for key in ('worker', 'main'):
        assert inspect.signature(actual[key]) == inspect.signature(before[key])
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_actual361_frozen_guard_archive_and_runtime_construction',
                  freeze_sha256=FREEZE, archive_sha256=ARCHIVE,
                  sources=len(sources), archive_and_parent_verification=evidence,
                  actual_guard_namespace_constructed=True, original_worker_signatures_preserved=True,
                  output_root=str(namespace['P']), no_Native_qualification_assigned=True,
                  model_import_firewall=True, NPU_used=False,
                  model_or_performance_qualification_inherited=False, full594_qualified=False,
                  Raw=None, Goodput=None)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'sources', 'output_root', 'NPU_used')}))


if __name__ == '__main__':
    main()
