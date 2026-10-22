"""Construct the actual frozen synthetic guard without any device imports."""
import importlib.abc
import json
from pathlib import Path
import sys


B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-FIA-multi-owner-page-offset-invariance-v363'
D = B / 'budget-diagnostics'
FREEZE = '292e7a4a0c4a945879e298b73204c898f3ebc9c86f2b746fba7ba60a03e2083e'
ARCHIVE = '2f32e507905f95692461cbf03406f2f8b39fa0755f806a3606aeaabaa9a8bf11'


class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'triton'):
            raise RuntimeError('ROOT363 HOST guard attempted device/model import: ' + fullname)


def main():
    output = D / 'HOST_ROOT363_ACTUAL_FROZEN_GUARD.json'
    assert not output.exists()
    sys.meta_path.insert(0, NoDeviceImports())
    sys.path.insert(0, str(P))
    import guard363
    guard363.EXPECTED_FREEZE = FREEZE
    guard363.EXPECTED_ARCHIVE = ARCHIVE
    own, sources, evidence = guard363.prepared()
    namespace = guard363.execution_namespace()
    assert own['synthetic_cases'] == 640 and own['ABA_rows'] == 1920
    assert namespace['MAX_SECONDS'] == 600 and namespace['P'] == P and namespace['G'] == P
    assert namespace['MAX_TAG_BYTES'] == 64 * 1024 ** 2
    assert namespace['MAX_AGGREGATE_BYTES'] == 256 * 1024 ** 2
    assert namespace['SCOPE']['models_loaded'] is namespace['SCOPE']['formal_Goodput'] is False
    assert namespace['SCOPE']['groups'] == [24] and namespace['SCOPE']['rows'] == 1920
    assert namespace['SCOPE']['KV_heads'] == [2, 3]
    assert callable(namespace['execute_locked'])
    argv = guard363.launch_argv('HOST-source-only-no-launch')
    assert str(P / 'startup363.sh') in argv and str(B / 'serial-v1/runtime.sh') in argv
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_actual363_frozen_guard_archive_and_owned_namespace',
                  freeze_sha256=FREEZE, archive_sha256=ARCHIVE, sources=len(sources),
                  archive_and_parent_verified=evidence, original_owned_lifecycle_preserved=True,
                  actual_scope=dict(namespace['SCOPE']), output_root=str(P),
                  device_import_firewall=True, NPU_used=False, model_qualified=False,
                  numerical_offset_invariance_qualified=False, Raw=None, Goodput=None)
    with output.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'sources', 'output_root', 'NPU_used')}))


if __name__ == '__main__':
    main()
