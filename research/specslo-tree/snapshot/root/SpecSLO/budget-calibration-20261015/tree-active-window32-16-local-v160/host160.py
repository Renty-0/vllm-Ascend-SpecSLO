"""Retain four new bounded HOST cases; no old suite or backend imports."""
import ast
import importlib.abc
import json
import sys
import time
import unittest
from check160 import local_files, external_files
from window160 import P, require, sha


class Firewall(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):
            raise ImportError('V160 HOST backend firewall: '+fullname)


def main():
    sys.meta_path.insert(0, Firewall())
    before, external = local_files(), external_files()
    start = time.perf_counter()
    for path in P.glob('*.py'):
        ast.parse(path.read_text())
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(P), pattern='test_host160.py'))
    after, external_after = local_files(), external_files()
    require(before == after and external == external_after, 'all selected bytes unchanged during HOST')
    require('torch' not in sys.modules and 'torch_npu' not in sys.modules, 'HOST imported no backend')
    value = dict(schema='V160.active-window-source-HOST.v1', status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        elapsed_seconds=time.perf_counter()-start, local_before=before, local_after=after,
        external_before=external, external_after=external_after, Torch_imported=False, NPU_used=False,
        actual_parent_source_builder_executed=True, actual_isolated_installer_with_HOST_stubs=True,
        original_decode_body_executed=False, pair_orchestration_with_HOST_stubs=True,
        actual_original_clocks_executed_on_synthetic463=True, model_constructed=False,
        device_qualification=False, full_measured=False, formal_Goodput=False)
    path = P/('HOST_ATTEMPT160_'+str(time.time_ns())+'.json')
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    if result.wasSuccessful():
        cpu=P/'CPU_RESULT160.json';require(not cpu.exists(),'preserve any earlier success proof')
        with cpu.open('xb') as f:f.write(path.read_bytes())
    print(json.dumps(dict(path=str(path),sha256=sha(path),status=value['status'])))
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    raise SystemExit(main())
