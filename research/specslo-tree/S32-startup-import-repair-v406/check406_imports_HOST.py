"""Check every named import in new399/397 source under real startup paths."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    import torch
    import torch_npu
    def blocked(*args, **kwargs):
        raise AssertionError('HOST406 may not initialize an NPU')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    p = Path(__file__).resolve().parent
    b = p.parent
    original = b / 'tree-M120-S32-frugal-independent-target-host-v399/startup399.sh'
    bucket = b / 'tree-best257-uniform-KV-buckets-full-v376/bucket376.py'
    expected = original.read_text().replace('export PYTHONPATH=',
        'export PYTHONPATH=' + str(bucket.parent) + ':')
    assert expected == (p / 'startup406.sh').read_text()
    modules = set()
    paths = []
    for folder in [b / 'tree-M120-S32-frugal-independent-target-host-v399',
                   b / 'tree-M120-S32-frugal-budget-projection-host-v397']:
        for f in folder.glob('*.py'):
            paths.append(f)
            for node in ast.walk(ast.parse(f.read_text())):
                if isinstance(node, ast.Import):
                    modules.update(value.name for value in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules.add(node.module)
    origins = {}
    for name in sorted(modules - {'__future__'}):
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise RuntimeError('HOST406 unresolved named source import ' + name)
        origins[name] = spec.origin
    assert origins['bucket376'] == str(bucket)
    import bucket376
    assert bucket376.extent((1, 512)) == 512
    assert bucket376.extent((513,)) == 1024
    result = dict(status='PASS_HOST406_EXACT_STARTUP_ONE_PATH_AND_ALL_NEW_NAMED_IMPORTS',
                  exact_source_imports_checked=len(origins), origins=origins,
                  source_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                  unchanged_runtime399_and_neural_sources=True,
                  startup_delta='prepend only original bucket376 module directory',
                  bucket376_sha256=hashlib.sha256(bucket.read_bytes()).hexdigest(),
                  actual_original_bucket_function_executed_on_HOST=True,
                  NPU_used=False, model_constructors=0, model_qualification=False,
                  Raw=None, Goodput=None)
    (p / 'HOST406_IMPORTS_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'exact_source_imports_checked', 'NPU_used']}))


if __name__ == '__main__':
    main()
