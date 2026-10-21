"""Resolve all named imports in the actual new runtime and component sources."""
import ast
import importlib.util
import json
from pathlib import Path


def main():
    p = Path(__file__).resolve().parent
    b = p.parent
    modules = set()
    source_count = 0
    for folder in [p, b / 'tree-S24-S32-whole-ticket-routing-host-v407',
                   b / 'tree-S24-S32-physical-dispatch-host-v408',
                   b / 'tree-M120-S32-frugal-independent-target-host-v399']:
        for path in folder.glob('*.py'):
            source_count += 1
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    modules.update(value.name for value in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules.add(node.module)
    origins = {}
    for name in sorted(modules - {'__future__'}):
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise RuntimeError('HOST409 unresolved named import ' + name)
        origins[name] = spec.origin
    assert origins['packing398'] == str(b / 'tree-active64-whole-proposal-segment-limit-v398/packing398.py')
    assert origins['bucket376'] == str(b / 'tree-best257-uniform-KV-buckets-full-v376/bucket376.py')
    assert origins['geometry409'] == str(p / 'geometry409.py')
    assert origins['shape409'] == str(p / 'shape409.py')
    result = dict(status='PASS_HOST409_ALL_NEW_NAMED_RUNTIME_IMPORTS_AT_ACTUAL_STARTUP',
                  sources_scanned=source_count, named_imports=len(origins), origins=origins,
                  NPU_used=False, model_constructors=0, production_permission=False,
                  Raw=None, Goodput=None)
    (p / 'HOST409_IMPORTS_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'sources_scanned', 'named_imports', 'NPU_used']}))


if __name__ == '__main__':
    main()
