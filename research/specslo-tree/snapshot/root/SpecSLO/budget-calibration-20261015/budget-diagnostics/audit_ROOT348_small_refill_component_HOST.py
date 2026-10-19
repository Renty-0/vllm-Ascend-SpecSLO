"""Reopen complete small-refill component files without model/framework imports."""
import ast
import copy
import datetime
import gzip
import hashlib
import json
import sys
import types
from pathlib import Path


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
SOURCE = BASE / 'tree-small-refill-incumbent-prefix-protection-v351/prefill351.py'
ORIGINAL = BASE / 'tree-causal80-small-cohort-role-aware-audit-full-v209/prefill209.py'
TAG = 'M120-B8-refill4-singleton-aware-root-v348-full463-ownAR-v1'
ACTUAL = ORIGINAL.parent / 'actual' / TAG
DIRECTORY = ACTUAL / 'live-refill-small-cohort-own-current-process'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def main():
    output = BASE / 'budget-diagnostics/ROOT348_INDEPENDENT_SMALL_REFILL_COMPONENT_AUDIT.json'
    require(not output.exists(), 'fresh independent evidence')
    freeze = read(SOURCE.parent / 'FREEZE351.json')
    require(sha(SOURCE.parent / 'FREEZE351.json') ==
            '3d37ddcc17ca8702a2035a1e52f9048bdc34145d41520b9b78c74e5c962a0008',
            'exact frozen numerical source351')
    full_freeze = BASE / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348/FREEZE348.json'
    require(sha(full_freeze) ==
            '09119c918ad6d28d137a731cf041e69f7a0135dc302169455fa944df6caab5c8',
            'exact active full348 transitive source inventory')
    full_sources = read(full_freeze)['files']
    for path in (SOURCE, ORIGINAL):
        require(sha(path) == full_sources[str(path)], 'unchanged projected/original source')
    require(sha(SOURCE) == freeze['files'][str(SOURCE)], 'same independent numerical351 source')
    # Execute only the actual declared reversible source projection and its
    # original pure-data audit function. No runtime numerical bodies execute.
    project = copy.deepcopy(next(node for node in ast.parse(SOURCE.read_text()).body
                                 if isinstance(node, ast.FunctionDef) and node.name == 'project'))
    namespace = dict(ast=ast, Path=Path, original=types.SimpleNamespace(__file__=str(ORIGINAL)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[project], type_ignores=[])),
                 str(SOURCE) + '::actual-source-project-only', 'exec'), namespace)
    projected = namespace['project']()
    audit = copy.deepcopy(next(node for node in ast.parse(projected).body
                               if isinstance(node, ast.FunctionDef) and node.name == 'audit'))
    scope = BASE / 'tree-high-normal-cap4-native-families-v148/scope_v146.py'
    require(sha(scope) == full_sources[str(scope)], 'unchanged original counter specification')
    sys.path.insert(0, str(scope.parent))
    from scope_v146 import NATIVE_COUNTERS
    require(len(NATIVE_COUNTERS) == 21, 'original21 counter names')
    namespace = dict(gzip=gzip, json=json, Path=Path, require=require,
                     NATIVE_COUNTERS=NATIVE_COUNTERS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[audit], type_ignores=[])),
                 str(SOURCE) + '::actual-original-projected-raw-audit-only', 'exec'), namespace)
    actual_audit = namespace['audit'](DIRECTORY)
    require(actual_audit['representative_sizes'] == [1, 3, 4, 8] and
            actual_audit['Native_zero_scalar_subtractions'] == 672,
            'all four sizes and two complete zero-increment sweeps')
    manifest = {}
    warm_cases = 0
    for rank in range(4):
        for size in (1, 3, 4, 8):
            for pass_index in range(3):
                path = DIRECTORY / f'rank{rank}-N{size}-pass{pass_index}.json.gz'
                manifest[str(path)] = sha(path)
            path = ACTUAL / 'small-refill351-prewarm' / f'rank{rank}-N{size}-pass0.json.gz'
            row = read(path)
            require(row['all_physical_hidden_original_matcher'] is True and
                    row['all_query_KV_original_matcher'] is True and
                    row['incumbent_prefix_protected'] is True,
                    'actual prewarm hidden/allKV/incumbent checks')
            require(row['all_query_KV_tensors'] == (56 if rank == 0 else 128),
                    'actual prewarm full role KV scope')
            require(row['next_draft']['planner']['untouched_owner_KV_protected'] is True,
                    'actual prewarm next-subset protection')
            manifest[str(path)] = sha(path)
            warm_cases += 1
        path = DIRECTORY / f'rank{rank}-component.json.gz'
        manifest[str(path)] = sha(path)
    require(warm_cases == 16, 'four actual prewarm cases on all four ranks')
    require(not any(n.split('.')[0] in ('torch', 'torch_npu', 'vllm') for n in sys.modules),
            'no framework/runtime imported')
    result = dict(status='PASS_complete_retained_current_small_refill_component_only',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  tag=TAG, actual_original_projected_audit=actual_audit,
                  formal_cases_per_rank=12, prewarm_cases_per_rank=4,
                  raw_component_files=manifest,
                  source_hashes={str(path): sha(path) for path in (SOURCE, ORIGINAL, scope)},
                  same_case_source_inverse_executed=True, source351_guard_not_used=True,
                  full348_original_initialization_used=True,
                  all408_target_nodes_complete=False, full463_complete=False,
                  whole_workload_AR_qualified=False, NPU_used=False,
                  Goodput=None, goal_achieved=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('raw_component_files', 'source_hashes')}))


if __name__ == '__main__':
    main()
