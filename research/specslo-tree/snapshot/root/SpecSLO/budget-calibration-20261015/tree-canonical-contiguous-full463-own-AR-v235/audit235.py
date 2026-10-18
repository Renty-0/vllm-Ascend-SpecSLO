"""Original raw audit, with declared canonical geometry and additional proofs."""
import ast
import copy
import gzip
import json
import types
from pathlib import Path

import audit209 as old


def original_with_new_geometry(directory, selected_windows):
    source = Path(old.__file__).read_text()
    before = ' from mapper209 import expected_geometry'
    after = ' from mapper231 import envelope as expected_geometry'
    if source.count(before) != 1:
        raise RuntimeError('ROOT235 exact original raw geometry evaluator import')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('ROOT235 entire raw-audit source inverse')
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name == 'original_audit']
    namespace = dict(vars(old))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-audit-with-declared-geometry', 'exec'), namespace)
    return namespace['original_audit'](directory, selected_windows)


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT235 actual audit: '+message)


def audit(directory, selected_windows):
    namespace = dict(vars(old), original_audit=original_with_new_geometry)
    actual = types.FunctionType(old.audit.__code__, namespace, old.audit.__name__,
                                old.audit.__defaults__, old.audit.__closure__)(directory, selected_windows)
    directory = Path(directory)
    reports = [json.loads((directory/f'rank{rank}-local-diagnostic.json').read_text()) for rank in range(4)]
    from neural_oracle231 import cases
    expected = cases()
    need(len(expected) == 159, 'all predeclared finite math cases including S29')
    target_heads = []
    for rank, report in enumerate(reports):
        proof = report['ROOT235_current_math']
        math = proof['math']
        need(proof['cases'] == 159 and proof['original_sampler_unchanged']
             and proof['original_component'] == report['component'], 'same current original and additional math')
        need(len(math['cases']) == 159, 'all actual independent math cases retained')
        heads = []
        for case, observation in zip(expected, math['cases'], strict=True):
            if rank == 0:
                need(observation is None, 'draft makes no target neural claim')
                continue
            need(observation['case'] == case and observation['exact_greedy']
                 and observation['all128_KV_original_matcher_passed']
                 and observation['candidate_tokens'] == observation['reference_tokens']
                 and not observation['candidate_KV_copied'], 'all independent real-node gates preserved')
            heads.append(observation['candidate_tokens'])
        if rank:
            need(math['all_independent_prefixes_unchanged'] is True, 'all independent committed KV prefixes protected')
            need(len(math['old_layout_negative_controls']) == 1
                 and math['old_layout_negative_controls'][0]['used_as_positive_qualification'] is False,
                 'old-layout negative control is separate')
            target_heads.append(heads)
        scope = report['standalone_scope']
        need(scope['zero_new_keys'] is False and scope['canonical_same_instance_cases'] == 159
             and scope['effective_max_target_requests'] == 16 and scope['effective_max_verification_candidates'] == 64,
             'truthful extended scope and effective internal limits')
        geometry = report['ROOT235_actual_geometry']['actual_calls_by_generation']
        if rank:
            for envelope in report['windows']['traces']:
                trace = json.loads(gzip.decompress(Path(envelope['original_trace']['artifact']['path']).read_bytes()))
                rows = geometry[str(trace['source_generation'])]
                calls = trace['normal_target_calls']
                need(len(rows) == len(calls), 'each normal target has an actual canonical metadata witness')
                for row, call in zip(rows, calls, strict=True):
                    branches = call['budgets'].count(5)
                    need(row['N'] == call['N'] <= 16 and row['C'] == sum(call['budgets']) <= 64
                         and row['M'] == call['physical_M'] == 80
                         and row['logical_queries'] == call['real_query_count']
                         and row['branches'] == branches
                         and row['attention_segments'] == call['physical_S']+branches
                         and row['mask_capacity'] == (10 if branches else 9)
                         and row['copied'] is bool(branches), 'actual effective budgets, branch views and graph family')
    need(target_heads[0] == target_heads[1] == target_heads[2], 'all3 target math heads agree')
    from audit_reference218 import audit as original_reference_audit
    reference = original_reference_audit(directory)
    # Preserve all original own-history reconstruction; add the actual new ABI.
    for path in sorted((directory/'postfull-independent-FIA-AR').glob('rank[123]-cohort*.json.gz')):
        row = json.loads(gzip.decompress(path.read_bytes()))
        for step in row['steps']:
            if step['metadata'] is not None:
                need(step['metadata']['backend'] == 'canonical_FULL9_M80_S16'
                     and step['metadata']['tree_mask_shape'] == [16, 1, 9, 4096], 'actual canonical own-AR metadata')
    return dict(actual, ROOT235_same_instance_independent_cases=159,
                ROOT235_effective_requests=16, ROOT235_effective_candidates=64,
                ROOT235_actual_FIA_metadata_audited=True,
                ROOT235_independent_reference=reference,
                whole_workload_AR_qualified=reference['same_backend_exact_output_qualified'])
