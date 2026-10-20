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
    after = ' from mapper255 import envelope as expected_geometry'
    if source.count(before) != 1:
        raise RuntimeError('ROOT257 exact original raw geometry evaluator import')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('ROOT257 entire raw-audit source inverse')
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name == 'original_audit']
    from window394 import windows
    namespace = dict(vars(old),windows=windows)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-audit-with-declared-geometry', 'exec'), namespace)
    return namespace['original_audit'](directory, selected_windows)


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT257 actual audit: '+message)


def audit(directory, selected_windows):
    namespace = dict(vars(old), original_audit=original_with_new_geometry)
    audit_source=__import__('inspect').getsource(old.audit)
    projected=audit_source.replace('from empty209 import audit_trace','from empty394 import audit_trace')
    if projected==audit_source:raise RuntimeError('one new declared active64 cohort scope')
    exec(compile(projected,__file__+'::original-full-with-active64-cohorts','exec'),namespace)
    actual=namespace['audit'](directory,selected_windows)
    directory = Path(directory)
    reports = [json.loads((directory/f'rank{rank}-local-diagnostic.json').read_text()) for rank in range(4)]
    from oracle257 import cases
    expected = cases()
    need(len(expected) == 227, 'all predeclared finite math cases within declared16 primary/side groups')
    target_heads = []
    for rank, report in enumerate(reports):
        proof = report['ROOT257_current_math']
        math = proof['math']
        need(proof['cases'] == 227 and proof['original_sampler_unchanged']
             and proof['original_component'] == report['component'], 'same current original and additional math')
        need(len(math['cases']) == 227, 'all actual independent math cases retained')
        heads = []
        for case, observation in zip(expected, math['cases'], strict=True):
            if rank == 0:
                need(observation is None, 'draft makes no target neural claim')
                continue
            need(observation['case'] == case and observation['exact_greedy']
                 and observation['all128_KV_original_matcher_passed']
                 and observation['hidden_bitwise_equal']and observation['KV_bitwise_equal']
                 and observation['candidate_tokens'] == observation['reference_tokens']
                 and not observation['candidate_KV_copied'], 'all independent real-node gates preserved')
            heads.append(observation['candidate_tokens'])
        if rank:
            need(math['all_independent_prefixes_unchanged'] is True, 'all independent committed KV prefixes protected')
            from audit_static_calls257 import audit_calls
            audit_calls(expected,proof['actual_static_graph_calls'])
            need(proof['bank']['independent_neural_prefills']==72
                 and proof['bank']['all_triplet_roots_equal']and proof['bank']['all_triplet_prefix_KV_equal'],
                 'all72 fresh independent own-prefix neural evaluations')
            need(len(math['old_layout_negative_controls']) == 1
                 and math['old_layout_negative_controls'][0]['used_as_positive_qualification'] is False,
                 'old-layout negative control is separate')
            target_heads.append(heads)
        scope = report['standalone_scope']
        need(scope['zero_new_keys'] is False and scope['canonical_same_instance_cases'] == 227
             and scope['effective_max_target_requests'] == 24 and scope['effective_max_verification_candidates'] == 96,
             'truthful extended scope and effective internal limits')
        geometry = report['ROOT257_actual_geometry']['actual_calls_by_generation']
        if rank:
            for envelope in report['windows']['traces']:
                trace = json.loads(gzip.decompress(Path(envelope['original_trace']['artifact']['path']).read_bytes()))
                need(trace['decode_counters']['spec_rhythm_target_slo_first_enabled']==1, 'actual LIVE SLO-first policy counter')
                rows = geometry[str(trace['source_generation'])]
                calls = trace['normal_target_calls']
                need(len(rows) == len(calls), 'each normal target has an actual canonical metadata witness')
                for row, call in zip(rows, calls, strict=True):
                    branches = call['budgets'].count(5)
                    from projection255 import plan as physical_plan
                    ends=[];cursor=0
                    for i,q in enumerate(call['physical_query_partition']):
                        for size in ([5,1]if i<call['N']and call['budgets'][i]==5 else[q]):
                            cursor+=size;ends.append(cursor)
                    projection=physical_plan(ends,call['real_query_count'])
                    need(row['physical_to_original']==projection['physical_to_original']
                         and row['original_to_physical']==projection['original_to_physical'],
                         'every actual model input/output permutation equals independently rebuilt physical plan')
                    need(row['N'] == call['N'] <= 24 and row['C'] == sum(call['budgets']) <= 96
                         and row['M'] == call['physical_M'] == 120
                         and row['logical_queries'] == call['real_query_count']
                         and row['branches'] == branches
                         and row['attention_segments']==24 and row['primary_side_groups']==row['N']+branches<=24
                         and row['fixed_KV_extent']==[2560]*24
                         and row['actual_q_partition']==list(range(5,121,5))
                         and row['mask_capacity'] == (11 if branches else 10)
                         and row['copied'] is bool(branches), 'actual effective budgets, branch views and graph family')
    need(target_heads[0] == target_heads[1] == target_heads[2], 'all3 target math heads agree')
    from audit_reference394 import audit as original_reference_audit
    reference = original_reference_audit(directory)
    # Preserve all original own-history reconstruction; add the actual new ABI.
    for path in sorted((directory/'postfull-independent-FIA-AR').glob('rank[123]-cohort*.json.gz')):
        row = json.loads(gzip.decompress(path.read_bytes()))
        for step in row['steps']:
            if step['metadata'] is not None:
                need(step['metadata']['backend'] == 'static_FULL_q5_M120_S24_KV2560'
                     and step['metadata']['tree_mask_shape'] == [24, 1, 10, 4096], 'actual canonical own-AR metadata')
    return dict(actual, ROOT257_same_instance_independent_cases=227,
                ROOT257_effective_requests=24, ROOT257_effective_candidates=96,
                ROOT257_actual_FIA_metadata_audited=True,
                ROOT257_independent_reference=reference,
                whole_workload_AR_qualified=reference['same_backend_exact_output_qualified'])
