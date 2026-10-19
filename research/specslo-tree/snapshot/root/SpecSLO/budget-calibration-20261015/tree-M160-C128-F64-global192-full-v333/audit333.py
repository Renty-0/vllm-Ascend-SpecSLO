"""Original raw audit, with declared canonical geometry and additional proofs."""
import ast
import copy
import gzip
import json
import types
from pathlib import Path

from audit_domain333 import old


def original_with_new_geometry(directory, selected_windows):
    source = Path(old.__file__).read_text()
    before = ' from mapper209 import expected_geometry'
    after = ' from mapper333 import envelope as expected_geometry'
    if source.count(before) != 1:
        raise RuntimeError('ROOT333 exact original raw geometry evaluator import')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('ROOT333 entire raw-audit source inverse')
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name == 'original_audit']
    from window333 import windows
    namespace = dict(vars(old),windows=windows)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-audit-with-declared-geometry', 'exec'), namespace)
    return namespace['original_audit'](directory, selected_windows)


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT333 actual audit: '+message)


def audit(directory, selected_windows):
    namespace = dict(vars(old), original_audit=original_with_new_geometry)
    audit_source=__import__('inspect').getsource(old.audit)
    projected=audit_source.replace('from empty209 import audit_trace','from empty333 import audit_trace')
    if projected==audit_source:raise RuntimeError('one new declared active48 cohort scope')
    exec(compile(projected,__file__+'::original-full-with-active48-cohorts','exec'),namespace)
    actual=namespace['audit'](directory,selected_windows)
    directory = Path(directory)
    reports = [json.loads((directory/f'rank{rank}-local-diagnostic.json').read_text()) for rank in range(4)]
    from audit_cost333 import audit as audit_current_cost
    from check333 import sha as source_sha, P as source_root
    current_cost=audit_current_cost(directory,[r['ROOT333_current_math']for r in reports],expected_source_sha256=source_sha(source_root/'FREEZE333.json'))
    from oracle333 import cases
    expected = cases()
    need(len(expected) == 532, 'all predeclared finite math cases within declared32 primary/side groups')
    target_heads = []
    for rank, report in enumerate(reports):
        proof = report['ROOT333_current_math']
        math = proof['math']
        need(proof['cases'] == 532 and proof['original_sampler_unchanged']
             and proof['original_component'] == report['component'], 'same current original and additional math')
        need(len(math['cases']) == 532, 'all actual independent math cases retained')
        heads = []
        for case, observation in zip(expected, math['cases'], strict=True):
            if rank == 0:
                need(observation is None, 'draft makes no target neural claim')
                continue
            need(observation['case'] == case and observation['exact_greedy']
                 and observation['all128_KV_original_matcher_passed']
                 and observation['hidden_bitwise_equal']and observation['KV_bitwise_equal']and observation['graph_eager_KV_bitwise_equal']
                 and observation['candidate_tokens'] == observation['reference_tokens']
                 and not observation['candidate_KV_copied'], 'all independent real-node gates preserved')
            heads.append(observation['candidate_tokens'])
        if rank:
            need(math['all_independent_prefixes_unchanged'] is True, 'all independent committed KV prefixes protected')
            from audit_static_calls333 import audit_calls
            audit_calls(expected,proof['actual_static_graph_calls'])
            need(proof['bank']['independent_neural_prefills']==96
                 and proof['bank']['all_triplet_roots_equal']and proof['bank']['all_triplet_prefix_KV_equal'],
                 'all96 fresh independent own-prefix neural evaluations')
            need(len(math['old_layout_negative_controls']) == 1
                 and math['old_layout_negative_controls'][0]['used_as_positive_qualification'] is False,
                 'old-layout negative control is separate')
            target_heads.append(heads)
        scope = report['standalone_scope']
        need(scope['zero_new_keys'] is False and scope['canonical_same_instance_cases'] == 532
             and scope['effective_max_target_requests'] == 32 and scope['effective_max_verification_candidates'] == 128,
             'truthful extended scope and effective internal limits')
        geometry = report['ROOT333_actual_geometry']['actual_calls_by_generation']
        paired=report['ROOT333_actual_geometry']['actual_paired_gain']
        need(paired['generation']is not None and bool(paired['by_generation']),'actual online paired gain observations by original generation')
        from risk333 import verify_snapshot
        verify_snapshot(paired,paired['generation'])
        if rank==0:
            draft=report['ROOT333_actual_geometry']['draft_numeric_proof']
            need(len(draft['primary_alternate45']['cases'])==45 and len(draft['postbranch20']['cases'])==20,'all actual same-instance draft and postbranch neural cases')
            for envelope in report['windows']['traces']:
                trace=json.loads(gzip.decompress(Path(envelope['original_trace']['artifact']['path']).read_bytes()))
                if trace['scope_count']==463:
                    generation=str(trace['source_generation'])
                    generated=report['ROOT333_actual_geometry']['actual_genuine_draft_counts'].get(generation,{})
                    need(generated.get('calls',0)>0 and generated['extra_raw_decisions']==6*generated['owners'] and generated['private_copy_graph_replays']==generated['calls'],
                         'full463 actually used genuine additional alternate generation with truthful extra work')
        if rank:
            need(report['ROOT333_actual_geometry']['checked_raw_byte_KV_copy_installed'] is True
                 and report['ROOT333_actual_geometry']['actual_checked_branch_layouts'] >= 172
                 and report['ROOT333_actual_geometry']['actual_warmed_row_widths'] in ([256],[384]),
                 'actual target raw-copy bindings, HOST contract checks and pre-capture compilation')
            for envelope in report['windows']['traces']:
                trace = json.loads(gzip.decompress(Path(envelope['original_trace']['artifact']['path']).read_bytes()))
                need(trace['decode_counters']['spec_rhythm_target_slo_first_enabled']==1, 'actual LIVE SLO-first policy counter')
                need(trace['decode_counters']['spec_rhythm_ROOT333_group_budget_enabled']==1, 'actual LIVE spare branch group policy counter')
                rows = geometry[str(trace['source_generation'])]
                calls = trace['normal_target_calls']
                if trace['scope_count']==463:
                    need(any(any(b>4 for b in call['budgets']) for call in calls), 'actual cost-selected branch executed; large-N capability separately proven by fresh408 math')
                need(len(rows) == len(calls), 'each normal target has an actual canonical metadata witness')
                for row, call in zip(rows, calls, strict=True):
                    branches = sum(b>4 for b in call['budgets'])
                    from projection255 import plan as physical_plan
                    ends=[];cursor=0
                    for i,q in enumerate(call['physical_query_partition']):
                        for size in ([5,call['budgets'][i]-4]if i<call['N']and call['budgets'][i]>4 else[q]):
                            cursor+=size;ends.append(cursor)
                    projection=physical_plan(ends,call['real_query_count'])
                    expected_grid=next((b for b in(128,256,512,1024,1792)if b>=row['actual_copy_slots']),0)if branches else 0
                    need(row['copy_grid']==expected_grid and ((f'|ROOT296:copy-grid:{expected_grid}'in row['actual_capture_key'])if branches else('|ROOT296:copy-grid:'not in row['actual_capture_key'])),
                         'every actual model graph has the exact validated copy-grid capture key')
                    need(row['physical_to_original']==projection['physical_to_original']
                         and row['original_to_physical']==projection['original_to_physical'],
                         'every actual model input/output permutation equals independently rebuilt physical plan')
                    need(row['N'] == call['N'] <= 32 and row['C'] == sum(call['budgets']) <= 128
                         and row['M'] == call['physical_M'] == 160
                         and row['logical_queries'] == call['real_query_count']
                         and row['branches'] == branches
                         and row['attention_segments']==32 and row['primary_side_groups']==row['N']+branches<=32
                         and row['fixed_KV_extent']==[2560]*32
                         and row['actual_q_partition']==list(range(5,161,5))
                         and row['mask_capacity'] == (11 if branches else 10)
                         and row['copied'] is bool(branches), 'actual effective budgets, branch views and graph family')
    need(target_heads[0] == target_heads[1] == target_heads[2], 'all3 target math heads agree')
    from audit_reference333 import audit as original_reference_audit
    reference = original_reference_audit(directory)
    # Preserve all original own-history reconstruction; add the actual new ABI.
    for path in sorted((directory/'postfull-independent-FIA-AR').glob('rank[123]-cohort*.json.gz')):
        row = json.loads(gzip.decompress(path.read_bytes()))
        for step in row['steps']:
            if step['metadata'] is not None:
                need(step['metadata']['backend'] == 'static_FULL_q5_M160_S32_KV2560'
                     and step['metadata']['tree_mask_shape'] == [32, 1, 10, 4096], 'actual canonical own-AR metadata')
    return dict(actual, ROOT333_same_instance_independent_cases=532, ROOT333_actual_current_target_cost=current_cost,
                ROOT333_effective_requests=32, ROOT333_effective_candidates=128,
                ROOT333_actual_FIA_metadata_audited=True,
                ROOT333_independent_reference=reference,
                whole_workload_AR_qualified=reference['same_backend_exact_output_qualified'])
