"""Structural retained finite component audit; no tensor or model re-execution."""
import json
from pathlib import Path
import statistics


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT399 retained audit: '+message)


def audit(folder):
    from domain399 import cases
    folder = Path(folder)
    expected = cases()
    proofs = []
    costs = {}
    ends = []
    total = 0
    for q in [4]*24+[3]*8:
        total += q
        ends.append(total)
    for rank in range(4):
        value = json.loads((folder/f'rank{rank}-numeric.json').read_text())
        need(value['rank'] == rank and value['production_permission'] is value['full463'] is False,
             'finite only real rank')
        need(len(value['banks']) == 5 and value['actual_model_cases'] == (200 if rank else 0),
             'actual explicit case count per role')
        need(value['target_S'] == 32 and value['reference_S'] == 24 and value['physical_M'] == 120,
             'declared candidate/reference physical shapes')
        for bank in value['banks']:
            extent = bank['extent']
            need(extent in (512, 1024, 1536, 2048, 2560), 'explicit visible extent')
            math = bank['math']
            need(len(math['cases']) == 40, 'all finite cases scheduled')
            if rank:
                need(math['all_independent_prefixes_unchanged'] is bank['all96_prefixes_protected'] is True,
                     'all96 original own prefixes protected')
                for wanted, row in zip(expected, math['cases'], strict=True):
                    need(row['case'] == wanted, 'exact finite case order and budgets')
                    need(all(row[key] is True for key in ('exact_greedy', 'hidden_bitwise_equal',
                        'KV_bitwise_equal', 'graph_eager_KV_bitwise_equal',
                        'all128_KV_original_matcher_passed', 'allphysical_graph_eager_hidden_KV')),
                        'strict complete real cross-S32/S24 and fullphysical graph/eager proofs')
                    need(row['actual_physical_S'] == 32 and row['independent_reference_S'] == 24
                         and row['actual_physical_M'] == 120 and row['candidate_KV_copied'] is False,
                         'independently computed canonical reference')
                need(len(bank['actual_static_graph_calls']) >= 40, 'all40 actual finite graph dispatches')
                for call in bank['actual_static_graph_calls']:
                    need(call['M'] == 120 and call['attention_segments'] == 32
                         and call['actual_seq_lengths_q'] == ends and call['sequence_lens'] == [extent]*32
                         and call['copy_enabled'] is False and call['used_aclgraph'] is True,
                         'actual finite S32 graph dispatch geometry')
                measured = bank['target_graph_cost']
                for kind in ('S24', 'S32'):
                    samples = measured['samples_ms'][kind]
                    need(len(samples) == 20 and all(type(v) in (int, float) and v > 0 for v in samples),
                         'actual rotated matched graph cost samples')
                    need(measured['median_ms'][kind] == statistics.median(samples)
                         and measured['p95_ms'][kind] == sorted(samples)[18], 'cost arithmetic')
                costs.setdefault(str(extent), []).append(dict(rank=rank, **measured))
        proofs.append(dict(rank=rank, scheduled_cases=200, executed_model_cases=200 if rank else 0,
                           model_dtype=value['model_dtype']))
    return dict(status='PASS_FINITE399_S32_TO_CANONICAL_S24_STRICT_MATH_AND_GRAPH_COST',
                ranks=proofs, target_cases_each=200, costs=costs,
                target_S=32, reference_S=24, M=120, global_active_request_limit_required=64,
                source_tensor_reexecution_by_HOST=False, original_full_components_executed=False,
                whole_workload_AR_qualified=False, full463=False, production_permission=False,
                Raw=None, Goodput=None, adopted=False)
