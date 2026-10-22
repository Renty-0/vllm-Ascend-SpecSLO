"""Add the requested stable80% SLO gate to the unchanged nine-cell comparison.

HOST arithmetic only. This wrapper issues no model/Native permission and
changes no prepared baseline source or measured metric.
"""
import hashlib,math
from pathlib import Path
import final_common_comparison as original

ORDERS=original.ORDERS
collect=original.collect

def aggregate(cells):
    value=original.aggregate(cells)
    # Complete fixed-length outputs make attained-request and good-token
    # fractions identical; reject inconsistent externally supplied metrics.
    for cell in cells:
        a=cell['metrics']['all']
        original.need(type(a['attained'])is int and 0<=a['attained']<=463,
                      'actual integer attained-request count')
        original.need(math.isclose(a['Goodput'],a['Raw']*a['attained']/463,
                      rel_tol=1e-10,abs_tol=1e-8),
                      'complete256-token request Goodput conservation')
    counts=value['method_summaries']['qualified_dynamic_tree']['attained_all']
    ratios=[n/463 for n in counts]
    stable=all(n*5>=463*4 for n in counts)
    throughput=value['stated_performance_and_full_AR_goals_met']
    value.update(schema='SpecSLO.final-common-clock-nine-cell-comparison-with-stable-SLO.v2',
        original_comparison_source_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        tree_SLO_attained_all=counts,tree_SLO_attainment_fraction_all=ratios,
        required_tree_SLO_attainment_fraction=0.8,
        tree_every_independent_repeat_meets_80_percent_SLO=stable,
        original_throughput_and_full_AR_conditions_met=throughput,
        stated_performance_and_full_AR_goals_met=throughput and stable,
        stated_all_Goodput_SLO_and_full_AR_goals_met=throughput and stable,
        SLO_rule='each of the three predeclared independent tree runs, not only the median',
        NPU_used=False,new_NN_permission_issued=False)
    return value
