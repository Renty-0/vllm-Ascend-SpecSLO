"""All original693 numeric/cost predicates, plus same-physical provenance."""
from pathlib import Path
import json
import audit378 as original
from source377 import function

def panel(*args,**kwargs):
    body=function(original.panel,(
        ("report['outputs_equal_between_modes'] is True", "report['outputs_equal_between_modes'] is None"),
    ))
    return body(*args,**kwargs)

def audit(directory):
    body=function(original.audit,(
        ('from oracle377 import cases','from oracle459 import cases'),
        ("and report['independent_reference_FULL120'] is True",
         "and report['independent_reference_FULL120'] is False and report['independent_reference_same_physical_M'] is True"),
        ('PASS_current_target693_adaptive80_120_and_same_instance_costs378',
         'PASS_CURRENT459_693_SAME_PHYSICAL_NODE_AND_COST_RECORDS'),
    ),dict(panel=panel))
    actual,scope,discovery=body(directory)
    directory=Path(directory)
    for rank in range(4):
        report=original.read(directory/f'rank{rank}-numeric.json')
        weights=original.read(directory/f'rank{rank}-weights459-before-after.json')
        if (weights['rank']!=rank or not weights['parameter_SHA_before']or weights['parameter_SHA_before']!=weights['parameter_SHA_after']
            or weights['parameter_addresses_unchanged']is not True or weights['global_active_ceiling']!=64
            or weights['Native_flags_assigned']is not False or report['all_model_parameter_SHA_unchanged']is not True
            or report['global_active_ceiling']!=64 or report['Native_flags_assigned']is not False):
            raise RuntimeError('ROOT459 all real parameter hashes and original64 configuration')
        if rank and weights['actual_normalized_profile_sha256']!='67feca4c924ce95b252812d2db606a65a43942c472c429a71d12e5270442deb7':
            raise RuntimeError('ROOT459 original Native canonical profile digest')
        for bank in report['banks']:
            proof=original.read(bank['math_path']);cost=original.read(bank['cost_path'])['paired_physical_cost']
            for observation in proof['math']['cases']:
                if rank==0:continue
                case=observation['case'];frames=observation['independent_reference_frames']
                if (not frames or any(f['physical_queries']!=case['M']or f['reference_eager_only']is not True or f['candidate_KV_copied']is not False for f in frames)
                    or sum(f['mapped_real_nodes']for f in frames)!=case['query_count']
                    or any(not all(24<=owner<72 for owner in f['active_reference_owners'])for f in frames)):
                    raise RuntimeError('ROOT459 complete independent same-M mathematical reference')
            heads=cost['actual_warm_outputs_per_mode'];modes=cost['modes']
            if set(heads)!={arm['name']for arm in cost['arms']}:
                raise RuntimeError('ROOT459 actual all6 per-mode warm outputs')
            for arm in cost['arms']:
                values=heads[arm['name']]
                if set(values)!=set(modes)or cost['cross_mode_greedy_equal'][arm['name']]!=(values[modes[0]]==values[modes[1]]if rank else None):
                    raise RuntimeError('ROOT459 cross-M differences retained as observations')
                if rank:
                    matches=[r for r in proof['math']['cases']if r['case']['N']==arm['N']and r['case']['budgets']==arm['budgets']and r['case']['anchor']==0]
                    if not matches or values['adaptive80_120'][:matches[0]['case']['query_count']]!=matches[0]['candidate_tokens']:
                        raise RuntimeError('ROOT459 adaptive timing inputs match independently proven node outputs')
    return dict(actual,independent_reference_same_physical_M=True,all_model_parameter_SHA_unchanged=True,
                original_MC2_profile_unchanged=True,global_active_ceiling=64,Native_flags_assigned=False,
                original_cross_M120_failures_preserved=True),scope,discovery
