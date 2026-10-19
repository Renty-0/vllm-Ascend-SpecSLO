"""Prepare source-only C128/F64/G192 integration; fresh331 evidence required."""
from pathlib import Path
import importlib.util
import json

B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-measured-cost-risk-single-prepare-full-v316'
P=B/'tree-M160-C128-F64-global192-full-v333'


def main():
    spec=importlib.util.spec_from_file_location('_old316_source_renaming',B/'budget-diagnostics/build_ROOT316_measured_cost_full_HOST.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rename=module.rename
    P.mkdir(exist_ok=False)
    for source in OLD.iterdir():
        if source.is_file()and source.suffix in('.py','.sh'):
            target=P/source.name.replace('316','333')
            target.write_text(rename(source.read_text().replace(OLD.name,P.name),'316','333',python=source.suffix=='.py'))
    (P/'FIXTURE333.json').write_bytes((B/'tree-M160-S32-N32-target-math-cost-v326/FIXTURE326.json').read_bytes())
    # Exact geometry implementations already checked separately; this does not
    # inherit their device evidence into a new full experiment.
    for file,origin in [('oracle333','oracle326'),('baseline333','baseline326'),
                        ('metadata333','metadata326'),('frames333','frames326'),
                        ('bank333','bank326'),('audit_static_calls333','audit_static_calls326')]:
        (P/f'{file}.py').write_text(f'"""Exact M160/S32 implementation; fresh full-instance proof required."""\nfrom {origin} import *\n')
    # Preserve the new role limits and original repair bindings; the only new
    # selection predicate will consume exact331 timing evidence.
    budget=rename((B/'tree-C128-F64-global192-groups32-host-v328/budget328.py').read_text(),'328','333')
    a=budget.index('def eligible(');z=budget.index('\ndef prepared_shape',a)
    budget=budget[:a]+'from eligibility333 import eligible\n\n'+budget[z:]
    (P/'budget333.py').write_text(budget)
    eligibility=rename((B/'tree-current-prepare-cost-eligibility-host-v312/budget312.py').read_text(),'312','333')
    eligibility=eligibility[:eligibility.index('\ndef prepared_shape')]
    eligibility=eligibility.replace('24 - len(normal)','32 - len(normal)').replace('(96 - 4 * len(normal))','(128 - 4 * len(normal))')
    (P/'eligibility333.py').write_text(eligibility)
    cost=rename((B/'tree-current-prepare-cost-eligibility-host-v312/cost312.py').read_text(),'312','333')
    cost=cost.replace('ROOT311_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json','ROOT331_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json')
    cost=cost.replace("EXPECTED = '2591887d1d7a3f941ca790a2c6223accac38b2cb833ee672fa7c503858924ed7'","EXPECTED = 'PENDING_FRESH331_AUDIT_DO_NOT_LAUNCH'")
    cost=cost.replace("len(audit['target_costs']) != 33","len(audit['target_costs']) != 48")
    cost=cost.replace("rows = audit['target_costs']","rows = [row for row in audit['target_costs'] if row['branch_owners'] > 0]")
    (P/'cost333.py').write_text(cost)
    # Own constructor projection; no global config module mutation.
    (P/'config333.py').write_text('''"""Fresh normal128 constructor; original arrival/model/runtime parameters."""\nfrom dataclasses import replace\nfrom startup_config_v132 import make_config as previous\n\ndef make_config():\n    return replace(previous(),spec_rhythm_verification_budget=128)\n''')
    folders=['tree-M160-S32-N32-target-math-cost-v326','tree-M160-actual165-MC2-bridge-host-v330','tree-M160-C128-capacity-binding-v329']
    for file in('runtime333.py','check333.py'):
        path=P/file;text=path.read_text();anchor='B = P.parent\n';assert text.count(anchor)==1
        path.write_text(text.replace(anchor,anchor+'sys.path[:0]='+repr([str(B/f)for f in folders])+'\n'))
    path=P/'startup333.sh';text=path.read_text();text=text.replace('export PYTHONPATH=','export PYTHONPATH=' + ':'.join(str(B/f)for f in folders)+':');path.write_text(text)
    path=P/'runtime333.py';text=path.read_text();anchor='    changes = (\n';assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'        ("   from config209 import make_config", "   from config333 import make_config"),\n')
    path.write_text(text)
    # Other geometry/scope adapters remain deliberately unmodified at this
    # stage. HOST gates and exact331 evidence must precede source freezing.
    review=dict(status='SOURCE_ONLY_INCOMPLETE_NOT_FROZEN_NOT_LAUNCHABLE',
        objective='physical FIA160/S32, actual165 MC2 projections, normal128/future64/global192',
        source_parent316_retained=True,exact_326_geometry_sources_reused=True,
        genuine_draft45_postbranch20_original_component_all532_full463_ownAR_still_required=True,
        missing=['exact331 cost evidence SHA','full static_state/extension/scope/reference geometry integration','actual installed C128 binding in full runtime','HOST full regression and immutable closure'],
        NPU_used=False,Goodput=None,adopted=False)
    (P/'SOURCE_ONLY333_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
    for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
    print(json.dumps(review))

if __name__=='__main__':main()
