"""All75 exact419 cases on the existing current target and new dispatcher."""
import ast
import copy
import json
from pathlib import Path


def actor():
    import actor419
    from projection425 import project_function
    return project_function(actor419.run,(
        ('from binding419 import diagnostic','from binding425 import diagnostic'),))


def run(engine,native,stage,output):
    from binding425 import install
    binding=stage('ROOT425/existing-target-readonly-binding',lambda:install(engine,native))
    folder=Path(output)/'parent4-current-component425'
    stage('ROOT425/component-private-output',lambda:folder.mkdir(exist_ok=False) if engine.rank==0 else None)
    result=actor()(engine,native,stage,folder)
    engine._root425_bounded_actual=result
    # A genuine completed component report, separate from full-serving reports.
    # Every true field below is checked by the retained419 independent auditor.
    def retain():
        if (len(result['traces'])!=75 or result['two_complete21_zero_traces'] is not True
                or engine.cache_allocation is not None or engine.cache_block_tables is not None):
            raise RuntimeError('ROOT425 exact completed75-case component and released cache')
        value=dict(rank=engine.rank,bounded_diagnostic_completed=True,
            full463_completed=False,production_permission=False,cache_released=True,errors=[],
            component=dict(original173=engine._v173_qualified_component,
                bounded=result,current_target_binding391=binding,
                actual_metadata_matches391=engine._parent4_target391_binding['actual_metadata_matches']))
        with (folder/f'rank{engine.rank}-local-diagnostic.json').open('x') as f:json.dump(value,f,separators=(',',':'))
        return True
    stage('ROOT425/retain-complete-component-report',retain)
    def retained_audit():
        from audit419 import audit
        return audit(folder)
    audited=stage('ROOT425/all75-retained-original-audit',retained_audit)
    if not (audited['passed'] is True and audited['full_required_natural_coverage'] is True):
        raise RuntimeError('ROOT425 all natural mixed/promote/reject/leaf coverage required')
    engine._root425_bounded_audit=audited
    return dict(folder=str(folder),actual_cases_each_rank=75,retained_audit=audited,
        target_reinstalled=False,model_instance_id=id(engine.model),
        production_permission=False,Raw=None,Goodput=None)
