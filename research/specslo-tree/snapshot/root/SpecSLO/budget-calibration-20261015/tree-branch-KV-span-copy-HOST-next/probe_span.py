"""Original coupled proof and costs with one declared target copy change."""


def prepared():
    import coupled
    from projection425 import project_function
    from overlap_cost import run as costs
    from span_install import snapshot
    anchor = "    after=stage('coupled/weights-after',lambda:fingerprints(engine))"
    return project_function(coupled.run, (
        ('from target_install import install', 'from span_install import install'),
        (anchor, "    cost=_concurrent_cost(engine,native,stage,output,prefixes,root_values,copy_state)\n" + anchor),
        ('target_passes=42,global_active_limit=64,real_draft_generation=True,',
         'target_passes=42,global_active_limit=64,real_draft_generation=True,pipeline_cost=cost,span_copy=_span_snapshot(engine),'),
        ('performance_measured=False,production_permission=False,Raw=None,Goodput=None)',
         "performance_measured=True,performance_scope='component_concurrent_API_cost_only',production_permission=False,Raw=None,Goodput=None)"),
    ), dict(_concurrent_cost=costs, _span_snapshot=snapshot))


def run(*args, **kwargs):
    return prepared()(*args, **kwargs)
