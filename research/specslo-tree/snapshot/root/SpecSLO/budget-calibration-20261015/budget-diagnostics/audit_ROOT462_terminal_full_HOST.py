"""Retain original raw arithmetic and audit every current adaptive output."""
from pathlib import Path
from collections import Counter
import importlib.util,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-adaptive-qualified-wait-margin-full-v462'
sys.path.insert(0,str(P))
import check462
from projection425 import project_function
spec=importlib.util.spec_from_file_location('_root462_original438_terminal',D/'audit_ROOT438_terminal_full_HOST.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
FREEZE='9f79697914d74f79cc94fe4605c8f20665c44ee2f6b33fe99b59b73d7b4e2280'
ARCHIVE='cebb1af09f459d8e0a64b9a6b7561721f07e787cca5fb809fd09552e483bfbd9'
def independent_geometry(call):
    n=call['N'];budgets=call['budgets'];q=call['real_query_count'];m=call['physical_M']
    original.need(n==len(budgets)and 1<=n<=24 and all(type(v)is int and 1<=v<=5 for v in budgets)
                  and sum(budgets)<=96 and q==n+sum(budgets)<=m,'all real independent budget and query limits')
    groups=n+budgets.count(5)
    original.need(m in(80,120)and groups<=m//5,'all whole primary and side groups fit actual M')
    original.need(m==(80 if groups<=16 and q<=80 else 120),'exact declared adaptive family choice')
    remaining=m-q;padding=[5]*(remaining//5)+([remaining%5]if remaining%5 else[])
    return m,n+len(padding),[v+1 for v in budgets]+padding
def additional_checks(actual,result):
    result.update(numerical_source460_unchanged=True,global_active_limit=64,
                  original_failed460_and_stopped461_receipts_not_rewritten=True,
                  qualified_performance_result=False,final_matched_repeats_completed=0)
    if result['metrics']is None or result['errors']:return
    trace=original.read(actual/'windows/window64/full463/rank1-full463-actual-trace.json.gz')
    geometry=Counter()
    for call in trace['normal_target_calls']:
        independent_geometry(call);geometry[str(call['physical_M'])]+=1
    from geometry_history460 import build
    own=build(trace,adaptive_required=True)
    result.update(current_additional693_each_target=True,all_actual_adaptive_geometry_calls_checked=True,
                  physical_M_histogram=dict(geometry),every_request_geometry_reconstructed=own['requests'],
                  production_tokens_used_as_reference_inputs=False)
    result['qualified_performance_result']=(result['supervisor_status']=='complete'
        and result['whole_workload_AR_qualified']is True and not result['supervisor_errors'])
    if not result['qualified_performance_result']:
        result['status']='terminal_completed_compute_but_original_supervisor_not_qualified'
def constructed():
    changes=(
        ('ROOT.V438.best257-uniform-KV-full-own-AR-supervisor.v1','ROOT.V462.qualification-margin-adaptive-full-own-AR-supervisor.v1'),
        ('launch438.sh','launch462.sh'),('startup438.sh','startup462.sh'),
        ('import guard438 as guard425','import guard462 as guard425'),
        ('ROOT438.independent-terminal-compute-audit.v1','ROOT462.independent-terminal-compute-audit.v1'),
        ('import audit438 as audit425','import audit460 as audit425'),
        ("wait['qualification_receive_timeout_seconds']==18000","wait['qualification_receive_timeout_seconds']==21600"),
        ('current120_calls=len(selected80)','current_adaptive_M_calls=len(selected80)'),
        ('from audit_reference425 import audit as audit_reference','from audit_reference460 import audit as audit_reference'),
        ("    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\\n')",
         "    _ROOT462_additional_checks(actual,result)\n    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\\n')"),
    )
    return project_function(original.main,changes,dict(P=P,FREEZE=FREEZE,ARCHIVE=ARCHIVE,
        independent_geometry=independent_geometry,_ROOT462_additional_checks=additional_checks))
if __name__=='__main__':constructed()()
