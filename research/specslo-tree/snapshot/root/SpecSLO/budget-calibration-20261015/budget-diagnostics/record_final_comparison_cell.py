"""Bind terminal real receipts and raw metrics into one predeclared result cell."""
import argparse,hashlib,json
from pathlib import Path
from final_common_comparison import ORDERS,collect,need


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(plan_path,expected,block,position,receipt_path,output,tree_audit_path=None):
    need(sha(plan_path)==expected,'exact predeclared final command plan')
    plan=json.loads(Path(plan_path).read_text());need(len(plan['cells'])==9,'nine planned cells')
    wanted=plan['cells'][(block-1)*3+position]
    need(wanted['block']==block and wanted['position']==position
         and wanted['method']==ORDERS[block-1][position],'declared block position and method')
    receipt=json.loads(Path(receipt_path).read_text())
    need(receipt['status']=='complete'and receipt['errors']==[]and receipt['tag']==wanted['tag']
         and receipt['sources_unchanged']is True and receipt['devices_after']['all_idle']is True
         and receipt['owned_remaining']==[],'complete same-tag owned execution, source continuity and actual idle')
    for phase in('pre_verifier','post_verifier'):
        value=receipt[phase]
        need(value['exit_code']==0 and value['actual_result']['status']=='pass'
             and value['actual_result']['errors']==[]and sha(value['path'])==value['sha256'],
             'exact retained original source/environment verification log')
    validation=dict(source_and_environment_verified=True,physical_cards=receipt['physical_cards'],
        all_owned_workers_exited=True,all4_idle_after=True)
    method=wanted['method'];evidence=dict(receipt_path=str(receipt_path),receipt_sha256=sha(receipt_path),
        plan_path=str(plan_path),plan_sha256=expected)
    if method=='qualified_dynamic_tree':
        need(tree_audit_path is not None,'fresh independent complete tree audit')
        audit=json.loads(Path(tree_audit_path).read_text())
        need(audit['status']=='terminal_completed_compute_audited'and audit['supervisor_status']=='complete'
             and audit['errors']==[]and audit['original_receipt_sha256']==sha(receipt_path)
             and audit['source_freeze_sha256']==plan['candidate_freeze_sha256']
             and audit['whole_workload_AR_qualified']is True,'same selected tree, independent full gates and exact terminal bytes')
        ref=audit['independent_complete_own_AR']
        need(ref['same_backend_exact_output_qualified']is True and ref['references_per_target']==463
             and {r['rank']for r in ref['by_target_rank']}=={1,2,3}
             and all(r['comparison']['requests']==r['comparison']['equal_requests']==463
                     and r['comparison']['mismatching_requests']==0
                     and r['comparison']['exact_all_tokens_equal']is True for r in ref['by_target_rank']),
             'all three actual independent target histories match all463 requests')
        metrics=collect(method,wanted['outputs']['raw_trace'])
        validation.update(all_original_and_new_gates_passed=True,whole_workload_AR_qualified=True,
            AR_equal_requests=463,AR_different_requests=0,candidate_freeze_sha256=plan['candidate_freeze_sha256'])
        evidence.update(tree_audit_path=str(tree_audit_path),tree_audit_sha256=sha(tree_audit_path))
    else:
        need(receipt['block']==block and receipt['position']==position and receipt['method']==method
             and receipt['plan_sha256']==expected and receipt['actual_result_complete']is True,
             'same original baseline guard and exact plan cell')
        paths=wanted['outputs']
        if method=='fixed_gamma4':metrics=collect(method,paths['result'],witness_directory=paths['witness_directory'])
        else:
            baseline=receipt['original_result_audit']
            need(baseline['ordinary_MC2_actually_enabled']is True
                 and baseline['MC2_projection_modules_by_rank']==[128]*4,'actual four-rank MC2 baseline proof')
            primary=receipt['metrics']['source_descriptors'][0]['path']
            metrics=collect(method,primary)
            validation.update(ordinary_MC2_actually_enabled=True,MC2_projection_modules_by_rank=[128]*4)
        need(metrics==receipt['metrics'],'independent fresh arithmetic equals retained original baseline normalization')
    output=Path(output);need(not output.exists(),'fresh result cell; preserve all prior outcomes')
    cell=dict(block=block,position=position,method=method,tag=wanted['tag'],metrics=metrics,
              validation=validation,evidence=evidence)
    with output.open('x')as stream:json.dump(cell,stream,indent=2);stream.write('\n')
    return cell


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--expected-plan-sha256',required=True);p.add_argument('--block',type=int,required=True)
    p.add_argument('--position',type=int,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--tree-audit',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cell=record(a.plan,a.expected_plan_sha256,a.block,a.position,a.receipt,a.output,a.tree_audit)
    print(json.dumps(dict(method=cell['method'],block=cell['block'],position=cell['position'],metrics=cell['metrics']['all'])))
