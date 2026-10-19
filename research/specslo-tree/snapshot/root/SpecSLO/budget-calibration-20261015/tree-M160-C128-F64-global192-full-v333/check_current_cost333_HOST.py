"""Synthetic current-cost data tests source/model/ABI/rotation/state bindings."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

P=Path(__file__).resolve().parent
import runtime333
sys.path.append(str(P.parent/"tree-high-normal-cap4-native-families-v148"))
from cost333 import configurations,checked_costs,bind_current,current,ratio,snapshot
from budget_policy import TreeBudgetPolicy


def panels(rank):
    rows=[]
    for n,forks in configurations():
        names=tuple(range(4,9))if forks else(4,)
        base=50.+n*.01+rank*.001
        pairs=[]
        for iteration in range(20):
            order=names[iteration%len(names):]+names[:iteration%len(names)]
            elapsed={str(b):base*(1. if b==4 else 1.1+.003*(forks-1)+.001*(b-5))for b in names}
            pairs.append(dict(order=order,milliseconds=elapsed))
        rows.append(dict(N=n,branch_owners=forks,rounds=20,pairs=pairs,original_greedy_output_stable=True))
    counters=dict.fromkeys(('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
                  'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks'),0)
    return dict(rank=rank,target_math_on_rank=True,no_new_capture_or_validation_in_costs=True,
                graph_counters_before=dict(counters),graph_counters_after=dict(counters),rows=rows)


def main():
    source='a'*64;profile='b'*64
    reports=[panels(rank)for rank in(1,2,3)]
    setup,slope,rows=checked_costs(reports)
    assert len(rows)==48 and setup>=.103-1e-12 and abs(slope-.003)<1e-12
    model=json.loads((P.parent/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    policy=TreeBudgetPolicy(model)
    rejected=[]
    for label,change in[
        ('rotation',lambda r:r[0]['rows'][0]['pairs'][0]['order'].reverse()),
        ('time',lambda r:r[0]['rows'][0]['pairs'][0]['milliseconds'].__setitem__('4',float('nan'))),
        ('graph_counter',lambda r:r[0]['graph_counters_after'].__setitem__('aclgraph_captures',1)),
        ('missing_panel',lambda r:r[0]['rows'].pop()),
    ]:
        bad=copy.deepcopy(reports)
        # JSON-like source uses lists; the generator's order tuple is immutable.
        if label=='rotation':bad[0]['rows'][0]['pairs'][0]['order']=list(reversed(bad[0]['rows'][0]['pairs'][0]['order']))
        else:change(bad)
        try:checked_costs(bad)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Invalid cost accepted '+label)
    with tempfile.TemporaryDirectory(prefix='HOST333-current-cost-',dir=P)as folder:
        root=Path(folder);directory=root/'paired-current-target-cost333';directory.mkdir()
        engine=NS(rank=1,model=object(),_v101_output_directory=str(root),mc2_static_route_manifest=NS(profile_sha256=profile))
        records=[]
        for rank,report in zip((1,2,3),reports,strict=True):
            payload=dict(rank=rank,source_freeze_sha256=source,run_directory=str(root),
                model_identity=id(engine.model)if rank==1 else 100+rank,model_dtype='torch.bfloat16',
                fresh_math_cases=532,all_math_bitwise=True,physical_FIA_rows=160,physical_FIA_groups=32,
                actual_projection_rows=165,chained_flush_334=True,deferred_read_done=False,
                profile_sha256=profile,costs=report)
            path=directory/f'rank{rank}-target-cost.json';path.write_text(json.dumps(payload));records.append((path,payload))
        try:ratio(policy,1)
        except RuntimeError:rejected.append('unbound')
        else:raise AssertionError('Missing calibration accepted')
        for label,key,value in [('source','source_freeze_sha256','c'*64),('model','model_identity',0),('ABI','deferred_read_done',True)]:
            path,payload=records[0];bad=dict(payload);bad[key]=value;path.write_text(json.dumps(bad))
            try:bind_current(TreeBudgetPolicy(model),engine,directory,source)
            except ValueError:rejected.append(label)
            else:raise AssertionError('Foreign calibration accepted '+label)
            path.write_text(json.dumps(payload))
        proof=bind_current(policy,engine,directory,source)
        assert proof['same_instance_before_performance']is True
        assert ratio(policy,0)==1. and all(ratio(policy,n+1)>=ratio(policy,n)for n in range(12))
        binding=current(policy)
        from audit_cost333 import audit as independent_audit
        proofs=[dict(paired_target_cost_calibration=proof,actual_cold_math_graph_calls_only=True,untimed_cost_graph_call_count=0 if r==0 else 1596,all_prefixes_exact_after_cost=None if r==0 else True)for r in range(4)]
        independent=independent_audit(root,proofs,expected_source_sha256=source)
        assert independent['all48_target_cost_rows_recomputed']is True
        from risk333 import begin_generation
        begin_generation(policy,1);begin_generation(policy,2)
        assert current(policy)is binding and ratio(policy,1)>=1.103-1e-12
        try:bind_current(policy,engine,directory,source)
        except RuntimeError:rejected.append('rebind')
        else:raise AssertionError('Calibration rebound')
        foreign=TreeBudgetPolicy(model);foreign._ROOT333_cost_envelope=binding
        try:ratio(foreign,1)
        except RuntimeError:rejected.append('foreign_policy')
        else:raise AssertionError('Foreign policy accepted bound curve')
    assert len(rejected)==10 and 'torch_npu'not in sys.modules
    out=dict(status='HOST_current_cost_binding_and_generation_persistence_passed',
        target_panels=48,rotations_per_panel=20,invalid_cases_rejected=rejected,
        actual_original_risk_generation_reset_preserves_static_calibration=True,
        synthetic_HOST_cost_only=True,no_actual_model_or_NPU_run=True,
        independent_full_auditor_recomputes_same48_rows=True,no_latency_or_Goodput_claim=True,Goodput=None)
    with(P/'HOST_CURRENT_COST333_RESULT.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
    print(json.dumps(out))

if __name__=='__main__':main()
