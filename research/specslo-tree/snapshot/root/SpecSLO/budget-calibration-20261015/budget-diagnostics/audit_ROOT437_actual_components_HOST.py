"""Read completed current75+6 evidence; no model, device or permission use."""
import datetime,hashlib,importlib.abc,json,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-refill-witness-boolean-retention-full-v437'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='boolean-witness-retention-parent4-E4-global64-root-v437-fresh75-plus6-full463-ownAR-v1'

class Firewall(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST437 no model imports '+fullname)


def main():
    sys.meta_path.insert(0,Firewall())
    line=next(l for l in(P/'startup437.sh').read_text().splitlines()if l.startswith('export PYTHONPATH='))
    value=line.removeprefix('export PYTHONPATH=');F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
    for old,new in(('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(DONOR)),('${PYTHONPATH:-}','')):value=value.replace(old,new)
    assert '$'not in value
    sys.path[:0]=value.split(':')
    assert hashlib.sha256((P/'FREEZE437.json').read_bytes()).hexdigest()=='8ecd9f97279d1ae6023e4ca14188954b34575b26940ec64d1848c8b93a99f8d8'
    actual=DONOR/'actual'/TAG
    assert not list(actual.glob('rank*-last-error.json'))
    for rank in range(4):
        assert(actual/'multiincumbent32-prefill427-own-current-process'/f'rank{rank}-component.json.gz').exists()
    from audit419 import audit
    from incumbents427 import audit as incumbent_audit
    bounded=audit(actual/'parent4-current-component425')
    incumbent=incumbent_audit(actual/'multiincumbent32-prefill427-own-current-process')
    assert bounded['passed']is bounded['full_required_natural_coverage']is True
    assert incumbent['passed']is True
    result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='PASS_HOST437_RETAINED_ACTUAL_75_PLUS6_COMPONENT_AUDITS',
        actual_current75_each_rank=bounded,actual_current6_incumbent_cases=incumbent,
        full463_performance_not_qualified_by_this_audit=True,
        whole_workload_AR_qualified=False,new_engine_permission_inherited=False,
        Raw=None,Goodput=None,NPU_used=False,NN_math_reexecuted_by_HOST=False)
    with(D/'ROOT437_COMPLETED_75_PLUS6_INDEPENDENT_COMPONENT_AUDIT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],all75_per_rank=True,all6_per_rank=True,NPU_used=False)))


if __name__=='__main__':main()
