"""Independently reread completed75+6 raw cases, never model code."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-nonempty-qualified-RPC-full-v436'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='nonempty-refill-parent4-E4-global64-root-v436-current75-plus6-full463-ownAR-v1'

class Firewall(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST436 no model imports '+fullname)


def main():
    sys.meta_path.insert(0,Firewall())
    line=next(l for l in(P/'startup436.sh').read_text().splitlines()if l.startswith('export PYTHONPATH='))
    value=line.removeprefix('export PYTHONPATH=')
    F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
    for old,new in(('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(DONOR)),('${PYTHONPATH:-}','')):
        value=value.replace(old,new)
    assert '$'not in value
    sys.path[:0]=value.split(':')
    receipt=DONOR/f'receipt-{TAG}.json'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()=='3baa7f94960380c3b8fab6de032dec7452df414f0946827efca433cfef019f1c'
    terminal=json.loads((D/'ROOT436_TERMINAL_INDEPENDENT_ACTUAL_FAILURE.json').read_text())
    assert terminal['errors']==[] and terminal['metrics']is None
    from audit419 import audit
    from incumbents427 import audit as incumbent_audit
    actual=DONOR/'actual'/TAG
    bounded=audit(actual/'parent4-current-component425')
    incumbent=incumbent_audit(actual/'multiincumbent32-prefill427-own-current-process')
    assert bounded['passed']is bounded['full_required_natural_coverage']is True
    assert incumbent['passed']is True
    result=dict(status='PASS_HOST436_RETAINED_ACTUAL_75_PLUS6_COMPONENT_AUDITS',
        actual_current75_each_rank=bounded,actual_current6_incumbent_cases=incumbent,
        full463_completed=False,whole_workload_AR_qualified=False,
        new_engine_permission_inherited=False,Raw=None,Goodput=None,NPU_used=False,
        NN_math_reexecuted_by_HOST=False)
    output=D/'ROOT436_COMPLETED_75_PLUS6_INDEPENDENT_COMPONENT_AUDIT.json'
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],all75_per_rank=True,all6_per_rank=True,NPU_used=False)))


if __name__=='__main__':main()
