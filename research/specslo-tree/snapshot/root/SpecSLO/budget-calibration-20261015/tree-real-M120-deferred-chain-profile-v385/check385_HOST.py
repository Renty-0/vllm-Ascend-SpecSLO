"""Verify group forwarding and unchanged original measurement source before launch."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(P))


def main():
    from original385 import TargetGroup,EXAMPLE
    group=object();calls=[]
    class Dist:
        ReduceOp='original-reduction-enum'
        def is_initialized(self):return True
        def get_world_size(self,group=None):assert group is target;return 3
        def get_rank(self,group=None):assert group is target;return 1
        def all_reduce(self,*args,**kwargs):calls.append(('reduce',args,kwargs))
        def all_gather_object(self,*args,**kwargs):calls.append(('gather',args,kwargs))
        def barrier(self,*args,**kwargs):calls.append(('barrier',args,kwargs))
        def destroy_process_group(self,*args,**kwargs):raise AssertionError('borrower must not destroy outer group')
    target=group;proxy=TargetGroup(Dist(),target)
    proxy.init_process_group('hccl');proxy.all_reduce('actual');proxy.all_gather_object('a','b');proxy.barrier()
    proxy.destroy_process_group()
    assert proxy.get_rank()==1 and proxy.get_world_size()==3 and proxy.ReduceOp==Dist.ReduceOp
    assert all(v[2]['group']is target for v in calls)
    assert proxy.borrowed_init_checks==proxy.borrowed_destroy_calls==1
    tree=ast.parse(EXAMPLE.read_text());names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert {'main','_measure','_compare','_capture','_load_payload'}<=names
    import guard385,runtime385
    ns=guard385.execution_namespace();source=runtime385.worker_source();compile(source,'HOST385.actual-worker','exec')
    assert ns['ACTION']=='real-M120-deferred-chain-original-graph-qualifier'
    assert ns['SCOPE']['finite_cases']==1 and ns['SCOPE']['independent_prefixes']==72
    assert source.count('from probe385 import run')==1 and source.count('numeric_cases=1')==2
    try:guard385.actual_audit('/nonexistent-HOST385')
    except FileNotFoundError:pass
    else:raise AssertionError('expected nonexistent data after actual audit import')
    from oracle255 import cases
    assert len([c for c in cases()if c['name']=='N24-all4-anchor0'])==1
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    value=dict(status='PASS_HOST385_TP_GROUP_FORWARDING_AND_ACTUAL_LIFECYCLE',NPU_used=False,
        original_example=str(EXAMPLE),original_example_sha256=hashlib.sha256(EXAMPLE.read_bytes()).hexdigest(),
        borrowed_group_not_destroyed=True,original_measurement_functions_used=True,
        model_qualified=False,new_profile_installed=False,Raw=None,Goodput=None)
    (P/'HOST385_CONSTRUCTION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
