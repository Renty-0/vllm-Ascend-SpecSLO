"""Execute actual admission with the new current-SLO condition and CPU I/O."""
import ast,copy,hashlib,json,random
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace,FunctionType


def main():
    import torch,torch_npu
    attempts=[]
    def blocked(*a,**k):attempts.append(True);raise AssertionError('HOST438 no model or NPU')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    from vllm_ascend.spec_decode.pearl import native_engine as native,spec_rhythm as rhythm
    native.NativePearlEngine.__init__=blocked
    import generation438,admission438,projection431,witness431,witness437,trace438,trace437
    B=Path(__file__).resolve().parent.parent
    fixture=B/'tree-nonempty-wholeprompt-refill-resource-host-v426/check426_HOST.py'
    node=next(n for n in ast.parse(fixture.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    nodes=[copy.deepcopy(n)for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in('Clock','build')]
    ns=dict(torch=torch,native=native,rhythm=rhythm,SimpleNamespace=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(fixture)+'::explicit-CPU-IO-doubles','exec'),ns)
    parent=projection431.patch((B/'tree-parent4-canonical-tail-observer-binding-v425/HOST425.generated-LIVE.py').read_text())
    proposed=generation438.project(parent);compile(proposed,'HOST438.actual438-LIVE','exec')
    before=ast.parse(parent);after=ast.parse(proposed)
    fn=next(n for n in ast.walk(after)if isinstance(n,ast.FunctionDef)and n.name=='admit_available')
    assert ast.unparse(fn.body[0])=='from admission438 import low_water as _low_water438'
    # Check complete source restoration, including every original clock/callback.
    undo=copy.deepcopy(after);uf=next(n for n in ast.walk(undo)if isinstance(n,ast.FunctionDef)and n.name=='admit_available')
    uf.body.pop(0)
    assignment=next(n for n in uf.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='_REFILL_LOW_WATER426')
    assignment.value=ast.Constant(32)
    assert ast.dump(undo,include_attributes=False)==ast.dump(before,include_attributes=False)
    # Only HOST dependency injection removes the actual permission import.
    function=copy.deepcopy(fn);imports=[n for n in function.body if isinstance(n,ast.ImportFrom)and n.module=='witness431']
    assert len(imports)==1;function.body.remove(imports[0])
    permission_ns=dict(vars(witness431),current=lambda engine:None,_FINISH=None)
    clone=lambda f:FunctionType(f.__code__,permission_ns,f.__name__,f.__defaults__,f.__closure__)
    host_before,host_finish=clone(witness431.before),clone(witness431.finish)
    rng=random.Random(438);positive=deferred=0;max_live=0;exact_controls=[]
    controls=[(0,64,False,True),(16,32,True,True),(17,32,True,False),(32,32,False,True),(33,16,False,False)]
    trials=[(old,new,strict,expected)for old,new,strict,expected in controls]
    trials.extend((rng.randrange(0,65),rng.randrange(0,65),bool(rng.randrange(2)),None)for _ in range(180))
    for ordinal,(old,new,strict,expected)in enumerate(trials):
        namespace,view=ns['build'](old,[63]*new,set())
        engine=namespace['self'];engine._dynamic_tree_generation=7;engine._root431_refill_rows={}
        for i in range(old):view['params'][i]=replace(view['params'][i],slo_tpot_ms=40. if strict else 150.)
        namespace.update(controller=view['controller'],_before429=host_before,_charged429=witness431.charged,_finish429=host_finish)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(function)],type_ignores=[])),'HOST438.actual-SLO-admission-CPU-only','exec'),namespace)
        namespace['admit_available']();actual=bool(view['publications'])
        if expected is not None:
            assert actual is expected
            exact_controls.append(dict(old=old,new=new,strict=strict,admitted=actual,actual_live=len(view['active'])))
        if old>(16 if strict else 32):assert not actual
        if actual and old:assert 8<=len(view['publications'])<=32
        assert len(view['active'])<=64
        max_live=max(max_live,len(view['active']));positive+=actual;deferred+=not actual
        if actual:
            rows=engine._root431_refill_rows[7];assert len(rows)==1
            assert rows[0]['incumbents']==list(range(old))
            assert rows[0]['incumbent_decode_after_ms']==[v+rows[0]['prefill_ms']for v in rows[0]['incumbent_decode_before_ms']]
    assert max_live==64 and positive and deferred
    # No future request is consulted by low_water: only genuine active IDs.
    params=[SimpleNamespace(slo_tpot_ms=150.),SimpleNamespace(slo_tpot_ms=40.)]
    assert admission438.low_water(params,[0])==32 and admission438.low_water(params,[1])==16
    t=trace438.namespace();p=trace437.namespace()
    for name in('make_trace','warm_to_fixedpoint','run_windows'):assert t[name].__code__ is p[name].__code__ and t[name].__globals__ is t
    assert t['canonical_patch']is generation438.patch and t['trace_source']is p['trace_source']
    assert not attempts
    result=dict(status='PASS_HOST438_ACTUAL_CPU_SLO_AWARE_REFILL_AND_FULL_SOURCE_INVERSE',
        actual_CPU_cases=len(trials),positive=positive,deferred=deferred,exact_controls=exact_controls,
        actual_live64_observed_in_CPU_controls=True,global_limit64_not_replaced_by_hard48=True,
        original_full_LIVE_AST_inverse=True,original_clocks_and_root_publications_retained=True,
        only_current_active_SLOs_inspected=True,HOST_permission_double_isolated=True,
        original437_boolean_retention_unchanged=True,NPU_used=False,Native_constructor_calls=0,Raw=None,Goodput=None)
    (Path(__file__).resolve().parent/'HOST438_POLICY_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
