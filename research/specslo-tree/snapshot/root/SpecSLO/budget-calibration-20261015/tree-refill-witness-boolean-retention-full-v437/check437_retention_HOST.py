"""Real CPU admission -> actual431 witness functions -> saved-trace expression.

Only permission checks are replaced in isolated test function namespaces.
No permission is issued, no production module is monkey patched, and all
model I/O and clocks are the explicit inherited CPU fixture doubles.
"""
import ast
import copy
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace


def main():
    import torch,torch_npu
    attempts=[]
    def blocked(*args,**kwargs):
        attempts.append(True)
        raise AssertionError('HOST437 no model or NPU use')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    from vllm_ascend.spec_decode.pearl import native_engine as native,spec_rhythm as rhythm
    native.NativePearlEngine.__init__=blocked
    import witness431,witness437,projection431,trace432,trace437
    fixture=projection431.B/'tree-nonempty-wholeprompt-refill-resource-host-v426/check426_HOST.py'
    main=next(n for n in ast.parse(fixture.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    nodes=[copy.deepcopy(n)for n in main.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in('Clock','build')]
    ns=dict(torch=torch,native=native,rhythm=rhythm,SimpleNamespace=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(fixture)+'::explicit-CPU-fixtures','exec'),ns)
    build=ns['build']
    def clone(fn,globals):
        return FunctionType(fn.__code__,globals,fn.__name__,fn.__defaults__,fn.__closure__)
    # Execute the actual431 before/finish implementations, retaining their code
    # identities and data checks. The isolated permission double cannot escape.
    host_ns=dict(vars(witness431),current=lambda engine:None,_FINISH=None)
    before=clone(witness431.before,host_ns)
    finish=clone(witness431.finish,host_ns)
    assert before.__code__ is witness431.before.__code__ and finish.__code__ is witness431.finish.__code__
    original_path=projection431.B/'tree-parent4-canonical-tail-observer-binding-v425/HOST425.generated-LIVE.py'
    projected=projection431.patch(original_path.read_text())
    actual=next(n for n in ast.walk(ast.parse(projected))if isinstance(n,ast.FunctionDef)and n.name=='admit_available')
    assert ast.unparse(actual.body[0]).startswith('from witness431 import')
    actual.body.pop(0)  # HOST-only dependency injection; production retains it.
    successful=old_rejections=0
    negatives=[]
    for old_count,new_count in ((0,8),(8,8),(24,16),(32,32)):
        namespace,view=build(old_count,[63]*new_count,set())
        engine=namespace['self']
        engine._dynamic_tree_generation=7
        engine._root431_refill_rows={}
        namespace.update(controller=view['controller'],_before429=before,_charged429=witness431.charged,_finish429=finish)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(actual)],type_ignores=[])),'HOST437.actual431-admission-CPU-IO-only','exec'),namespace)
        namespace['admit_available']()
        rows=engine._root431_refill_rows[7]
        assert len(rows)==1 and rows[0]['incumbents']==list(range(old_count))
        assert rows[0]['admitted']==[owner for owner,_ in view['publications']]
        assert rows[0]['incumbent_decode_after_ms']==[v+rows[0]['prefill_ms']for v in rows[0]['incumbent_decode_before_ms']]
        old_snapshot=clone(witness431.snapshot,dict(vars(witness431),current=lambda engine:None))
        new_snapshot=clone(witness437.snapshot,dict(vars(witness437),current=lambda engine:None))
        try:old_snapshot(engine,7)
        except RuntimeError as error:
            assert 'actual admissions retained' in str(error);old_rejections+=1
        else:raise AssertionError('strict bool failure not reproduced')
        # Evaluate exactly the expression added to the actual saved trace dict.
        keywords=[n for n in ast.walk(trace437.trace_source(64))if isinstance(n,ast.keyword)and n.arg=='ROOT432_refill_witnesses']
        assert len(keywords)==1
        expression=compile(ast.Expression(body=copy.deepcopy(keywords[0].value)),'HOST437.exact-trace-retention-expression','eval')
        def host_import(name):
            assert name=='witness437'
            return SimpleNamespace(snapshot=new_snapshot)
        saved=eval(expression,dict(engine=engine,generation=7,__import__=host_import))
        assert saved==rows and saved is not rows
        successful+=1
        for kind,value,generation in (('empty',[],7),('absent',None,7),('wrong_type',tuple(rows),7),('wrong_generation',rows,8)):
            engine._root431_refill_rows={7:value}
            try:new_snapshot(engine,generation)
            except RuntimeError:negatives.append(kind)
            else:raise AssertionError('missing witness accepted '+kind)
        try:witness437.snapshot(engine,7)
        except RuntimeError as error:assert 'current NN permission' in str(error)
        else:raise AssertionError('HOST fixture obtained production permission')
    old_tree=trace432.trace_source(64);new_tree=trace437.trace_source(64)
    selected=[n for n in ast.walk(new_tree)if isinstance(n,ast.keyword)and n.arg=='ROOT432_refill_witnesses']
    selected[0].value=ast.parse("__import__('witness431').snapshot(engine,generation)",mode='eval').body
    assert ast.dump(old_tree,include_attributes=False)==ast.dump(new_tree,include_attributes=False)
    old=ast.parse(__import__('inspect').getsource(witness431.snapshot));new=ast.parse(__import__('inspect').getsource(witness437.snapshot))
    checks=[n for n in ast.walk(new)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='need']
    assert len(checks)==1 and ast.unparse(checks[0].args[0])=='type(rows) is list and bool(rows)'
    checks[0].args[0]=ast.parse('type(rows)is list and rows',mode='eval').body
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    assert not attempts
    result=dict(status='PASS_HOST437_ACTUAL_CPU_ADMISSION_TO_TRACE_RETENTION_AND_STRICT_BOOLEAN_REGRESSION',
        actual_CPU_admission_to_saved_trace_cases=successful,old_bug_reproduced=old_rejections,
        controlled_missing_wrong_type_or_generation_negatives=len(negatives),
        actual431_before_finish_code_identity_retained=True,
        only_production_semantic_delta='bool(rows) instead of rows as strict-True predicate',
        production_permission_checks_retained_and_HOST_fixture_rejected=True,
        complete_original_trace_AST_inverse=True,complete_original_snapshot_AST_inverse=True,
        isolated_HOST_permission_double_only=True,production_modules_monkeypatched=False,
        NPU_used=False,Native_constructor_calls=0,NN_permission_issued=False,Raw=None,Goodput=None)
    (Path(__file__).resolve().parent/'HOST437_RETENTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
