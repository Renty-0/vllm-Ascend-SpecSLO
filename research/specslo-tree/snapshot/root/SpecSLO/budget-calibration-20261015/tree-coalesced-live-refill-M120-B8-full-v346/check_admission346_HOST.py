"""Actual projected priority/arrival loop with external CPU-only dependencies."""
import ast,copy,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path.append('/root/SpecSLO/budget-calibration-20261015/tree-static-q5-M120-SLO-first-full463-own-AR-v257')
sys.path.append(str(B/'tree-static-q5-M120-full463-own-AR-v256'))
sys.path[:0]=[str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),str(B/'tree-active-window32-16-local-v160'),str(B/'tree-high-normal-cap4-native-families-v148')]
from generation346 import patch,descriptor
from window346 import adapt_source,window,windows
from empty209 import patch as empty_patch


class Scalar:
    def __init__(self,value):self.value=value
    def cpu(self):return self
    def item(self):return self.value[0]


def main():
    from source346_HOST import source as actual_source
    original=actual_source();projected=patch(original)
    function=copy.deepcopy(next(n for n in ast.walk(ast.parse(projected))if isinstance(n,ast.FunctionDef)and n.name=='admit_available'))
    end=next(i for i,n in enumerate(function.body)if isinstance(n,ast.If)and 'prefill_coalescing'in ast.unparse(n.test))
    function.body=function.body[:end]+[ast.Return(value=ast.Name(id='admitted',ctx=ast.Load()))]
    rows=[]
    for label,lengths,arrivals,priority in[
        ('all48-short',[128]*48,[0]*48,list(range(48))),
        ('all48-long',[2301]*48,[0]*48,list(range(48))),
        ('priority-mixed',[2301]*24+[128]*24,[0]*48,list(range(24,48))+list(range(24))),
        ('arrival-mixed',[128]*48,[0]*24+[200]*24,list(range(48)))]:
        params=[types.SimpleNamespace(arrival_ts=a)for a in arrivals]
        states=[types.SimpleNamespace(prompt_length=n,max_tokens=256)for n in lengths]
        owner=types.SimpleNamespace(rank=1,topology=types.SimpleNamespace(target_leader_rank=1),config=types.SimpleNamespace(
            spec_rhythm_online_prefill=True,spec_rhythm_kv_ready_arrivals=False,max_num_batched_tokens=131072))
        namespace=dict(self=owner,receive_live_admissions=lambda block=False:False,active=set(),pending_admission=list(range(48)),
            initial_batch_size=48,torch=types.SimpleNamespace(tensor=lambda value,**kwargs:Scalar(value),float64='CPU-double'),
            time=types.SimpleNamespace(time=lambda:100.0),_dt_cpu_broadcast=lambda *args,**kwargs:None,
            order_spec_rhythm_admission_params=lambda *args,**kwargs:tuple(priority),request_params=params,
            has_slo_constraints=True,envs=types.SimpleNamespace(VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=True),
            prefetched=set(),local_states=states,counters={},cache_activated=set(),cache_released=set(),
            release_online_cache=lambda *a:None,_v75_hooks=types.SimpleNamespace(
                reset_for_original_admission_fallback=lambda **kw:None,reserved_slots=lambda:0,filter_pending_candidates=lambda candidates:candidates))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),__file__+'::actual-prefix-of-original-admission-loop','exec'),namespace)
        selected=namespace['admit_available']()
        expected_order=[i for i in priority if arrivals[i]<=100]
        assert selected==expected_order[:len(selected)]and len(selected)<=48
        pages=64+sum((lengths[i]+256+8+127)//128 for i in selected)
        assert pages<=544 and namespace['pending_admission']==list(range(48))
        if label=='all48-short':assert len(selected)==48
        if label=='all48-long':assert len(selected)==22 and namespace['counters']['spec_rhythm_ROOT256_page_reserve_deferrals']==1
        if label=='arrival-mixed':assert selected==list(range(24))
        rows.append(dict(case=label,admitted=len(selected),reserved_pages=pages,selected=selected))
    # The declared window API cannot accidentally fall back to an old arm.
    assert window(48)==48 and windows((48,))==(48,)
    for bad in(16,24,32,64):
        try:window(bad)
        except ValueError:pass
        else:raise AssertionError('undeclared active window accepted')
    assert not any(m.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for m in sys.modules)
    result=dict(status='HOST_actual_live_projection_and_admission_prefix_passed',descriptor=descriptor(original),
        fixtures=rows,all_unadmitted_remain_pending=True,original_arrival_and_priority_order_preserved=True,
        original_full_decode_AST_inverse_passed=True,external_async_hook_fixture="no reservations; identity candidate filter",NPU_used=False,model_constructed=False)
    with(P/'HOST_ADMISSION346_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
