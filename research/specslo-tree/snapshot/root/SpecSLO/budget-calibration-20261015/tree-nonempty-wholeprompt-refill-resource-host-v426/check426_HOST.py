"""Actual admission function/CPU request states with explicit model/clock doubles."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
from types import SimpleNamespace


def main():
    import torch
    import torch_npu
    def forbidden(*a,**k):raise AssertionError('HOST426 cannot initialize Native models or an NPU')
    torch.npu._lazy_init=forbidden;torch_npu.npu._lazy_init=forbidden
    from vllm_ascend.spec_decode.pearl import native_engine as native, spec_rhythm as rhythm
    native.NativePearlEngine.__init__=forbidden
    import refill426 as candidate
    rng=random.Random(426)
    class Clock:
        def __init__(self):self.steps=0
        def time(self):return 1000.
        def perf_counter(self):
            value=1.+.03*self.steps;self.steps+=1;return value

    def build(incumbents,lengths,future):
        total=incumbents+len(lengths)
        prompt_lengths=[63]*incumbents+list(lengths)
        draft=[native.PearlPipelineState(list(range(n))+([77]if i<incumbents else[]),n,
                 max_tokens=256,ignore_eos=True,request_id=str(i))for i,n in enumerate(prompt_lengths)]
        target=[s.clone()for s in draft]
        params=[native.NativeSamplingParams(temperature=0.,max_tokens=256,ignore_eos=True,
                 slo_class=('tight','normal','loose')[i%3],slo_tpot_ms=(40.,50.,150.)[i%3],
                 arrival_ts=1001. if i in future else 999.,request_id=str(i))for i in range(total)]
        states={i:rhythm.SpecRhythmRuntimeState(i,i%2,max_gamma=5,slo_tpot_ms=params[i].slo_tpot_ms,
                    slo_class=params[i].slo_class,output_token_budget=256,
                    delivered_tokens=1 if i<incumbents else 0,decode_elapsed_ms=10.)for i in range(total)}
        controller=rhythm.SpecRhythmPipelineController(states)
        for i in range(incumbents):controller.publish([controller.new_ticket(i,gamma=4,eager=False)])
        active=list(range(incumbents));pending=list(range(incumbents,total))
        cache_activated=set(active);activation=[];publications=[];prefills=[];polls=[];broadcasts=[]
        counters={k:0 for k in ('spec_rhythm_prefill_batches','spec_rhythm_prefill_requests',
            'spec_rhythm_prefill_singleton_batches','spec_rhythm_prefill_pair_batches',
            'spec_rhythm_prefill_multi_batches','spec_rhythm_peak_active_requests')}
        counters['spec_rhythm_peak_active_requests']=incumbents
        def prefill(prompts,local,owners):
            assert [target[i]for i in owners]==local
            assert prompts==[target[i].token_ids for i in owners]
            prefills.append(list(owners));return [100+i for i in owners]
        def deliver(i,p,s):publications.append((i,list(s.committed_completion_token_ids)))
        def activate(ids):activation.extend(ids);cache_activated.update(ids)
        def receive(*,block):polls.append(block);return False
        def broadcast(e,tensor,*,src,kind):
            assert src==1 and tensor.device.type=='cpu';broadcasts.append(kind)
        hooks=SimpleNamespace(reset_for_original_admission_fallback=lambda **kw:None,
            reserved_slots=lambda:0,filter_pending_candidates=tuple,
            reserve_at_original_admission=lambda **kw:False)
        engine=SimpleNamespace(rank=1,topology=SimpleNamespace(target_leader_rank=1),
            config=SimpleNamespace(spec_rhythm_online_prefill=True,spec_rhythm_kv_ready_arrivals=False,
                max_num_batched_tokens=32768,prefill_chunk_size=64,max_num_seqs=64),
            _prefill_and_sample_target_batch=prefill,_deliver_committed_tokens=deliver,eos_token_ids=(2,))
        namespace=dict(self=engine,receive_live_admissions=receive,active=active,
            pending_admission=pending,initial_batch_size=64,_v75_hooks=hooks,cache_activated=cache_activated,
            cache_released=set(),release_online_cache=lambda ids:None,torch=torch,time=Clock(),
            _dt_cpu_broadcast=broadcast,request_params=params,local_states=target,
            has_slo_constraints=True,envs=SimpleNamespace(VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=True),
            order_spec_rhythm_admission_params=native.order_spec_rhythm_admission_params,
            prefetched=set(),prefill_coalescing=False,states=states,draft_states=draft,target_states=target,
            activate_online_cache=activate,counters=counters,completed={},finished_decode_elapsed={},
            _finished=native._finished)
        return namespace,dict(controller=controller,active=active,pending=pending,states=states,
            draft=draft,target=target,params=params,activation=activation,publications=publications,
            prefills=prefills,polls=polls,broadcasts=broadcasts,counters=counters,prompt_lengths=prompt_lengths)

    positive=deferred=empty_exact=0
    for trial in range(240):
        incumbents=0 if trial%6==0 else rng.randrange(1,65)
        lengths=[rng.choice((32,127,255,511,1023,1800))for j in range(rng.randrange(0,65))]
        future={incumbents+j for j in range(len(lengths))if rng.random()<.2}
        low=rng.choice((16,24,32,48));minimum=rng.choice((2,8,16,32))
        ns,view=build(incumbents,lengths,future)
        before=[s.token_ids[:]for s in view['target'][:incumbents]]
        tickets=dict(view['controller'].ready)
        epochs={i:s.prefix_epoch for i,s in view['states'].items()}
        _,function=candidate.construct(ns,low_water=low,minimum_refill=minimum)
        assert function() is False
        admitted=[i for i,_ in view['publications']]
        assert view['polls']==[False]
        assert view['active']==list(range(incumbents))+admitted
        assert len(view['active'])<=64 and len(admitted)<= (32 if incumbents else 64)
        assert len(admitted)==len(set(admitted)) and not(set(admitted)&future)
        assert view['activation']==admitted
        assert [s.token_ids for s in view['target'][:incumbents]]==before
        assert view['controller'].ready==tickets
        assert {i:s.prefix_epoch for i,s in view['states'].items()}==epochs
        assert view['pending']==[i for i in range(incumbents,incumbents+len(lengths))if i not in admitted]
        used=64+sum((view['prompt_lengths'][i]+256+5+127)//128 for i in view['active'])
        assert used<=544
        if admitted:
            positive+=int(incumbents>0)
            assert view['prefills']==[admitted]
            assert view['broadcasts']==['arrival_clock','prefill_clock']
            for i in range(incumbents):assert abs(view['states'][i].decode_elapsed_ms-40.)<1e-8
            for i in admitted:
                assert view['draft'][i].committed_completion_token_ids==view['target'][i].committed_completion_token_ids==[100+i]
                assert view['states'][i].delivered_tokens==1
        else:
            deferred+=int(incumbents>0)
            assert view['prefills']==[] and view['activation']==[]
            assert all(view['states'][i].decode_elapsed_ms==10. for i in range(incumbents))
        if not incumbents:
            old_ns,old=build(incumbents,lengths,future)
            function,_=candidate.construct(old_ns,low_water=low,minimum_refill=minimum)
            assert function() is False
            for key in ('active','pending','activation','publications','prefills','polls','broadcasts','counters'):
                assert old[key]==view[key],key
            assert [asdict(s)for s in old['target']]==[asdict(s)for s in view['target']]
            assert [asdict(s)for s in old['draft']]==[asdict(s)for s in view['draft']]
            assert [asdict(s)for s in old['states'].values()]==[asdict(s)for s in view['states'].values()]
            empty_exact+=1
    assert positive and deferred and empty_exact==40
    value=dict(status='PASS_HOST426_ACTUAL_ADMISSION_FUNCTION_RESOURCE_AND_PUBLICATION_CONTROL',
        HOST_cases=240,positive_nonempty_refills=positive,nonempty_deferrals=deferred,
        unchanged_empty_refill_cases=empty_exact,source=candidate.descriptor(),
        actual_original_CPU_Pearl_states_SpecRhythm_controller_and_order=True,
        explicit_clock_broadcast_prefill_cache_IO_doubles=True,
        incumbent_tokens_and_ticket_epochs_preserved_on_HOST=True,
        exact_original_prefill_time_charge_and_once_root_publication=True,
        global_limit64_and_total_page_pool544=True,Native_models_constructed=0,
        NPU_used=False,NN_incumbent_KV_qualification=False,serving_installed=False,
        Raw=None,Goodput=None)
    p=Path(__file__).resolve().parent
    (p/'HOST426_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
