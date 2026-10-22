"""Actual immutable API receiver with explicit HOST pipes/processes and clocks."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    import torch,torch_npu
    attempted=[]
    def blocked(*args,**kwargs):
        attempted.append(True)
        raise AssertionError('HOST436 cannot construct Native models or initialize an NPU')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    from vllm_ascend.spec_decode.pearl import api
    api.PEARLEngine.__init__=blocked
    from config209 import make_config
    import wait436
    operation='V209 causal80-N9-to15-allB4-empty-active32-full463'
    class Connection:
        def __init__(self,reply):self.reply=reply
        def recv(self):return self.reply
        def poll(self):return True
    def make(reply=('calibrated',0,dict(HOST=True))):
        e=object.__new__(api.PEARLEngine)
        e.config=make_config();e._connections=[Connection(reply)]
        e._processes=[SimpleNamespace(is_alive=lambda:True)]
        return e
    class Clock:
        def __init__(self,elapsed):self.values=iter((0.,float(elapsed)))
        def monotonic(self):return next(self.values)
    def wait(pending,timeout):
        assert 0<timeout<=1.;return list(pending)
    e=make();original=e.config
    assert original.worker_timeout_seconds==6000
    with patch.object(api,'time',Clock(6500)),patch.object(api,'wait_for_connections',wait):
        try:e._receive_all(operation)
        except TimeoutError as error:
            assert '6000s'in str(error);old_error=str(error)
        else:raise AssertionError('the real original6000s receiver must reject late completion')
    with patch.object(api,'time',Clock(6500)),patch.object(api,'wait_for_connections',wait):
        replies=wait436.receive_qualification(e,operation)
    assert replies==[('calibrated',0,dict(HOST=True))]
    assert e.config is original and e.config.worker_timeout_seconds==6000
    record=wait436.snapshot()
    assert record['qualification_receive_timeout_seconds']==18000
    rejected=[]
    with patch.object(api,'time',Clock(18001)),patch.object(api,'wait_for_connections',wait):
        try:wait436.receive_qualification(e,operation)
        except TimeoutError as error:
            assert '18000s'in str(error);rejected.append('bounded18000_deadline_preserved')
        else:raise AssertionError('qualification must retain a finite deadline')
    bad=make(('error','HOST-controlled worker traceback'))
    with patch.object(api,'time',Clock(5)),patch.object(api,'wait_for_connections',wait):
        try:wait436.receive_qualification(bad,operation)
        except RuntimeError as error:
            assert 'HOST-controlled worker traceback'in str(error);rejected.append('worker_error_still_propagates')
        else:raise AssertionError('worker errors cannot become qualified results')
    try:wait436.receive_qualification(e,'generate')
    except RuntimeError:rejected.append('ordinary_generation_not_adapted')
    else:raise AssertionError('only explicit original qualification operation may use the view')
    assert e.config is original and e._receive_all.__func__ is api.PEARLEngine._receive_all
    assert not attempted
    result=dict(status='PASS_HOST436_ACTUAL_API_RECEIVER_BOUNDED_QUALIFICATION_VIEW',
        real_original6000s_timeout_reproduced=old_error,
        real_original_receiver_accepts6500s_with_qualification_view=True,
        actual_receiver_function_identity_preserved=True,
        original_engine_config_and_worker_config_and_production_timeout_unchanged=True,
        controlled_negatives=rejected,explicit_HOST_clock_pipe_and_process_doubles=True,
        record=record,Native_models_constructed=0,NPU_used=False,
        NN_qualification=False,Raw=None,Goodput=None)
    (Path(__file__).resolve().parent/'HOST436_RPC_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
