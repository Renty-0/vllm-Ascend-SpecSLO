"""CPU-only original API-receiver and unchanged configuration controls."""
from pathlib import Path
from dataclasses import dataclass
from types import ModuleType
import json,sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import check462,guard462,runtime462,wait462
@dataclass
class Config:
    worker_timeout_seconds:int=6000
class PEARLEngine:
    def __init__(self):self.config=Config();self._connections=[object()];self._processes=[object()]
    def _receive_all(self,operation):
        assert operation=='V209 causal80-N9-to15-allB4-empty-active32-full463'
        assert self._connections is engine._connections and self._processes is engine._processes
        return self.config.worker_timeout_seconds
name='vllm_ascend.spec_decode.pearl.api';previous=sys.modules.get(name);fake=ModuleType(name);fake.PEARLEngine=PEARLEngine;sys.modules[name]=fake
try:
    engine=PEARLEngine();cfg=engine.config
    assert wait462.receive_qualification(engine,'V209 causal80-N9-to15-allB4-empty-active32-full463')==21600
    assert engine.config is cfg and cfg.worker_timeout_seconds==6000
    snapshot=wait462.snapshot();assert snapshot['outer_supervisor_total_limit_seconds']==21600
    assert snapshot['original_receive_function_and_owned_connections_preserved']is True
    assert snapshot['worker_configuration_changed']is False and snapshot['production_timeout_changed']is False
    try:wait462.receive_qualification(engine,'unrelated-production-operation')
    except RuntimeError:pass
    else:raise AssertionError('ROOT462 unrelated receive operation accepted')
finally:
    if previous is None:sys.modules.pop(name)
    else:sys.modules[name]=previous
ns=guard462.execution_namespace()
assert ns['execute_locked'].__globals__ is ns and ns['MAX_SECONDS']==21600
assert ns['MAX_TAG_BYTES']==256*1024**2 and ns['MAX_AGGREGATE_BYTES']==3584*1024**2
source=runtime462.source();assert 'from wait462 import receive_qualification'in source
assert source[source.index('def worker('):source.index('def main(')]==runtime462.runtime460.source()[runtime462.runtime460.source().index('def worker('):runtime462.runtime460.source().index('def main(')]
for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
value=dict(status='PASS_HOST462_ORIGINAL_RECEIVER_ONLY_WAIT_VIEW_AND_NUMERICAL_WORKER_INVERSE',
    CPU_fixture_only=True,original_config_object_retained=True,original_worker_timeout_seconds=6000,
    qualification_only_wait_seconds=21600,unrelated_operation_rejected=True,
    actual_numerical_worker_source_identical=True,full_supervisor_limits_checked=True,
    all_NN_gates_still_required=True,NPU_used=False,Native_constructor_calls=0,Raw=None,Goodput=None)
with(P/'HOST462_WAIT_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
