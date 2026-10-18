"""One current cap4/causal80 parent173, then genuine multiowner prefill/next4."""
from base190 import parent_module,require
def run(engine,native,stage,policy,output):
 proof=parent_module('component173').run(engine,native,stage,policy,output)
 require(proof is engine._v173_qualified_component,'same current original173 proof')
 from prefill190 import run as prefill
 extra=prefill(engine,native,stage,output)
 result=dict(proof,multiowner_prefill=extra)
 engine._v190_qualified_component=result
 return result
