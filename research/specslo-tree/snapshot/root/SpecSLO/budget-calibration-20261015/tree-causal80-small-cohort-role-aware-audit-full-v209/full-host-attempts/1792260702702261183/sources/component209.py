"""One current cap4/causal80 parent173, then genuine multiowner prefill/next4."""
from base209 import parent_module,require
def run(engine,native,stage,policy,output):
 proof=parent_module('component173').run(engine,native,stage,policy,output)
 require(proof is engine._v173_qualified_component,'same current original173 proof')
 from prefill209 import run as prefill
 extra=prefill(engine,native,stage,output)
 parent=dict(proof,multiowner_prefill=extra)
 engine._v209_parent_component=parent
 from actor209 import run as usage
 own=usage(engine,native,stage,policy,output)
 from mapper209 import issue
 stage('issue-current-seven-N-usage',lambda:issue(engine,own))
 result=dict(parent,usage=own,old_padding_behavior_equal=proof['old_padding_behavior_equal']and own['audit']['crossM_behavior_equal'],numerical_variant_only=proof['numerical_variant_only']or own['audit']['numerical_variant_only'])
 engine._v209_qualified_component=result
 return result
