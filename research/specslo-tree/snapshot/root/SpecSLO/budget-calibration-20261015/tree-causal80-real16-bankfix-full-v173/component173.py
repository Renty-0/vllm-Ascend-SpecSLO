"""Original own B4 protocol component, without unused S19/S35 mixed actors."""
from base173 import P,BASE,functions,inherited_module,require
from pathlib import Path
_MODULE=None

def original_B4_run(engine,native,stage,policy,output):
 global _MODULE
 actor_module=inherited_module('actor_v146')
 actor_ns=dict(vars(actor_module))
 functions(BASE/'actor_v146.py',actor_ns,changes=((
  'future_full_requires_21scope26phase_and_two65zero=True',
  'future_full_requires_standalone_scope_and_two65zero=True',1),))
 full=inherited_module('full_process_v146');namespace=dict(vars(full),run_actor=actor_ns['run_actor'],
  save=actor_module.save,stage4=actor_module.stage4)
 functions(BASE/'full_process_v146.py',namespace,changes=(
  ('target_entries=247','target_entries=245',1),
  ('own component starts strictly after original 86+28 or 201+44+S19/S35 gates',
   'own component starts after original114 draft or201+44 target; no mixed scope actor',1),
  ('from actor_v146 import run_actor,save,stage4','',1),))
 proof=namespace['run_full_process_component'](engine,native,engine._v75_original_protocol,directory=output,policy=policy)
 require(proof['audit']['component_numerical_qualified'] is True and proof['Native_flags_assigned'] is False,
         'actual current B4 fullphysical/FTF/protocol/next/twozero proof')
 proof=dict(proof,old_S19_S35_actor_executed=False,old_21scope26phase_claimed=False)
 engine._v173_qualified_component=proof
 return proof


def run(engine,native,stage,policy,output):
 original=original_B4_run(engine,native,stage,policy,output)
 from actor173 import run as causal80
 extra=causal80(engine,native,stage,policy,output)
 from mapper173 import issue
 stage('issue-own-causal80-mapper',lambda:issue(engine,extra))
 result=dict(original,causal80=extra,math_independent_qualified=True,
  old_padding_behavior_equal=extra['audit']['old_padding_behavior_equal'],
  numerical_variant_only=extra['audit']['numerical_variant_only'],adopted=False)
 engine._v173_qualified_component=result
 return result
