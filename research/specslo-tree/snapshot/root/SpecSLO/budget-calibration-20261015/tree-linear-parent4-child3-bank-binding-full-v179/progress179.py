"""Same actual inventory operations; only receipt destinations are private."""
from pathlib import Path
from base179 import P,BASE,functions,inherited_module,require
_OUTPUT=None

def configure(output):
 global _OUTPUT,qualify_prefill,qualify_target,qualify_draft,preflight_MC2,memory_checkpoint
 output=Path(output).resolve();require(output.is_relative_to(P) and output.is_dir(),'current output for constructor receipts')
 require(_OUTPUT is None or _OUTPUT==output,'one actual constructor output per process')
 _OUTPUT=output
 memory=inherited_module('inventory_memory');memory_ns=dict(vars(memory),OUTPUT=output)
 functions(BASE/'inventory_memory.py',memory_ns,names=('checkpoint',),changes=((
  "Path(__file__).resolve().parent/'startup-progress'","OUTPUT/'startup-progress'",1),))
 memory_checkpoint=memory_ns['checkpoint']
 for module_name,function_name,export in (
  ('prefill_inventory','qualify','qualify_prefill'),('target_inventory','qualify','qualify_target'),
  ('draft_inventory','qualify','qualify_draft'),('mc2_inventory_contract','preflight','preflight_MC2')):
  module=inherited_module(module_name);namespace=dict(vars(module),OUTPUT=output)
  old="directory=P/'startup-progress'" if module_name=='mc2_inventory_contract' else "Path(__file__).resolve().parent/'startup-progress'"
  new="directory=OUTPUT/'startup-progress'" if module_name=='mc2_inventory_contract' else "OUTPUT/'startup-progress'"
  # target_inventory also reads the actual current MC2 preflight directory.
  count=2 if module_name=='target_inventory' else 1
  changes=[(old,new,count)]
  if module_name in ('target_inventory','draft_inventory'):
   changes.append(('from inventory_memory import checkpoint','from progress179 import memory_checkpoint as checkpoint',1))
  functions(BASE/(module_name+'.py'),namespace,names=(function_name,),changes=tuple(changes))
  globals()[export]=namespace[function_name]
 return dict(output=str(output),read_sources=str(BASE),five_inventory_bodies_preserved=True,
             progress_files_private=True,Native_flags_assigned=False)
