"""Concrete P159 hooks for the shared full runner; no import-time install."""
import json,sys
from pathlib import Path
from base165 import require,V148,V159,checked,sha,bind_source_epoch
_factory=None
_entry=None

def prepare_install(output_directory):
 global _factory,_entry
 from source159 import module
 # The future shared runner's redirection is only I/O; its P148 source inputs,
 # actual constructor and real inventory qualifiers remain unchanged.
 from binding165 import entry_namespace
 from progress165 import configure
 from component165 import qualifying_capability
 import live159
 bind_source_epoch()
 _factory=live159.prepare_helpers(qualifying_capability())
 configure(output_directory);_entry=entry_namespace(output_directory)
 return _entry,_factory['controls']['CandidateControls']

def after_install(native,graph,live):
 import live159,observer151
 require(_factory is not None and _entry is not None,'own pre-install hooks completed')
 status=live159.finish_install(native,live)
 _entry['_live_builder']=live.build_live_runtime
 # Original full constructor already qualifies every draft PA family using
 # its own F/T/F actor. Do not install the local cold-key auto-qualifier ahead
 # of that inventory: it would change the original first/changed boundaries.
 observer151.install(native,graph)
 record=graph.NativeACLGraphRunner._record_execution
 def tracked(runner,kind,execution,output,*,entry_key=None,token_rows=()):
  if kind=='target':runner._v151_last_target_key=entry_key
  return record(runner,kind,execution,output,entry_key=entry_key,token_rows=token_rows)
 graph.NativeACLGraphRunner._record_execution=tracked
 return dict(P159_installed=True,new_shared4_and_union=True,common_draft_status=True,
  mathematical_or_AR_qualification_inherited=False,model_call_or_flag_from_HOST=False)

def loaded_origins():
 """Named canonical modules plus the actual new function namespaces.

 No getattr probing of arbitrary sys.modules / torch.classes pseudo modules.
 """
 import importlib
 from import_origins_v132 import KEYS
 require(_factory is not None,'real prepared new factory')
 result={}
 for name in KEYS:
  if name not in sys.modules:importlib.import_module(name)
  module=sys.modules.get(name)
  require(module is not None,'required actual canonical module '+name)
  path=Path(vars(module).get('__file__','')).resolve()
  wanted=V159/'fast159.generated.py' if name=='fast_tree_draft_v5' else V148/(name+'.py')
  require(path==wanted,'explicit actual canonical module origin '+name)
  checked(path);result[name]=dict(path=str(path),sha256=sha(path))
 for label,function in (('actual_shape',_factory['shape']['shape']),('actual_controls',_factory['controls']['CandidateControls'].__init__)):
  path=Path(function.__code__.co_filename).resolve();checked(path)
  result[label]=dict(path=str(path),sha256=sha(path),actual_callable=True)
 return result

def qualify_component(engine,native,stage,policy,output_directory):
 from component165 import run
 return run(engine,native,stage,policy,output_directory)
