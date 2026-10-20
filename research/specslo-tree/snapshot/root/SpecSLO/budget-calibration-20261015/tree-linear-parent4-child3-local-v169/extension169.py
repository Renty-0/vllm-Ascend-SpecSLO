"""Current private pre-install hooks; no shared4 parent4 production activation."""
import sys
from pathlib import Path
from base169 import P,V148,V159,require,checked,sha,bind_source_epoch
_factory=None
_entry=None

def prepare_install(output_directory):
 global _factory,_entry
 from binding169 import entry_namespace
 from progress169 import configure
 import live169
 bind_source_epoch();_factory=live169.prepare_helpers()
 configure(output_directory);_entry=entry_namespace(output_directory)
 return _entry,_factory['controls']['CandidateControls']

def after_install(native,graph,live):
 import observer151
 from status159 import install
 status=install(native,live);_entry['_live_builder']=live.build_live_runtime
 observer151.install(native,graph)
 old=graph.NativeACLGraphRunner._record_execution
 def record(runner,kind,execution,output,*,entry_key=None,token_rows=()):
  if kind=='target':runner._v151_last_target_key=entry_key
  return old(runner,kind,execution,output,entry_key=entry_key,token_rows=token_rows)
 graph.NativeACLGraphRunner._record_execution=record
 return dict(new_parent4_prelude_installed=True,controlled_local_actor_only=True,common_draft_status_installed=True,
  production_gates_opened=False,Native_flags_assigned=False)

def loaded_origins():
 import importlib
 from import_origins_v132 import KEYS
 require(_factory is not None,'new actual parent4 factory prepared')
 values={}
 for name in KEYS:
  if name not in sys.modules:importlib.import_module(name)
  module=sys.modules[name];path=Path(vars(module).get('__file__','')).resolve()
  wanted=P/'fast169.generated.py' if name=='fast_tree_draft_v5' else V148/(name+'.py')
  require(path==wanted,'named actual current canonical origin '+name)
  if name!='fast_tree_draft_v5':checked(path)
  values[name]=dict(path=str(path),sha256=sha(path))
 for label,function in (('actual_shape',_factory['shape']['shape']),('actual_controls',_factory['controls']['CandidateControls'].__init__)):
  path=Path(function.__code__.co_filename).resolve()
  wanted=P/'shape169.generated.py' if label=='actual_shape' else V159/'controls159.generated.py'
  require(path==wanted,'actual new parent admission/control function namespace')
  if label=='actual_controls':checked(path)
  values[label]=dict(path=str(path),sha256=sha(path),actual_callable=True)
 return values

def qualify_component(engine,native,stage,policy,output_directory):
 from component169 import run
 return run(engine,native,stage,policy,output_directory)
