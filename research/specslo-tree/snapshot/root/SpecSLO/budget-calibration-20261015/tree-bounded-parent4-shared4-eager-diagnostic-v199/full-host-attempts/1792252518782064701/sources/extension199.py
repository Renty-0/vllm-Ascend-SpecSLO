"""Original173 install/current baseline, explicit bounded FastV5 origin only."""
def prepare_install(output):
 import importlib.util,sys
 from pathlib import Path
 from base199 import P,require,parent_module
 from source199 import entry_namespace
 from progress199 import configure
 from live199 import prepare_helpers
 from epoch199 import read
 path=P/'source_epoch.py';require('source_epoch'not in sys.modules,'never replace foreign canonical source reader')
 spec=importlib.util.spec_from_file_location('source_epoch',path);m=importlib.util.module_from_spec(spec);sys.modules['source_epoch']=m
 try:spec.loader.exec_module(m)
 except BaseException:sys.modules.pop('source_epoch',None);raise
 require(m.read is read,'explicit current source read identity');prepare_helpers();configure(output)
 return entry_namespace(output),parent_module('mapper173').controls_class()
def after_install(native,graph,live):
 from base199 import V151,functions,require
 import types
 ns=dict(__name__='_v199_original_observer',__file__=str(V151/'observer151.py'))
 functions(V151/'observer151.py',ns,names=('install',));ns['install'](native,graph)
 record=graph.NativeACLGraphRunner._record_execution
 def tracked(runner,kind,execution,output,*,entry_key=None,token_rows=()):
  if kind=='target':runner._v151_last_target_key=entry_key
  return record(runner,kind,execution,output,entry_key=entry_key,token_rows=token_rows)
 graph.NativeACLGraphRunner._record_execution=tracked
 return dict(sole_production_parent173=True,new_bounded_P4_E4=True,production_gates_opened=False,actual_steps=4,E3_source_contrast_only=True)
def loaded_origins():
 import sys,importlib
 from pathlib import Path
 from base199 import BASE,P,sha,checked,read,require
 from import_origins_v132 import KEYS
 pins=read(P/'FROZEN_SOURCE199.json')['files'];values={}
 for name in KEYS:
  module=importlib.import_module(name);path=Path(vars(module).get('__file__','')).resolve();expected=P/'fast199.generated.py'if name=='fast_tree_draft_v5'else BASE/(name+'.py')
  require(path==expected,'actual canonical origin '+name)
  if name=='fast_tree_draft_v5':require(pins.get(str(path))==sha(path),'ROOT frozen exact diagnostic fast source')
  else:checked(path)
  values[name]=dict(path=str(path),sha256=sha(path))
 for name,module in tuple(sys.modules.items()):
  path=vars(module).get('__file__')if hasattr(module,'__dict__')else None
  if type(path)is str and Path(path).is_absolute()and Path(path).resolve().is_relative_to(P):
   path=Path(path).resolve();require(path.is_file()and pins.get(str(path))==sha(path),'all genuine own module absolute source '+name)
 return values
def qualify_component(engine,native,stage,policy,output):
 from component199 import run
 return run(engine,native,stage,policy,output)
