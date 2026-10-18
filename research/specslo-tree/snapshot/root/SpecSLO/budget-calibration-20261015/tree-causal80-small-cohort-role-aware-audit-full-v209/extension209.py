"""Only173 production parent; new empty-active admission source and current math."""
def prepare_install(output_directory):
 from base209 import parent_module
 from source209 import entry_namespace
 from progress209 import configure
 configure(output_directory)
 from mapper209 import controls_class
 return entry_namespace(output_directory),controls_class()
def after_install(native,graph,live):
 from base209 import parent_module
 from mapper209 import install_pad
 install_pad(native)
 return dict(parent_module('extension173').after_install(native,graph,live),empty_active_refill_only=True,production_parent='frozen197',new_usage_real_N=list(range(9,16)),new_target_keys=0,new_MC2_profile=False,new_hot_collectives=0)
def loaded_origins():
 from base209 import parent_module,P,sha,read,require
 from pathlib import Path
 import sys
 original=parent_module('extension173').loaded_origins();pins=read(P/'FROZEN_SOURCE209.json')['files'];own={}
 for name,module in tuple(sys.modules.items()):
  path=vars(module).get('__file__')if hasattr(module,'__dict__')else None
  if type(path)is str and Path(path).is_absolute()and Path(path).resolve().is_relative_to(P):
   p=Path(path).resolve();require(p.is_file()and pins.get(str(p))==sha(p),'ROOT-frozen own source '+name);own[str(p)]=sha(p)
 return dict(original=original,private209_sources=own,no_parent_source_mutation=True)
def qualify_component(engine,native,stage,policy,output_directory):
 from component209 import run
 return run(engine,native,stage,policy,output_directory)
