"""Only173 production parent; new empty-active admission source and current math."""
def prepare_install(output_directory):
 from base202 import parent_module
 from source202 import entry_namespace
 from progress202 import configure
 configure(output_directory)
 from mapper202 import controls_class
 return entry_namespace(output_directory),controls_class()
def after_install(native,graph,live):
 from base202 import parent_module
 from mapper202 import install_pad
 install_pad(native)
 return dict(parent_module('extension173').after_install(native,graph,live),empty_active_refill_only=True,production_parent='frozen173',new_hot_collectives=0)
def loaded_origins():
 from base202 import parent_module,P,sha,read,require
 from pathlib import Path
 import sys
 original=parent_module('extension173').loaded_origins();pins=read(P/'FROZEN_SOURCE202.json')['files'];own={}
 for name,module in tuple(sys.modules.items()):
  path=vars(module).get('__file__')if hasattr(module,'__dict__')else None
  if type(path)is str and Path(path).is_absolute()and Path(path).resolve().is_relative_to(P):
   p=Path(path).resolve();require(p.is_file()and pins.get(str(p))==sha(p),'ROOT-frozen own source '+name);own[str(p)]=sha(p)
 return dict(original=original,private202_sources=own,no_parent_source_mutation=True)
def qualify_component(engine,native,stage,policy,output_directory):
 from component202 import run
 return run(engine,native,stage,policy,output_directory)
