"""Original173 installation and readonly185 admission decision core."""
def prepare_install(output_directory):
 from base190 import parent_module,core_module
 for name in ('cohort185','lifecycle185','source_patch185'):core_module(name)
 from source190 import entry_namespace
 from progress190 import configure
 configure(output_directory)
 return entry_namespace(output_directory),parent_module('mapper173').controls_class()
def after_install(native,graph,live):
 from base190 import parent_module
 value=parent_module('extension173').after_install(native,graph,live)
 return dict(value,vacancy_cohort190=dict(minimum_requests=4,maximum_wait_ms=150.,maximum_requests=32),
  production_parent='frozen173',cap4_preserved=True,prefill_only_candidate=True)
def loaded_origins():
 from base190 import parent_module,core_module,P,sha,read,require
 from pathlib import Path
 import sys
 original=parent_module('extension173').loaded_origins()
 core={}
 for name in ('cohort185','lifecycle185','source_patch185'):
  module=core_module(name);p=Path(vars(module)['__file__']).resolve();core[str(p)]=sha(p)
 pins=read(P/'FROZEN_SOURCE190.json')['files'];own={}
 for name,module in tuple(sys.modules.items()):
  path=vars(module).get('__file__')if hasattr(module,'__dict__')else None
  if type(path)is str and Path(path).is_absolute()and Path(path).resolve().is_relative_to(P):
   p=Path(path).resolve();require(p.is_file()and pins.get(str(p))==sha(p),'ROOT-frozen own source '+name);own[str(p)]=sha(p)
 return dict(original=original,readonly185_decision_core=core,private190_sources=own,no_parent_source_mutation=True)
def qualify_component(engine,native,stage,policy,output_directory):
 from component190 import run
 return run(engine,native,stage,policy,output_directory)
