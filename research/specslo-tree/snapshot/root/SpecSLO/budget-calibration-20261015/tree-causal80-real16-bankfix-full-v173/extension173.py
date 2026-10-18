"""Original draft/policy/Native startup, one separately qualified mapper arm."""
def prepare_install(output_directory):
 from source173 import entry_namespace
 from progress173 import configure
 from mapper173 import controls_class
 configure(output_directory)
 return entry_namespace(output_directory),controls_class()
def after_install(native,graph,live):
 return dict(P159_installed=False,active32_only=True,only_new_mapper_domain=[16,80,'causal'])
def loaded_origins():
 from import_origins_v132 import require_loaded_origins
 return require_loaded_origins()
def qualify_component(engine,native,stage,policy,output_directory):
 from component173 import run
 return run(engine,native,stage,policy,output_directory)
