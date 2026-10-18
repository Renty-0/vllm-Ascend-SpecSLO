"""Three explicit worker extension hooks; P165 can provide its own module."""
def prepare_install(output_directory):
 from source166 import entry_namespace
 from progress166 import configure
 from candidate_controls_v132 import CandidateControls
 configure(output_directory)
 return entry_namespace(output_directory),CandidateControls
def after_install(native,graph,live):
 return dict(P159_installed=False,active_only=True)
def loaded_origins():
 from import_origins_v132 import require_loaded_origins
 return require_loaded_origins()
def qualify_component(engine,native,stage,policy,output_directory):
 from component166 import run
 return run(engine,native,stage,policy,output_directory)
