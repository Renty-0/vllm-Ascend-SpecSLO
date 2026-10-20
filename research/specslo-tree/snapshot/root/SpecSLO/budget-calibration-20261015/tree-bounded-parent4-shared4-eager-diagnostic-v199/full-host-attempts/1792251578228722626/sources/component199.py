"""One fresh173 baseline factory; only own FastV5 origin binding differs."""
from base199 import P173,BASE,functions,inherited_module,require,parent_module
from pathlib import Path
_PARENT=parent_module('component173')
def baseline(engine,native,stage,policy,output):
 def selected_functions(path,namespace,**kwargs):
  if Path(path)==BASE/'actor_v146.py':
   kwargs['changes']=tuple(kwargs.get('changes',()))+((
    'from import_origins_v132 import require_loaded_origins as assert_origins',
    'from extension199 import loaded_origins as assert_origins',1),)
  return functions(path,namespace,**kwargs)
 ns=dict(vars(_PARENT),functions=selected_functions)
 functions(P173/'component173.py',ns)
 return ns['run'](engine,native,stage,policy,output)
def run(engine,native,stage,policy,output):
 proof=baseline(engine,native,stage,policy,output)
 stage('same-current173-proof',lambda:require(proof is engine._v173_qualified_component,'same current original173 baseline object'))
 from actor199 import run as bounded
 extra=bounded(engine,native,stage,output)
 return dict(original173=proof,bounded=extra,production_permission=False,full463=False,formal_Goodput=False,whole_AR=False)
