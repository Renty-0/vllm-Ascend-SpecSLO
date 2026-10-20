"""Selected pinned original functions in explicit namespaces, no parent edit."""
import ast,copy,importlib.util,json,sys
from pathlib import Path
from scope152 import P,B,require,sha
Q=B/'tree-normal4-parent3-shared4-eager-v151'
if str(Q) not in sys.path:sys.path.insert(1,str(Q))
def record():return json.loads((P/'PARENT_FILES152.json').read_text())
def checked(path):
 path=Path(path);require(record()['files'].get(str(path))==sha(path),'pinned readonly original '+str(path));return path
def functions(name,namespace,*,changes=()):
 path=checked(Q/name);text=path.read_text()
 for old,new,count in changes:
  require(text.count(old)==count,'exact isolated inherited function delta '+name);text=text.replace(old,new)
 nodes=[copy.deepcopy(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::V152-explicit-namespace','exec'),namespace)
 namespace['_inherited_function_origin']=dict(path=str(path),sha256=sha(path),isolated_changes=[list(c) for c in changes])
def source_contexts(case):
 from originals151 import original_function
 return original_function('short_scope_v126','source_contexts')(case)
def install_runtime_helpers(native,graph):
 r=record();origins={}
 for name in r['runtime_load_order']:
  path=P/'fast_tree_draft_v5.py' if name=='fast_tree_draft_v5' else checked(r['module_origins'][name])
  old=sys.modules.get(name);require(old is None or Path(old.__file__).resolve()==path,'canonical helper actual origin '+name)
  if old is None:
   spec=importlib.util.spec_from_file_location(name,path);old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
  origins[name]=dict(path=str(path),sha256=sha(path),new_local_variant=name=='fast_tree_draft_v5')
 for name,arg in (('root_seed',native),('parallel_steps',graph),('host_target_metadata',native),('fixed_mask_extent',native),('frontier_qualification',native),('tree_target_replay',graph),('host_tree_transport',native),('canonical_eager',native),('tail_hidden_observer',graph)):
  sys.modules[name].install(arg)
 return origins
