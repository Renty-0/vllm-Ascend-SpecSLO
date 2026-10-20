"""Explicit frozen parent source functions/modules; no original globals edited."""
import ast,copy,importlib.util,json,sys
from functools import lru_cache
from dataclasses import replace
from pathlib import Path
from scope151 import P,B,require,sha
PARENT=B/'tree-short-parent-eager-v134'
@lru_cache(maxsize=None)
def pins():return json.loads((P/'PARENT_FILES151.json').read_text())['files']
def checked(path):
 path=Path(path);require(pins().get(str(path))==sha(path),'exact frozen parent source '+str(path));return path
@lru_cache(maxsize=None)
def original_module(name):
 record=json.loads((P/'PARENT_FILES151.json').read_text());path=checked(record['module_origins'][name])
 alias='_v151_readonly_original_'+name;spec=importlib.util.spec_from_file_location(alias,path);module=importlib.util.module_from_spec(spec)
 sys.modules[alias]=module;spec.loader.exec_module(module);return module
@lru_cache(maxsize=None)
def original_function(module_name,function_name):
 path=checked(PARENT/(module_name+'.py'));tree=ast.parse(path.read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function_name))
 namespace=dict(Path=Path,json=json,replace=replace,V102=B/'tree-packed-eager01-cost-v102',require=require)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path)+'::V151-unchanged-function','exec'),namespace)
 return namespace[function_name]
def source_contexts(case):return original_function('short_scope_v126','source_contexts')(case)
def install_runtime_helpers(native,graph):
 """Load actual canonical helper names at their pinned files, then original hooks."""
 record=json.loads((P/'PARENT_FILES151.json').read_text());origins={}
 for name in record['runtime_load_order']:
  path=checked(record['module_origins'][name]);old=sys.modules.get(name)
  require(old is None or Path(old.__file__).resolve()==path,'no alias to a foreign helper '+name)
  if old is None:
   spec=importlib.util.spec_from_file_location(name,path);old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
  origins[name]=dict(path=str(path),sha256=sha(path))
 # Every install changes runtime hooks only through the original reviewed API.
 for name,arg in (('root_seed',native),('parallel_steps',graph),('host_target_metadata',native),('fixed_mask_extent',native),
                  ('frontier_qualification',native),('tree_target_replay',graph),('host_tree_transport',native),('canonical_eager',native),('tail_hidden_observer',graph)):
  sys.modules[name].install(arg)
 return origins
