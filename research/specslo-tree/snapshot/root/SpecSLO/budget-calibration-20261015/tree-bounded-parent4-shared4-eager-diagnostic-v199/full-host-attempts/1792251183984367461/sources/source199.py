"""Fresh namespaces with exact IO-only changes; no Native import on import."""
import ast, copy, types
from pathlib import Path
from base199 import P,BASE,checked,require,functions,inherited_module

ENTRY_CHANGES=(
 ("(P/'runtime_tree_decode.py').write_text(source)","(OUTPUT/'runtime_tree_decode.py').write_text(source)",1),
 ("compile(source,str(P/'runtime_tree_decode.py'),'exec')","compile(source,str(OUTPUT/'runtime_tree_decode.py'),'exec')",1),
 ("progress_dir=P/'generation-progress'","progress_dir=OUTPUT/'generation-progress'",1),
 ('from prefill_inventory import qualify as qualify_prefill','from progress199 import qualify_prefill',1),
 ('from target_inventory import qualify as qualify_target','from progress199 import qualify_target',1),
 ('from draft_inventory import qualify as qualify_draft','from progress199 import qualify_draft',1),
 ('from mc2_inventory_contract import preflight as preflight_MC2','from progress199 import preflight_MC2',1),
)

def entry_namespace(output):
 original=inherited_module('tree_entry')
 output=Path(output).resolve();require(output.is_relative_to(P) and output.is_dir(),'current private output exists')
 namespace=dict(vars(original),OUTPUT=output)
 text=functions(BASE/'tree_entry.py',namespace,changes=ENTRY_CHANGES,filename=str(P/'source199.py')+'::private-tree-entry')
 reverse=text
 for old,new,count in reversed(ENTRY_CHANGES):
  require(reverse.count(new)==count,'IO-only inverse uniqueness');reverse=reverse.replace(new,old)
 require(reverse==(BASE/'tree_entry.py').read_text(),'entire inherited tree-entry inverse')
 return namespace

def startup_auditor(output):
 original=inherited_module('startup_receipts_v75');namespace=dict(vars(original),OUTPUT=Path(output))
 functions(BASE/'startup_receipts_v75.py',namespace,names=('audit_actual_startup',),changes=(
  ("directory = P/'startup-progress'","directory = OUTPUT/'startup-progress'",1),
  ('from inventory_memory import checkpoint','from progress199 import memory_checkpoint as checkpoint',1)))
 return namespace['audit_actual_startup']

def expected_code(entry,output):
 """Change only expected identities for the reviewed IO-only shell/delegate."""
 original=inherited_module('poisson_coverage_probe')
 def expected(path,name):
  path=Path(path)
  if path==BASE/'tree_entry.py' and name=='fresh_decode':
   text=(BASE/'tree_entry.py').read_text()
   for old,new,count in ENTRY_CHANGES:text=text.replace(old,new)
   nodes=[copy.deepcopy(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
   unit=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
   return original._find_code(compile(unit,str(P/'source199.py')+'::private-tree-entry','exec'),name)
  if path==BASE/'runtime_tree_decode.py' and name=='_generate_spec_rhythm_tree_decode':
   return original._find_code(compile((Path(output)/'runtime_tree_decode.py').read_text(),str(Path(output)/'runtime_tree_decode.py'),'exec'),name)
  return original._expected_code(path,name)
 return expected

def original_observation(entry,output):
 original=inherited_module('poisson_coverage_probe');namespace=dict(vars(original),_expected_code=expected_code(entry,output))
 functions(BASE/'poisson_coverage_probe.py',namespace,names=('_assert_original_decode',))
 proxy=types.ModuleType('_V199_original_trace_source');proxy.__dict__.update(vars(original));proxy._assert_original_decode=namespace['_assert_original_decode']
 proxy._expected_code=namespace['_expected_code']
 return proxy
