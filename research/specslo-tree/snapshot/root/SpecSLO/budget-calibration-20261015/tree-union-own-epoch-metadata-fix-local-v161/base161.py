"""Small new namespace over immutable P157 and P154; no parent writes."""
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
Q=B/'tree-natural-positive-union-discovery-local-v154'
if str(Q) not in sys.path:sys.path.insert(1,str(Q))
ACTION='natural-positive-union-own-epoch-metadata-fix-local'
def require(value,message):
 if value is not True:raise RuntimeError('V161: '+message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record():return json.loads((P/'INPUTS161.json').read_text())
def checked(path):
 path=Path(path);require(record()['files'].get(str(path))==sha(path),'exact readonly dependency '+str(path));return path
def parent_guard():
 path=checked(B/'budget-diagnostics/v154-root-execution/common_guard.py')
 spec=importlib.util.spec_from_file_location('_V161_readonly_P154_guard',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def parent_ready():
 module=parent_guard();frozen,sources,ready=module.prepared();r=record()
 require(ready['freeze_sha256']==r['P154_freeze_SHA256'] and ready['archive']['sha256']==r['P154_archive_SHA256'],'readonly P154 prepared source/archive')
 return module,ready
def functions(path,namespace,*,names=None,changes=()):
 path=checked(path);text=path.read_text()
 for old,new,count in changes:
  require(text.count(old)==count,'exact inherited source delta '+old[:60]);text=text.replace(old,new)
 nodes=[copy.deepcopy(n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and (names is None or n.name in names)]
 require(names is None or {n.name for n in nodes}==set(names),'selected inherited functions')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::V161-isolated','exec'),namespace)
 namespace['_inherited_origin']=dict(path=str(path),sha256=sha(path),changes=[list(v) for v in changes])
def bind_source_epoch():
 import source154
 path=P/'source_epoch.py';old=sys.modules.get('source_epoch')
 if old is not None:
  value=vars(old);origin=value.get('__file__')
  require(type(origin)is str and Path(origin).resolve()==path and value.get('read') is source154.read,'reject unexpected loaded source_epoch')
  return old
 spec=importlib.util.spec_from_file_location('source_epoch',path);module=importlib.util.module_from_spec(spec);sys.modules['source_epoch']=module
 try:spec.loader.exec_module(module)
 except BaseException:
  if sys.modules.get('source_epoch') is module:sys.modules.pop('source_epoch')
  raise
 require(vars(module).get('read') is source154.read,'same actual current source reader');return module
