"""Read pinned functions into fresh explicit namespaces; never edit parents."""
import ast,copy,json
from pathlib import Path
from scope154 import P,P151,P152,require,sha
def inputs():return json.loads((P/'INPUTS154.json').read_text())
def checked(path):
 path=Path(path);require(inputs()['files'].get(str(path))==sha(path),'pinned read-only source '+str(path));return path
def functions(path,namespace,*,names=None,changes=()):
 path=checked(path);text=path.read_text()
 for old,new,count in changes:
  require(text.count(old)==count,'exact isolated source edit '+str(path));text=text.replace(old,new)
 nodes=[copy.deepcopy(n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and (names is None or n.name in names)]
 require(names is None or {n.name for n in nodes}==set(names),'all named original functions exist')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::V154-isolated','exec'),namespace)
 namespace['_inherited_origin']=dict(path=str(path),sha256=sha(path),names=None if names is None else list(names),changes=[list(c) for c in changes])
