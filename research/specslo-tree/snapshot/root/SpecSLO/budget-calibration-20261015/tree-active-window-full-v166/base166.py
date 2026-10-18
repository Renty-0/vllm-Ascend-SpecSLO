"""Small private epoch; all inherited source is read and pinned in place."""
import ast, copy, hashlib, importlib.util, json, sys
from pathlib import Path
P=Path(__file__).resolve().parent
B=P.parent
BASE=B/'tree-high-normal-cap4-native-families-v148'
ACTIVE=B/'tree-active-window32-16-local-v160'
ACTION='active-window-full463'
PARENT_ACTION='split192-candidate-full463'
for root in (BASE,ACTIVE):
 if str(root) not in sys.path:sys.path.append(str(root))
def require(ok,message):
 if ok is not True:raise RuntimeError('V166: '+message)
def sha(path):
 value=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
 return value.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def checked(path):
 path=Path(path);require(read(P/'INPUTS166.json')['files'].get(str(path))==sha(path),'exact source '+str(path));return path
def functions(path,namespace,*,names=None,changes=(),filename=None):
 path=checked(path);text=path.read_text()
 for old,new,count in changes:
  require(text.count(old)==count,'exact inherited source anchor '+old[:80]);text=text.replace(old,new)
 nodes=[copy.deepcopy(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
 require(names is None or {n.name for n in nodes}==set(names),'complete selected inherited definitions')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),filename or str(P/'source166.py')+'::'+path.name,'exec'),namespace)
 return text
def parent_ready():
 path=checked(BASE/'check_v132.py')
 spec=importlib.util.spec_from_file_location('_V166_readonly_P148_checker',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.require_local_launch(PARENT_ACTION)
def inherited_module(name):
 import importlib
 module=importlib.import_module(name)
 require(Path(vars(module)['__file__']).resolve()==BASE/(name+'.py'),'canonical inherited module '+name)
 checked(BASE/(name+'.py'));return module
def parent_archive():
 import tarfile
 pin=read(P/'INPUTS166.json')['parent_archive'];descriptor=Path(pin['descriptor'])
 require(sha(descriptor)==pin['descriptor_sha256'],'original P148 archive descriptor')
 data=read(descriptor);path=Path(data['path'])
 require(sha(path)==pin['sha256']==data['sha256'],'original P148 archive SHA in place')
 with tarfile.open(path,'r:gz') as archive:
  members=archive.getmembers();require(len(members)==len(data['members']) and len({m.name for m in members})==len(members),'original archive member inventory')
  for member in members:
   require(member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==data['members'][member.name],
           'original archive regular member bytes '+member.name)
 old=dict(path=str(path),sha256=pin['sha256'],members=len(members),verified_in_place=True,recopied=False)
 active_pin=read(P/'INPUTS166.json')['P160_archive'];active_path=Path(active_pin['path'])
 require(sha(active_path)==active_pin['sha256'],'immutable P160 archive SHA')
 frozen=read(ACTIVE/'FROZEN_SOURCE160.json')
 expected={**frozen['files'],**frozen['external_files'],**frozen['retained_HOST_evidence'],
  str(ACTIVE/'CPU_RESULT160.json'):sha(ACTIVE/'CPU_RESULT160.json'),
  str(ACTIVE/'FROZEN_SOURCE160.json'):sha(ACTIVE/'FROZEN_SOURCE160.json'),
  frozen['root_review']['path']:frozen['root_review']['sha256']}
 with tarfile.open(active_path,'r:gz') as archive:
  entries=archive.getmembers();require(len(entries)==len(expected)==42 and len({m.name for m in entries})==42,'P16042 source archive members')
  for member in entries:
   require(member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==expected['/'+member.name],'P160 member exact bytes')
 return dict(P148=old,P160=dict(path=str(active_path),sha256=active_pin['sha256'],members=42,verified_in_place=True,recopied=False))
