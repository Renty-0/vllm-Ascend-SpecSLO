"""Small private epoch; all inherited source is read and pinned in place."""
import ast, copy, hashlib, importlib.util, json, sys
from pathlib import Path
P=Path(__file__).resolve().parent
B=P.parent
BASE=B/'tree-high-normal-cap4-native-families-v148'
ACTIVE=B/'tree-active-window32-16-local-v160'
ACTION='causal80-N9-to15-allB4-empty-active32-full463'
PARENT_ACTION='split192-candidate-full463'
P173=B/'tree-causal80-real16-bankfix-full-v173'
for root in (BASE,ACTIVE,P173):
 if str(root) not in sys.path:sys.path.append(str(root))
def require(ok,message):
 if ok is not True:raise RuntimeError('V205: '+message)
def sha(path):
 value=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
 return value.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def checked(path):
 path=Path(path);require(read(P/'INPUTS205.json')['files'].get(str(path))==sha(path),'exact source '+str(path));return path
def functions(path,namespace,*,names=None,changes=(),filename=None):
 path=checked(path);text=path.read_text()
 for old,new,count in changes:
  require(text.count(old)==count,'exact inherited source anchor '+old[:80]);text=text.replace(old,new)
 nodes=[copy.deepcopy(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
 require(names is None or {n.name for n in nodes}==set(names),'complete selected inherited definitions')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),filename or str(P/'source205.py')+'::'+path.name,'exec'),namespace)
 return text
def parent_ready():
 path=checked(BASE/'check_v132.py')
 spec=importlib.util.spec_from_file_location('_V205_readonly_P148_checker',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module.require_local_launch(PARENT_ACTION)
def inherited_module(name):
 import importlib
 module=importlib.import_module(name)
 require(Path(vars(module)['__file__']).resolve()==BASE/(name+'.py'),'canonical inherited module '+name)
 checked(BASE/(name+'.py'));return module
def parent_archive():
 import tarfile
 pin=read(P/'INPUTS205.json')['parent_archive'];descriptor=Path(pin['descriptor'])
 require(sha(descriptor)==pin['descriptor_sha256'],'original P148 archive descriptor')
 data=read(descriptor);path=Path(data['path'])
 require(sha(path)==pin['sha256']==data['sha256'],'original P148 archive SHA in place')
 with tarfile.open(path,'r:gz') as archive:
  members=archive.getmembers();require(len(members)==len(data['members']) and len({m.name for m in members})==len(members),'original archive member inventory')
  for member in members:
   require(member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==data['members'][member.name],
           'original archive regular member bytes '+member.name)
 old=dict(path=str(path),sha256=pin['sha256'],members=len(members),verified_in_place=True,recopied=False)
 active_pin=read(P/'INPUTS205.json')['P160_archive'];active_path=Path(active_pin['path'])
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


def parent_module(name):
 import importlib
 module=importlib.import_module(name)
 require(Path(vars(module)['__file__']).resolve()==P173/(name+'.py'),'readonly production parent173 module '+name)
 checked(P173/(name+'.py'));return module

P197=B/'tree-empty-active-cohort-causal80-active32-full-v197'
def require_production_parent(*,archive=False):
 import tarfile
 pin=read(P/'INPUTS205.json')['P197'];f=read(P197/'FROZEN_SOURCE197.json')
 require(sha(P197/'FROZEN_SOURCE197.json')==pin['freeze_sha256'],'sole frozen production parent197')
 expected={**f['files'],**f['external_files'],**f['retained_HOST_evidence'],**f['execution_guard_inputs'],**f['peer_reviews'],str(P197/'CPU_RESULT197.json'):f['CPU_sha256'],str(P197/'FROZEN_SOURCE197.json'):pin['freeze_sha256'],f['root_review']['path']:f['root_review']['sha256']}
 for p,h in expected.items():require(sha(p)==h,'immutable197 source/evidence '+p)
 if archive:
  path=Path(pin['archive']['path']);require(sha(path)==pin['archive']['sha256'],'original197 archive in place')
  with tarfile.open(path,'r:gz')as a:
   members=a.getmembers();require(len(members)==len(expected)and len({m.name for m in members})==len(members),'197 exact regular archive inventory')
   for m in members:require(m.isfile()and '/'+m.name in expected and hashlib.sha256(a.extractfile(m).read()).hexdigest()==expected['/'+m.name],'197 archive member '+m.name)
 return dict(production_parent=str(P197),archive_verified_in_place=archive,device_qualification_inherited=False)


def require_source_parent(*,archive=False):
 import tarfile
 pin=read(P/'INPUTS205.json')['source_parent202'];root=Path(pin['path']);f=read(root/'FROZEN_SOURCE202.json')
 require(sha(root/'FROZEN_SOURCE202.json')==pin['freeze_sha256']and sha(root/'CPU_RESULT202.json')==pin['CPU_sha256'],'immutable202 REV3 source donor')
 expected={**f['files'],**f['external_files'],**f['retained_HOST_evidence'],**f['execution_guard_inputs'],**f['peer_reviews'],str(root/'CPU_RESULT202.json'):f['CPU_sha256'],str(root/'FROZEN_SOURCE202.json'):pin['freeze_sha256'],f['root_review']['path']:f['root_review']['sha256']}
 for path,h in expected.items():require(sha(path)==h,'exact202 source-donor byte '+path)
 if archive:
  ar=pin['archive'];require(sha(ar['path'])==ar['sha256'],'source donor202 archive SHA in place')
  with tarfile.open(ar['path'],'r:gz')as saved:
   members=saved.getmembers();require(len(members)==len(expected)==ar['members']and len({m.name for m in members})==len(members),'exact202 source-donor archive inventory')
   for m in members:require(m.isfile()and '/'+m.name in expected and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==expected['/'+m.name],'exact202 regular source-donor member '+m.name)
 return dict(source_parent=str(root),members=len(expected),archive_verified_in_place=archive,recopied=False,production_parent_remains197=True,device_qualification_inherited=False)
