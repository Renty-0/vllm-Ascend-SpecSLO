"""Pure HOST source reuse for the new full527 process, no old epoch writes."""
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
V148=B/'tree-high-normal-cap4-native-families-v148'
V151=B/'tree-normal4-parent3-shared4-eager-v151'
V152=B/'tree-union-shared4-protocol-local-v152'
V154=B/'tree-natural-positive-union-discovery-local-v154'
V159=B/'tree-live-shared4-union-v159'
V161=B/'tree-union-own-epoch-metadata-fix-local-v161'
V149=B/'tree-repair-frontier-fusion-local-v149'
BASE=V148
ACTIVE=B/'tree-active-window32-16-local-v160'
ACTION='linear-parent4-child3-local'
PARENT_ACTION='split192-candidate-full463'
def read(path):return json.loads(Path(path).read_text())
for q in (V159,V161,V154,V151,V148,V149,ACTIVE):
 if str(q) not in sys.path:sys.path.append(str(q))
def require(ok,message):
 if ok is not True:raise RuntimeError('V169: '+message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked(path):
 path=Path(path);pins=json.loads((P/'INPUTS169.json').read_text())['files']
 require(pins.get(str(path))==sha(path),'current readonly source '+str(path));return path
def functions(path,namespace,*,names=None,changes=(),filename=None):
 path=checked(path);source=path.read_text()
 for old,new,count in changes:
  require(source.count(old)==count,'exact inherited anchor '+old[:70]);source=source.replace(old,new)
 nodes=[copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
 require(names is None or {n.name for n in nodes}==set(names),'all selected functions')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),filename or str(path)+'::P169-own-namespace','exec'),namespace)
 namespace['_inherited169']=dict(path=str(path),sha256=sha(path),changes=[list(v) for v in changes])
 return source
def inherited_module(name):
 import importlib
 module=importlib.import_module(name)
 require(Path(vars(module)['__file__']).resolve()==BASE/(name+'.py'),'actual inherited canonical module '+name)
 checked(BASE/(name+'.py'));return module
def parent_ready():
 path=checked(BASE/'check_v132.py');spec=importlib.util.spec_from_file_location('_V169_readonly_P148_checker',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.require_local_launch(PARENT_ACTION)
def bind_source_epoch():
 import source169
 path=P/'source_epoch.py';old=sys.modules.get('source_epoch')
 if old is not None:
  require(vars(old).get('__file__')==str(path) and vars(old).get('read') is source169.read,'never replace a loaded foreign source reader')
  return old
 spec=importlib.util.spec_from_file_location('source_epoch',path);m=importlib.util.module_from_spec(spec);sys.modules['source_epoch']=m
 try:spec.loader.exec_module(m)
 except BaseException:sys.modules.pop('source_epoch',None);raise
 require(m.read is source169.read,'actual current physical source KV reader');return m

def parent_archive():
 import tarfile
 pin=read(P/'INPUTS169.json')['parent_archive'];descriptor=Path(pin['descriptor'])
 require(sha(descriptor)==pin['descriptor_sha256'],'original P148 archive descriptor')
 data=read(descriptor);path=Path(data['path'])
 require(sha(path)==pin['sha256']==data['sha256'],'original P148 archive SHA in place')
 with tarfile.open(path,'r:gz') as archive:
  members=archive.getmembers();require(len(members)==len(data['members']) and len({m.name for m in members})==len(members),'original archive member inventory')
  for member in members:
   require(member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==data['members'][member.name],
           'original archive regular member bytes '+member.name)
 old=dict(path=str(path),sha256=pin['sha256'],members=len(members),verified_in_place=True,recopied=False)
 active_pin=read(P/'INPUTS169.json')['P160_archive'];active_path=Path(active_pin['path'])
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
