"""Enumerate every extension attribute used by the actual frozen437 worker."""
import ast,importlib.abc,json,os,re,shlex,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-refill-witness-boolean-retention-full-v437'
F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST no model import '+fullname)
sys.meta_path.insert(0,NoModel())
for line in(P/'launch437.sh').read_text().splitlines():
    for value in shlex.split(line.rstrip(' \\')):
        if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
            key,setting=value.split('=',1);os.environ[key]=setting
os.environ['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
value=next(l for l in(P/'startup437.sh').read_text().splitlines()if l.startswith('export PYTHONPATH=')).removeprefix('export PYTHONPATH=')
for old,new in(('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')),('${PYTHONPATH:-}','')):value=value.replace(old,new)
assert '$'not in value
sys.path[:0]=value.split(':')
import runtime437,extension437
source=runtime437.source();(D/'ROOT438_READONLY_FROZEN437_ACTUAL_WORKER_SOURCE.py').write_text(source)
tree=ast.parse(source);attrs=sorted({n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='extension'})
missing=[name for name in attrs if not hasattr(extension437,name)]
print('ACTUAL_EXTENSION_ATTRIBUTES',attrs);print('MISSING',missing)
lines=source.splitlines();print('\n'.join(f'{i+1}:{lines[i]}'for i in range(85,min(120,len(lines)))))
result=dict(status='HOST_ACTUAL437_EXTENSION_INTERFACE_INVENTORY',referenced_attributes=attrs,missing_attributes=missing,NPU_used=False)
(D/'ROOT438_FROZEN437_INTERFACE_INVENTORY_HOST.json').write_text(json.dumps(result,indent=2)+'\n')
