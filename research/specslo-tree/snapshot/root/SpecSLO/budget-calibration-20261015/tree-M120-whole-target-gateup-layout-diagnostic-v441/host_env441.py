"""Literal retained source-auth settings. No model import or process launch."""
from pathlib import Path
import os,re,shlex,sys
B=Path(__file__).resolve().parent.parent
CURRENT=B/'tree-slo-aware-refill-complete-interface-full-v438'
F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')

def source_environment():
    result={}
    for line in(CURRENT/'launch438.sh').read_text().splitlines():
        for item in shlex.split(line.rstrip(' \\')):
            if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',item):
                k,v=item.split('=',1);result[k]=v
    result['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
    value=next(l for l in(CURRENT/'startup438.sh').read_text().splitlines()if l.startswith('export PYTHONPATH=')).removeprefix('export PYTHONPATH=')
    for x,y in[('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')),('${PYTHONPATH:-}','')]:value=value.replace(x,y)
    if '$'in value:raise RuntimeError('ROOT441 literal retained source path')
    result['PYTHONPATH']=str(Path(__file__).resolve().parent)+':'+str(B/'tree-M120-linear-layout-cost-diagnostic-v440')+':'+value
    return result

def setup():
    env=source_environment();os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':')
    return env
