"""Fresh complete common-Q11 model proof; no numerical gate is relaxed."""
import argparse
import ast
import copy
import fcntl
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'tree-M165-NZ166-target-math-cost-v320'
sys.path.insert(0,str(PREVIOUS))
import guard320 as previous
import guard318 as root
previous.PIN='1c809976a59f2158d99d24e3cc978b582fdbcdcb263821967a0a0f603e5712b1'
PIN=None

def prepared():
    old,sources,readiness=previous.prepared()
    manifest=P/'FREEZE323.json'
    root.previous.require(root.sha(manifest)==PIN,'explicit323 common-mask source pin')
    for path,wanted in json.loads(manifest.read_text())['files'].items():
        root.previous.require(root.sha(path)==wanted,'unchanged323 input '+path)
        sources[path]=wanted
    sources[str(manifest)]=PIN
    readiness.update(own323_freeze_sha256=PIN,common_Q11_mask_capacity=True,all546_model_gates_required=True)
    return old,sources,readiness

def launch_argv(tag):
    argv=previous.launch_argv(tag)
    argv[argv.index(str(PREVIOUS/'launch320.sh'))]=str(P/'launch323.sh')
    return argv

def storage(tag=None):
    row=previous.storage(tag)
    row['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    root.previous.require(row['aggregate_private_bytes']<=root.MAX_AGGREGATE_BYTES,'bounded323 private component')
    return row

def actual_audit(directory):
    # Retain every original acceptance predicate; replace only reconstruction
    # of the explicitly changed physical mask capacity11 for all families.
    old=root.previous
    source=Path(old.__file__).read_text()
    node=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='actual_audit'))
    changed=0
    for n in ast.walk(node):
        if isinstance(n,ast.ImportFrom)and n.module=='audit_static_calls317':
            n.module='audit_static_calls323';changed+=1
    old.require(changed==1,'one explicit physical-reconstruction import')
    namespace=dict(vars(old))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__+'::original-audit-common-mask','exec'),namespace)
    result=namespace['actual_audit'](directory)
    for rank in (1,2,3):
        report=json.loads((Path(directory)/f'rank{rank}-numeric.json').read_text())
        raw=report['fresh165_raw_operator']['checks']
        old.require({c['slots_dtype']for c in raw}=={'torch.int32','torch.int64'}and all(c['reference_slots_dtype']=='torch.int32'and c['reference_slots_values_exactly_representable_int32']for c in raw),'both candidate slot dtypes retained with supported native reference API')
        old.require(all(c['mask_shape']==[33,1,11,4096]for c in report['actual_static_graph_calls']),'all actual positive model graphs use common11 capacity')
    return result

def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    root.previous.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit323 pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=root.previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=actual_audit,storage=storage,MAX_AGGREGATE_BYTES=root.MAX_AGGREGATE_BYTES)
    with root.previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
