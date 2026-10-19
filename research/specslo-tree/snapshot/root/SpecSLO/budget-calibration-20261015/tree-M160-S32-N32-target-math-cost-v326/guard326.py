"""Fresh M160 raw/model/cost component, original verifier and owned cleanup."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
B=P.parent
sys.path[:0]=[str(B/'tree-M165-raw-reference-api-v318'),str(B/'tree-M165-N32-independent-target-math-cost-v317')]
import guard318 as original
original.PIN='c6bdba824d76400a101ea8054aac31206bdf1ca4182d5c016eefd387d4f98a96'
PIN=None
ACTION='alternate-chain-B1to8-M160-N32-node-component'

def prepared():
    old,sources,readiness=original.prepared()
    from check326 import ready
    frozen=ready(PIN)
    sources.update(frozen['files']);sources[str(P/'FREEZE326.json')]=PIN
    readiness.update(own326_freeze_sha256=PIN,physical_M=160,physical_groups=32,declared_neural_cases=532,
                     original408_budgets_and_anchors_retained=True,device_qualification_inherited=False)
    return frozen,sources,readiness

def launch_argv(tag):
    return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash','LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/sbin:/usr/bin:/bin',
        'SPECSLO_V326_ROOT_FREEZE_SHA='+PIN,'/bin/bash',str(P/'launch326.sh'),'--action',ACTION,'--tag',tag]

def storage(tag=None):
    row=original.storage(tag)
    row['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    original.previous.require(row['aggregate_private_bytes']<=original.MAX_AGGREGATE_BYTES,'bounded private326 diagnostic')
    return row

def actual_audit(directory):
    # The function is copied from the original317 auditor with only physical
    # names and case count changes. Its exact token/hidden/KV predicates stay.
    namespace=dict(vars(original.previous),HERE=P)
    exec(compile((P/'actual_audit326.py').read_text(),str(P/'actual_audit326.py'),'exec'),namespace)
    result=namespace['actual_audit'](directory)
    for rank in (1,2,3):
        report=json.loads((Path(directory)/f'rank{rank}-numeric.json').read_text())
        raw=report['fresh160_raw_operator']['checks']
        original.previous.require({r['slots_dtype']for r in raw}=={'torch.int32','torch.int64'}and all(r['reference_slots_dtype']=='torch.int32'for r in raw),'both candidate dtypes, supported native reference')
    return result

def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    original.previous.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit326 source pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=original.previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=actual_audit,storage=storage,ACTION=ACTION,MAX_AGGREGATE_BYTES=original.MAX_AGGREGATE_BYTES)
    with original.previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
