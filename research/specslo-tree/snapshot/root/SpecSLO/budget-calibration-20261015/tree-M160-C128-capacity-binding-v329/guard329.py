"""Full fresh532 model/cost proof after actual installed capacity repair."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'tree-M160-S32-N32-target-math-cost-v326'
sys.path.insert(0,str(PREVIOUS))
import guard326 as previous
previous.PIN='1c8878b7490ed85684aea9686a25652b2488f30074769b0568243e5c476fbc43'
PIN=None

def prepared():
    old,sources,readiness=previous.prepared()
    from check329 import ready
    frozen=ready(PIN)
    sources.update(frozen['files']);sources[str(P/'FREEZE329.json')]=PIN
    readiness.update(own329_freeze_sha256=PIN,installed_accounting_C128_Q160_N32_groups32_F12=True,constructor_candidates128=True,
        partial410_not_inherited=True)
    return old,sources,readiness

def launch_argv(tag):
    argv=previous.launch_argv(tag)
    argv.insert(argv.index('/bin/bash'),'SPECSLO_V329_ROOT_FREEZE_SHA='+PIN)
    argv[argv.index(str(PREVIOUS/'launch326.sh'))]=str(P/'launch329.sh')
    return argv

def storage(tag=None):
    row=previous.storage(tag)
    row['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    previous.original.previous.require(row['aggregate_private_bytes']<=previous.original.MAX_AGGREGATE_BYTES,'bounded private329 fresh component')
    return row

def actual_audit(directory):
    result=previous.actual_audit(directory)
    for rank in range(4):
        proof=json.loads((Path(directory)/f'rank{rank}-capacity329-before-model.json').read_text())
        previous.original.previous.require(proof['rank']==rank and proof['new']==dict(C=128,Q=160,N=32,groups=32,branches=12)
            and proof['same_actual_accounting_policy_instance']is True and proof['model_loaded']is False
            and proof['constructor_candidates']==128,'all4 actual private capacity and constructor bindings before models')
    return result

def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    previous.original.previous.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit329 pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=previous.original.previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=actual_audit,storage=storage,ACTION=previous.ACTION,MAX_AGGREGATE_BYTES=previous.original.MAX_AGGREGATE_BYTES)
    with previous.original.previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
