"""Guard331 return-contract fix; exact331 model/runtime remain unchanged."""
import argparse
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'tree-FIA160-projection165-MC2-target-math-cost-v331'
sys.path.insert(0,str(PREVIOUS))
import guard331 as previous
previous.PIN='fea0e3c69858b0e8b4b35f82332a103ed256f9a0733980541b5c6785bb3df569'
DRIVER=previous.previous.previous.original.previous
MAX_AGGREGATE_BYTES=previous.previous.previous.original.MAX_AGGREGATE_BYTES
PIN=None


def prepared():
    frozen,sources,readiness=previous.prepared()
    manifest=P/'FREEZE332.json'
    DRIVER.require(PIN is not None and hashlib.sha256(manifest.read_bytes()).hexdigest()==PIN,'explicit guard332 pin')
    own=json.loads(manifest.read_text())
    for path,sha in own['files'].items():
        DRIVER.require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,'unchanged guard332 source')
    sources.update(own['files']);sources[str(manifest)]=PIN
    readiness.update(guard332_freeze_sha256=PIN,actual331_neural_runtime_retained=True,
                     actual_audit_tuple_contract_preserved=True)
    return frozen,sources,readiness


def corrected_audit():
    source=inspect.getsource(previous.actual_audit)
    before="result['bridge331_actual_FIA160_projection165_fresh532_passed']=True"
    after="result[0]['bridge331_actual_FIA160_projection165_fresh532_passed']=True"
    DRIVER.require(source.count(before)==1,'one exact guard331 tuple-result correction')
    changed=source.replace(before,after)
    DRIVER.require(changed.replace(after,before)==source,'whole guard331 audit inverse')
    namespace=dict(vars(previous))
    exec(compile(changed,__file__+'::331-original-audit-tuple-fix','exec'),namespace)
    return namespace['actual_audit']


def actual_audit(directory):
    return corrected_audit()(directory)


def storage(tag=None):
    row=previous.storage(tag)
    row['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    DRIVER.require(row['aggregate_private_bytes']<=MAX_AGGREGATE_BYTES,'bounded guard332')
    return row


def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    DRIVER.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit guard332 pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=DRIVER.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=previous.launch_argv,actual_audit=actual_audit,storage=storage,
                     ACTION=previous.previous.previous.ACTION,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES)
    with DRIVER.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
