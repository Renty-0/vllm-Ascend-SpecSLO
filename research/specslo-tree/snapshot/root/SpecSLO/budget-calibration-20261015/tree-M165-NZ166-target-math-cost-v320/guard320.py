"""Controlled NZ threshold comparison, unchanged complete318 qualification."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'tree-M165-raw-reference-api-v318'
sys.path.insert(0,str(PREVIOUS))
import guard318 as previous
previous.PIN='c6bdba824d76400a101ea8054aac31206bdf1ca4182d5c016eefd387d4f98a96'
PIN=None

def prepared():
    old,sources,readiness=previous.prepared()
    manifest=P/'FREEZE320.json'
    previous.previous.require(previous.sha(manifest)==PIN,'explicit320 NZ-threshold-control pin')
    frozen=json.loads(manifest.read_text())
    for path,wanted in frozen['files'].items():
        previous.previous.require(previous.sha(path)==wanted,'unchanged320 input '+path)
    sources.update(frozen['files']);sources[str(manifest)]=PIN
    readiness.update(own320_freeze_sha256=PIN,NZ_threshold=166,original_matchers_unchanged=True)
    return old,sources,readiness

def launch_argv(tag):
    argv=previous.launch_argv(tag)
    argv[argv.index(str(PREVIOUS/'launch318.sh'))]=str(P/'launch320.sh')
    return argv

def storage(tag=None):
    row=previous.storage(tag)
    row['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    previous.previous.require(row['aggregate_private_bytes']<=previous.MAX_AGGREGATE_BYTES,'bounded private320 additive control')
    return row

def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    previous.previous.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit320 pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=previous.previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=previous.actual_audit,storage=storage,MAX_AGGREGATE_BYTES=previous.MAX_AGGREGATE_BYTES)
    with previous.previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
