"""Additive diagnostic routing fix; original numerical gates stay in place."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'tree-M165-first-divergence-v321'
sys.path.insert(0,str(PREVIOUS))
import guard321 as previous
import guard318 as root
previous.PIN='14d7d5c3dae2e1f7ec4e02205df5ffd9e5722af6e37dbf1b9e8993107f1c05af'
PIN=None

def prepared():
    old,sources,readiness=previous.prepared()
    manifest=P/'FREEZE322.json'
    root.previous.require(root.sha(manifest)==PIN,'explicit322 diagnostic-routing repair pin')
    for path,wanted in json.loads(manifest.read_text())['files'].items():
        root.previous.require(root.sha(path)==wanted,'unchanged322 input '+path)
        sources[path]=wanted
    sources[str(manifest)]=PIN
    readiness.update(own322_freeze_sha256=PIN,negative_control_recorder_routing_fixed=True)
    return old,sources,readiness

def launch_argv(tag):
    argv=previous.launch_argv(tag)
    argv[argv.index(str(PREVIOUS/'launch321.sh'))]=str(P/'launch322.sh')
    return argv

def storage(tag=None):
    row=previous.storage(tag)
    row['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    root.previous.require(row['aggregate_private_bytes']<=root.MAX_AGGREGATE_BYTES,'bounded322 private diagnostic')
    return row

def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    root.previous.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit322 pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=root.previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=root.actual_audit,storage=storage,MAX_AGGREGATE_BYTES=root.MAX_AGGREGATE_BYTES)
    with root.previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
