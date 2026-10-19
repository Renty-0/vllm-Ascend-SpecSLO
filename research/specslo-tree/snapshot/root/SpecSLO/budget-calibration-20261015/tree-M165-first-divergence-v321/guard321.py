"""Frozen original qualification with extra intermediate diagnostic records."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'tree-M165-NZ166-target-math-cost-v320'
sys.path.insert(0,str(PREVIOUS))
import guard320 as previous
previous.PIN='1c809976a59f2158d99d24e3cc978b582fdbcdcb263821967a0a0f603e5712b1'
PIN=None

def prepared():
    old,sources,readiness=previous.prepared()
    manifest=P/'FREEZE321.json'
    require=previous.previous.previous.require
    require(previous.previous.sha(manifest)==PIN,'explicit321 intermediate trace pin')
    for path,wanted in json.loads(manifest.read_text())['files'].items():
        require(previous.previous.sha(path)==wanted,'unchanged321 input '+path)
        sources[path]=wanted
    sources[str(manifest)]=PIN
    readiness.update(first_four_layers_eager_intermediates=True,own321_freeze_sha256=PIN)
    return old,sources,readiness

def launch_argv(tag):
    argv=previous.launch_argv(tag)
    argv[argv.index(str(PREVIOUS/'launch320.sh'))]=str(P/'launch321.sh')
    return argv

def storage(tag=None):
    row=previous.storage(tag)
    row['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    previous.previous.previous.require(row['aggregate_private_bytes']<=previous.previous.MAX_AGGREGATE_BYTES,'bounded321 private diagnostic')
    return row

def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    previous.previous.previous.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit321 pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=previous.previous.previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=previous.previous.actual_audit,storage=storage,MAX_AGGREGATE_BYTES=previous.previous.MAX_AGGREGATE_BYTES)
    with previous.previous.previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
