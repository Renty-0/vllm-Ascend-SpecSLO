"""Owned private compiler-header adapter; all numerical365 sources unchanged."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'tree-primary4-top2-chain-runtime-init-v367'))
import guard367 as parent
component=parent.component
sha=parent.sha
require=parent.require
EXPECTED_FREEZE=None
EXPECTED_ARCHIVE=None

def prepared():
    require(sha(HERE/'FREEZE368.json')==EXPECTED_FREEZE,'exact368 freeze')
    require(sha(HERE/'SOURCE368.tar.gz')==EXPECTED_ARCHIVE,'exact368 archive')
    own=json.loads((HERE/'FREEZE368.json').read_text())
    for filename,expected in own['files'].items():
        require(sha(filename)==expected,'immutable368 '+filename)
    inventory=dict(own['files'],**{str(HERE/'FREEZE368.json'):EXPECTED_FREEZE})
    with tarfile.open(HERE/'SOURCE368.tar.gz','r:gz') as saved:
        members=saved.getmembers()
        require({m.name for m in members}=={p.lstrip('/') for p in inventory} and len(members)==len(inventory),'exact368 inventory')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest()==inventory['/'+member.name],'exact368 archive bytes')
    pins=json.loads((parent.HERE/'SOURCE367_RECEIPT.json').read_text())
    parent.EXPECTED_FREEZE=pins['freeze_sha256']
    parent.EXPECTED_ARCHIVE=pins['archive_sha256']
    original,sources,readiness=parent.prepared()
    for filename in set(sources)&set(inventory):
        require(sources[filename]==inventory[filename],'identical368 dependency')
    sources.update(inventory)
    readiness.update(runtime368_freeze_sha256=EXPECTED_FREEZE,runtime368_archive_sha256=EXPECTED_ARCHIVE,original365_neural_files_unchanged=True,original_triton_device_initialization=True,host_precompiled_header_disabled=True,source_members=len(sources))
    return original,sources,readiness

def launch_argv(tag):
    argv=parent.launch_argv(tag)
    original=str(parent.HERE/'startup367.sh')
    require(argv.count(original)==1,'one original runtime launch')
    argv[argv.index(original)]=str(HERE/'startup368.sh')
    return argv

def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True)
    parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args()
    EXPECTED_FREEZE,EXPECTED_ARCHIVE=args.expected_freeze_sha256,args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None and all(re.fullmatch('[0-9a-f]{64}',p) for p in (EXPECTED_FREEZE,EXPECTED_ARCHIVE)),'safe explicit368 tag/pins')
    prepared()
    namespace=component.execution_namespace()
    namespace['prepared']=prepared
    namespace['launch_argv']=launch_argv
    lock=component.LOCK.open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()

if __name__=='__main__':
    main()
