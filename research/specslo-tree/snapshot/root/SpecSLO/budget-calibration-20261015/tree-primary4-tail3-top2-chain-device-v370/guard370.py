"""Owned real draft first/catch-up plus tail3 component; full original lifecycle."""
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
sys.path.insert(0,str(BASE/'tree-primary4-top2-chain-compiler-cache-v368'))
import guard368 as parent
component=parent.component
sha=parent.sha
require=parent.require
EXPECTED_FREEZE=None
EXPECTED_ARCHIVE=None

def prepared():
    require(sha(HERE/'FREEZE370.json')==EXPECTED_FREEZE,'exact370 freeze')
    require(sha(HERE/'SOURCE370.tar.gz')==EXPECTED_ARCHIVE,'exact370 archive')
    own=json.loads((HERE/'FREEZE370.json').read_text())
    host=json.loads((HERE/'HOST370_RESULT.json').read_text())
    require(host['status'].startswith('PASS_HOST') and host['NPU_used'] is False and host['probe_sha256']==sha(HERE/'probe370.py'),'current370 HOST/source prerequisite')
    for filename,expected in own['files'].items():
        require(sha(filename)==expected,'immutable370 '+filename)
    inventory=dict(own['files'],**{str(HERE/'FREEZE370.json'):EXPECTED_FREEZE})
    with tarfile.open(HERE/'SOURCE370.tar.gz','r:gz') as saved:
        members=saved.getmembers()
        require({m.name for m in members}=={p.lstrip('/') for p in inventory} and len(members)==len(inventory),'exact370 inventory')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest()==inventory['/'+member.name],'exact370 archive bytes')
    pins=json.loads((parent.HERE/'SOURCE368_RECEIPT.json').read_text())
    parent.EXPECTED_FREEZE=pins['freeze_sha256']
    parent.EXPECTED_ARCHIVE=pins['archive_sha256']
    original,sources,readiness=parent.prepared()
    for filename in set(sources)&set(inventory):
        require(sources[filename]==inventory[filename],'identical370 dependency')
    sources.update(inventory)
    readiness.update(runtime370_freeze_sha256=EXPECTED_FREEZE,runtime370_archive_sha256=EXPECTED_ARCHIVE,original365_neural_files_unchanged=True,original_triton_device_initialization=True,host_precompiled_header_disabled=True,source_members=len(sources))
    return original,sources,readiness

def launch_argv(tag):
    argv=parent.launch_argv(tag)
    original=str(parent.HERE/'startup368.sh')
    require(argv.count(original)==1,'one original runtime launch')
    argv[argv.index(original)]=str(HERE/'startup370.sh')
    return argv


def storage(tag=None):
    helper=component.parent.donor_guard()
    actual=0 if tag is None else helper.bytes_under(HERE/'actual'/tag)
    total=helper.bytes_under(HERE)+helper.bytes_under(component.HERE)
    require(actual<=component.MAX_TAG_BYTES and total<=component.MAX_AGGREGATE_BYTES,'private370+365 bounded data')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)

def actual_audit(directory):
    from audit370 import audit
    result=audit(directory)
    require(isinstance(result,tuple) and len(result)==3,'original supervisor exact three-item audit ABI')
    return result

def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True)
    parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args()
    EXPECTED_FREEZE,EXPECTED_ARCHIVE=args.expected_freeze_sha256,args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None and all(re.fullmatch('[0-9a-f]{64}',p) for p in (EXPECTED_FREEZE,EXPECTED_ARCHIVE)),'safe explicit370 tag/pins')
    prepared()
    namespace=component.execution_namespace()
    namespace['prepared']=prepared
    namespace['launch_argv']=launch_argv
    namespace['P']=HERE
    namespace['G']=HERE
    namespace['storage']=storage
    namespace['actual_audit']=actual_audit
    namespace['SCOPE']=dict(models_loaded=['Qwen3-0.6B'],target_model_loaded=False,full463=False,formal_Goodput=False,adopted=False,first_catchup_plus_tail3=True,math_rows=99,timing_calls=320)
    lock=component.LOCK.open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()

if __name__=='__main__':
    main()
