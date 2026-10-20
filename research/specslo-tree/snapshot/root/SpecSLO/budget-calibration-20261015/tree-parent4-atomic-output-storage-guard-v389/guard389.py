"""Retain original runtime; tolerate only vanished .tmp in its actual output."""
import argparse
import fcntl
from pathlib import Path
import re
import sys

HERE=Path(__file__).parent;B=HERE.parent
sys.path.insert(0,str(B/'tree-parent4-owned-exit-path-v388'))
import guard388 as parent
sys.path.insert(0,str(B/'budget-diagnostics/v177-root-safe-storage-overlay'))
import safe_storage177
PARENT_FREEZE='1d660aacce3653ea25492c29e91cebd8dbf5862b5a81af1f09c3473c5b4557a9'
PARENT_ARCHIVE='47e475f190079d25eb03b8110a70a8b93dea51bade40065ae72d93497e86a777'
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
DONOR=parent.parent.original.DONOR
require=parent.parent.original.require


def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,files,previous=parent.prepared()
    own,extra=parent.parent.checked_archive(HERE,389,EXPECTED_FREEZE,EXPECTED_ARCHIVE)
    require(own['parent388_freeze_sha256']==PARENT_FREEZE and own['parent388_archive_sha256']==PARENT_ARCHIVE,'bound original388 sources')
    require(not set(files)&set(extra),'disjoint389 source guard')
    files.update(extra)
    return own,files,dict(parent=previous,source_members=len(files),own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE,device_qualification_inherited=False,
        runtime_changed=False,storage_delta='existing177 .tmp race rule, exact current actual directory')


def bytes_under(path):
    return safe_storage177.bytes_under(path,live_roots=(DONOR/'actual',),require=require)


def storage(tag=None):
    original=parent.parent.original
    actual=0 if tag is None else bytes_under(DONOR/'actual'/tag)
    directories=(DONOR,original.HERE,parent.parent.HERE,parent.HERE,HERE)
    require(len(set(directories))==len(directories),'unique complete bounded storage roots')
    total=sum(bytes_under(p)for p in directories)
    require(actual<=original.MAX_TAG_BYTES and total<=original.MAX_AGGREGATE_BYTES,'unchanged private bounded storage limits')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)


def execution_namespace():
    namespace=parent.execution_namespace()
    namespace.update(prepared=prepared,storage=storage,G=HERE)
    require(namespace['execute_locked'].__globals__ is namespace,'actual storage callback namespace')
    return namespace


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args();EXPECTED_FREEZE,EXPECTED_ARCHIVE=args.expected_freeze_sha256,args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None and all(re.fullmatch('[0-9a-f]{64}',x)for x in [EXPECTED_FREEZE,EXPECTED_ARCHIVE]),'explicit safe389 pins/tag')
    prepared();namespace=execution_namespace()
    with parent.parent.original.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);namespace['execute_locked'](args.tag)


if __name__=='__main__':main()
