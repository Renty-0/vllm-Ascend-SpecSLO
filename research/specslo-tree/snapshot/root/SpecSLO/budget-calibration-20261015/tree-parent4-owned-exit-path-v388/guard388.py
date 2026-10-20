"""Complete386 source supervisor with an explicit owned-exit dependency path."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).parent;B=HERE.parent
PARENT=B/'tree-parent4-full-archive-guard-v387'
sys.path.insert(0,str(PARENT))
import guard387 as parent
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
PARENT_FREEZE='b15f3876538e5f1a04e2d8c16950e16e264034205783f45d4d08e6dcad3be1a4'
PARENT_ARCHIVE='9c61ef0e3efe3f9bedb981614b89d42ec7167792d3d78d766fe74e86052fe329'


def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,files,old=parent.prepared()
    own,extra=parent.checked_archive(HERE,388,EXPECTED_FREEZE,EXPECTED_ARCHIVE)
    parent.original.require(own['parent387_freeze_sha256']==PARENT_FREEZE and own['parent387_archive_sha256']==PARENT_ARCHIVE,'bound supervisor387 sources')
    parent.original.require(not set(files)&set(extra),'disjoint388 guarded dependency source')
    files.update(extra)
    return own,files,dict(own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,
        parent_supervisor=old,source_members=len(files),device_qualification_inherited=False,
        runtime_delta='add existing shutdown234 path to parent and inherited spawn sys.path only')


def launch_argv(tag):
    parent.original.EXPECTED_FREEZE=parent.BASE_FREEZE
    args=parent.original.launch_argv(tag)
    before=str(parent.BASE/'startup386.sh');after=str(HERE/'startup388.sh')
    if args.count(before)!=1:raise RuntimeError('ROOT388 exact original startup path')
    return [after if a==before else a for a in args]


def execution_namespace():
    namespace=parent.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,G=HERE)
    scope=dict(namespace['SCOPE'])
    scope.pop('real_draft_generation',None)
    scope.update(real_draft_generation_requested=True,fresh_original173_baseline_required=True,
                 bounded_parent4_child4_protocol_requested=True,production_permission=False)
    namespace['SCOPE']=scope
    if namespace['execute_locked'].__globals__ is not namespace:raise RuntimeError('ROOT388 actual supervisor namespace')
    return namespace


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    a=parser.parse_args();EXPECTED_FREEZE,EXPECTED_ARCHIVE=a.expected_freeze_sha256,a.expected_archive_sha256
    parent.original.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)is not None
        and all(re.fullmatch('[0-9a-f]{64}',x)for x in [EXPECTED_FREEZE,EXPECTED_ARCHIVE]),'safe388 tag and pins')
    prepared();ns=execution_namespace()
    with parent.original.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](a.tag)


if __name__=='__main__':main()
