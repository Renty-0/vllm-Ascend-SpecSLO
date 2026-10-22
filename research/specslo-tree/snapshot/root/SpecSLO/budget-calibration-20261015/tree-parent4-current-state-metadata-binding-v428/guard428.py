"""Exact425 owned supervision plus additive current-reference binding."""
import argparse,ast,copy,fcntl,hashlib,json,re,tarfile
from pathlib import Path
from check428 import P,B,PARENT_FREEZE,PARENT_ARCHIVE,ready,sha
import guard425 as parent

EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
require=parent.require
DONOR=parent.DONOR
ACTION=parent.ACTION
LOCK=parent.LOCK


def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,sources,previous=parent.prepared()
    own=ready(EXPECTED_FREEZE);extra=dict(own['files']);extra[str(P/'FREEZE428.json')]=EXPECTED_FREEZE
    require(not set(extra)&set(sources),'disjoint own428 source')
    require(sha(P/'SOURCE428.tar.gz')==EXPECTED_ARCHIVE,'exact additive428 archive')
    with tarfile.open(P/'SOURCE428.tar.gz','r:gz')as t:
        members=t.getmembers()
        require(len(members)==len(extra)and {m.name for m in members}=={p.lstrip('/')for p in extra},'exact428 archive inventory')
        for m in members:require(m.isfile()and hashlib.sha256(t.extractfile(m).read()).hexdigest()==extra['/'+m.name],'exact428 archived bytes')
    sources.update(extra)
    return own,sources,dict(parent425=previous,own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE,reference_selector_only=True,device_qualification_inherited=False)


def storage(tag=None):
    value=parent.storage(tag)
    value['aggregate_private_bytes']+=parent.donor_guard().bytes_under(P)
    require(value['aggregate_private_bytes']<=parent.MAX_AGGREGATE_BYTES,'unchanged aggregate private cap')
    return value


def launch_argv(tag):
    args=parent.launch_argv(tag);old=str(parent.HERE/'launch425.sh')
    require(args.count(old)==1,'one actual425 launch entry')
    return [str(P/'launch428.sh')if v==old else v for v in args]


def actual_audit(directory):
    from projection425 import project_function
    actual=project_function(parent.actual_audit,(
        ('from audit425 import audit','from audit428 import audit'),))
    return actual(directory)


def execution_namespace():
    import inspect
    from projection425 import project_function
    factory=project_function(parent.execution_namespace,(
        ('ROOT.V425.best257-uniform-KV-full-own-AR-supervisor.v1',
         'ROOT.V428.best257-uniform-KV-full-own-AR-supervisor.v1'),
        ('ROOT425 supervisor received','ROOT428 supervisor received'),
    ),dict(HERE=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit))
    namespace=factory()
    namespace['SCOPE']=dict(namespace['SCOPE'],wrapper428_reference_selector_correction_only=True,
        current_backend_state_field='uniform_KV_bucket425_enabled',production_implementation425_unchanged=True)
    require(namespace['execute_locked'].__globals__ is namespace,'actual428 owned callbacks')
    return namespace


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args();EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None and all(re.fullmatch('[a-f0-9]{64}',v)is not None for v in (EXPECTED_FREEZE,EXPECTED_ARCHIVE)),'safe428 tag and pins')
    prepared();namespace=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);namespace['execute_locked'](args.tag)


if __name__=='__main__':main()
