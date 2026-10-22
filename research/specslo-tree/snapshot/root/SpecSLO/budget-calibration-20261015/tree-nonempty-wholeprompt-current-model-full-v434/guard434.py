"""Conditional future434, preserving original425 supervision and private caps."""
import argparse,fcntl,hashlib,re,tarfile
from check434 import P,B,COMPOSITION,COMPOSITION_FREEZE,COMPOSITION_ARCHIVE,PARENT_FREEZE,PARENT_ARCHIVE,ready,sha
import guard433 as parent
import guard425 as base

EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
require=base.require
DONOR,ACTION,LOCK=base.DONOR,base.ACTION,base.LOCK


def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,sources,previous=parent.prepared()
    import check432
    for folder,(number,freeze,archive)in check432.PARENTS.items():
        if number==433:continue
        extra=check432.checked_archive(B/folder,number,freeze,archive)
        require(not set(extra)&set(sources),'disjoint frozen HOST support source '+folder)
        sources.update(extra)
    extra=check432.checked_archive(COMPOSITION,432,COMPOSITION_FREEZE,COMPOSITION_ARCHIVE)
    require(not set(extra)&set(sources),'disjoint frozen432 full composition source')
    sources.update(extra)
    own=ready(EXPECTED_FREEZE);extra=dict(own['files']);extra[str(P/'FREEZE434.json')]=EXPECTED_FREEZE
    require(not set(extra)&set(sources),'disjoint own434 source')
    require(sha(P/'SOURCE434.tar.gz')==EXPECTED_ARCHIVE,'exact own434 archive')
    with tarfile.open(P/'SOURCE434.tar.gz','r:gz')as saved:
        members=saved.getmembers()
        require(len(members)==len(extra)and {m.name for m in members}=={p.lstrip('/')for p in extra},'exact434 source inventory')
        for m in members:require(m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==extra['/'+m.name],'exact434 archived bytes')
    sources.update(extra)
    return own,sources,dict(parent433=previous,own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE,frozen432_implementation=True,
        new_nonempty_refill_requires_current427_NN=True,device_qualification_inherited=False)


def storage(tag=None):
    value=parent.storage(tag)
    import check432
    directories={P,COMPOSITION,*[B/folder for folder,(n,_,_)in check432.PARENTS.items()if n!=433]}
    value['aggregate_private_bytes']+=sum(base.donor_guard().bytes_under(p)for p in directories)
    require(value['aggregate_private_bytes']<=base.MAX_AGGREGATE_BYTES,'unchanged aggregate private cap')
    return value


def launch_argv(tag):
    args=parent.launch_argv(tag);old=str(parent.P/'launch433.sh')
    require(args.count(old)==1,'one original433 launch entry')
    return[str(P/'launch434.sh')if v==old else v for v in args]


def actual_audit(directory):
    from projection425 import project_function
    actual=project_function(base.actual_audit,(('from audit425 import audit','from audit434 import audit'),))
    return actual(directory)


def execution_namespace():
    from projection425 import project_function
    factory=project_function(base.execution_namespace,(
        ('ROOT.V425.best257-uniform-KV-full-own-AR-supervisor.v1','ROOT.V434.best257-uniform-KV-full-own-AR-supervisor.v1'),
        ('ROOT425 supervisor received','ROOT434 supervisor received'),
    ),dict(HERE=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit))
    ns=factory()
    ns['SCOPE']=dict(ns['SCOPE'],frozen432_wholeprompt_nonempty_refill=True,
        original_current427_extra_NN_cases=6,real_incumbents=32,
        original_incumbent_time_charge_preserved=True,global_active_limit=64)
    require(ns['execute_locked'].__globals__ is ns,'actual434 owned callbacks')
    return ns


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args();EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None
        and all(re.fullmatch('[a-f0-9]{64}',v)is not None for v in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)),'safe434 tag/pins')
    prepared();ns=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](args.tag)


if __name__=='__main__':main()
