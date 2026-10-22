"""Bounded original385 supervisor, plus current438 storage and source closure."""
import argparse,fcntl,hashlib,json,re,tarfile
from pathlib import Path
from host_env454 import setup
setup()
from check454 import P,B,ready,sha
import guard385 as parent
import guard453 as current
from projection425 import project_function
DONOR=parent.DONOR;LOCK=parent.LOCK
MAX_SECONDS=3600
MAX_TAG_BYTES=192*1024**2
MAX_AGGREGATE_BYTES=3584*1024**2
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None

def parent_pins():
    r=json.loads((parent.HERE/'SOURCE385_RECEIPT.json').read_text());parent.EXPECTED_FREEZE=r['freeze_sha256'];parent.EXPECTED_ARCHIVE=r['archive_sha256']
    r=json.loads((current.P/'SOURCE453_RECEIPT.json').read_text());current.EXPECTED_FREEZE=r['freeze_sha256'];current.EXPECTED_ARCHIVE=r['archive_sha256']

def prepared():
    parent_pins();own=ready(EXPECTED_FREEZE)
    _,old,_=parent.prepared();_,recent,_=current.prepared()
    for path,h in old.items():
        if path in recent and recent[path]!=h:raise RuntimeError('ROOT454 shared exact source closure')
    sources=dict(old,**recent);extra=dict(own['files']);extra[str(P/'FREEZE454.json')]=EXPECTED_FREEZE
    if set(extra)&set(sources):raise RuntimeError('ROOT454 disjoint additive own source')
    if sha(P/'SOURCE454.tar.gz')!=EXPECTED_ARCHIVE:raise RuntimeError('ROOT454 explicit archived source')
    with tarfile.open(P/'SOURCE454.tar.gz','r:gz')as t:
        members=t.getmembers()
        if len(members)!=len(extra)or {m.name for m in members}!={p.lstrip('/')for p in extra}:raise RuntimeError('ROOT454 complete source archive')
        for m in members:
            if not m.isfile()or hashlib.sha256(t.extractfile(m).read()).hexdigest()!=extra['/'+m.name]:raise RuntimeError('ROOT454 exact archive bytes')
    sources.update(extra)
    return own,sources,dict(source_members=len(sources),own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,NN_permission_inherited=False,operator_diagnostic_only=True)

def storage(tag=None):
    value=current.storage(tag)
    value['aggregate_private_bytes']+=parent.original_owner.donor_guard().bytes_under(P)
    if value['actual_tag_bytes']>MAX_TAG_BYTES or value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:raise RuntimeError('ROOT454 unchanged private192/3584 limits')
    return value

def launch_argv(tag):
    parent_pins();args=parent.launch_argv(tag);old=str(parent.HERE/'startup385.sh')
    if args.count(old)!=1:raise RuntimeError('ROOT454 one actual retained runtime startup')
    args=[str(P/'startup454.sh')if a==old else a for a in args]
    from host_env454 import source_environment
    pins=source_environment();pins.pop('PYTHONPATH');pins['SPECSLO_V454_ROOT_FREEZE_SHA']=EXPECTED_FREEZE
    # env accepts later duplicate assignments; same old385 value is preserved.
    first=args.index('/bin/bash');args[first:first]=[k+'='+v for k,v in sorted(pins.items())]
    return args

def actual_audit(directory):
    from audit454 import audit
    return audit(directory)

def execution_namespace():
    factory=project_function(parent.execution_namespace,(
        ('ROOT.V379.original-root-linear-target-supervisor.v1','ROOT.V454.actual-local-linear-layout-cost-supervisor.v1'),
        ('ROOT385 supervisor received','ROOT454 supervisor received'),
        ('real current M120 layer0 payloads; original128-epilogue TP3 graph qualifier and timings; no profile installed or full463 performance','genuine32 separately prefilled owners, physicalM160/S32/q5; layers0/31/63 attention/down observed in memory; unchanged original exact1024-change and31-sample p95>=1.02 operator qualifier; no model profile installed or serving qualification'),
        ("ACTION='real-M120-deferred-chain-original-graph-qualifier'","ACTION='M160-real32-native-epilogue-qualification'"),
    ),dict(HERE=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit,
        MAX_SECONDS=MAX_SECONDS,MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES))
    ns=factory()
    if ns['execute_locked'].__globals__ is not ns:raise RuntimeError('ROOT454 actual owned supervisor callbacks')
    return ns

def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--expected-freeze-sha256',required=True);p.add_argument('--expected-archive-sha256',required=True);a=p.parse_args()
    EXPECTED_FREEZE=a.expected_freeze_sha256;EXPECTED_ARCHIVE=a.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)or any(not re.fullmatch('[a-f0-9]{64}',x)for x in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):raise RuntimeError('ROOT454 safe tag and explicit pins')
    prepared();ns=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](a.tag)
if __name__=='__main__':main()
