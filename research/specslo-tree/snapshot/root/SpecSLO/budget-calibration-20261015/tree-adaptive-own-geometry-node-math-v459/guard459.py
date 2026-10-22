"""Fresh own-physical node diagnostic, original owned supervisor and cleanup."""
from pathlib import Path
import argparse,fcntl,hashlib,json,re,tarfile
from host_env459 import setup,source_environment
setup()
from check459 import P,B,ready,sha
import guard385 as owner
import guard457 as parent
from projection425 import project_function
DONOR=owner.DONOR;LOCK=owner.LOCK
MAX_SECONDS=3600;MAX_TAG_BYTES=192*1024**2;MAX_AGGREGATE_BYTES=3584*1024**2
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
ACTION='adaptive-own-physical80-120-693-node-math-and-cost'
def parent_pins():
    r=json.loads((owner.HERE/'SOURCE385_RECEIPT.json').read_text())
    owner.EXPECTED_FREEZE=r['freeze_sha256'];owner.EXPECTED_ARCHIVE=r['archive_sha256']
    r=json.loads((parent.P/'SOURCE457_RECEIPT.json').read_text())
    parent.EXPECTED_FREEZE=r['freeze_sha256'];parent.EXPECTED_ARCHIVE=r['archive_sha256']
def prepared():
    parent_pins();own=ready(EXPECTED_FREEZE);_,files,_=parent.prepared()
    extra=dict(own['files']);extra[str(P/'FREEZE459.json')]=EXPECTED_FREEZE
    if sha(P/'SOURCE459.tar.gz')!=EXPECTED_ARCHIVE:raise RuntimeError('ROOT459 exact current archive')
    with tarfile.open(P/'SOURCE459.tar.gz','r:gz')as archive:
        members=archive.getmembers()
        if len(members)!=len(extra)or {m.name for m in members}!={p.lstrip('/')for p in extra}:
            raise RuntimeError('ROOT459 complete current source archive inventory')
        for m in members:
            if not m.isfile()or hashlib.sha256(archive.extractfile(m).read()).hexdigest()!=extra['/'+m.name]:
                raise RuntimeError('ROOT459 archived exact source bytes')
    for path,h in extra.items():
        if path in files and files[path]!=h:raise RuntimeError('ROOT459 shared identical source closure')
        files[path]=h
    return own,files,dict(source_members=len(files),own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,
                         NN_permission_inherited=False,own_physical_node_cases=693,bank_prefills_per_target=216)
def storage(tag=None):
    value=parent.storage(tag);value['aggregate_private_bytes']+=owner.original_owner.donor_guard().bytes_under(P)
    if value['actual_tag_bytes']>MAX_TAG_BYTES or value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:
        raise RuntimeError('ROOT459 unchanged private192/3584MiB bounds')
    return value
def launch_argv(tag):
    parent_pins();args=owner.launch_argv(tag);old=str(owner.HERE/'startup385.sh')
    if args.count(old)!=1:raise RuntimeError('ROOT459 one retained runtime startup')
    args=[str(P/'startup459.sh')if v==old else v for v in args]
    pins=source_environment();pins.pop('PYTHONPATH');pins['SPECSLO_V459_ROOT_FREEZE_SHA']=EXPECTED_FREEZE
    first=args.index('/bin/bash');args[first:first]=[k+'='+v for k,v in sorted(pins.items())];return args
def actual_audit(directory):
    from audit459 import audit
    return audit(directory)
def execution_namespace():
    factory=project_function(owner.execution_namespace,(
        ('ROOT.V379.original-root-linear-target-supervisor.v1','ROOT.V459.fresh693-own-physical-node-supervisor.v1'),
        ('ROOT385 supervisor received','ROOT459 supervisor received'),
        ('real current M120 layer0 payloads; original128-epilogue TP3 graph qualifier and timings; no profile installed or full463 performance',
         'fresh693 own-physical M80/M120 independent spine/side hidden/token/all128KV cases over3 prefix banks, original raw Native graph checks and all6x20 paired target costs; old cross-M120 failures retained; no serving score or own-AR permission'),
        ("ACTION='real-M120-deferred-chain-original-graph-qualifier'","ACTION='adaptive-own-physical80-120-693-node-math-and-cost'"),
    ),dict(HERE=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit,
           MAX_SECONDS=MAX_SECONDS,MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES))
    ns=factory();ns['SCOPE'].update(finite_cases=693,independent_prefixes=216)
    if ns['execute_locked'].__globals__ is not ns:raise RuntimeError('ROOT459 actual original owned callbacks')
    return ns
def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True);args=parser.parse_args()
    EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)or any(not re.fullmatch('[a-f0-9]{64}',v)for v in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):
        raise RuntimeError('ROOT459 safe tag and explicit source pins')
    prepared();ns=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](args.tag)
if __name__=='__main__':main()
