"""Original owned supervisor, full parent source rechecked once per traversal."""
import argparse,fcntl,hashlib,json,re,tarfile
from pathlib import Path
from host_env457 import setup,source_environment
setup()
from check457 import P,B,ready,sha
from flat457 import verified
import guard385 as parent
import guard455 as storage_parent
from projection425 import project_function
DONOR=parent.DONOR;LOCK=parent.LOCK
MAX_SECONDS=3600;MAX_TAG_BYTES=192*1024**2;MAX_AGGREGATE_BYTES=3584*1024**2
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
ACTION='M80-own-geometry-whole-model-reference-cost'
def parent_pins():
 r=json.loads((parent.HERE/'SOURCE385_RECEIPT.json').read_text());parent.EXPECTED_FREEZE=r['freeze_sha256'];parent.EXPECTED_ARCHIVE=r['archive_sha256']
def prepared():
 own,files=ready(EXPECTED_FREEZE,include_parent=True);extra=dict(own['files']);extra[str(P/'FREEZE457.json')]=EXPECTED_FREEZE
 if sha(P/'SOURCE457.tar.gz')!=EXPECTED_ARCHIVE:raise RuntimeError('ROOT457 exact new source archive')
 with tarfile.open(P/'SOURCE457.tar.gz','r:gz')as archive:
  members=archive.getmembers()
  if len(members)!=len(extra)or {m.name for m in members}!={p.lstrip('/')for p in extra}:raise RuntimeError('ROOT457 complete current source archive')
  for m in members:
   if not m.isfile()or hashlib.sha256(archive.extractfile(m).read()).hexdigest()!=extra['/'+m.name]:raise RuntimeError('ROOT457 archived exact bytes')
 for path,h in extra.items():
  if path in files and files[path]!=h:raise RuntimeError('ROOT457 exact shared closure')
  files[path]=h
 return own,files,dict(source_members=len(files),own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,
  original455_entire_source_and_all_archives_reverified=True,NN_permission_inherited=False,declared_neural_cases=1)
def storage(tag=None):
 value=storage_parent.storage(tag);value['aggregate_private_bytes']+=parent.original_owner.donor_guard().bytes_under(P)
 if value['actual_tag_bytes']>MAX_TAG_BYTES or value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:raise RuntimeError('ROOT457 same private limits192/3584MiB')
 return value
def launch_argv(tag):
 parent_pins();args=parent.launch_argv(tag);old=str(parent.HERE/'startup385.sh')
 if args.count(old)!=1:raise RuntimeError('ROOT457 one actual startup')
 args=[str(P/'startup457.sh')if v==old else v for v in args]
 pins=source_environment();pins.pop('PYTHONPATH');pins['SPECSLO_V457_ROOT_FREEZE_SHA']=EXPECTED_FREEZE
 first=args.index('/bin/bash');args[first:first]=[k+'='+v for k,v in sorted(pins.items())];return args
def actual_audit(directory):
 from audit457 import audit
 return audit(directory)
def execution_namespace():
 factory=project_function(parent.execution_namespace,(
  ('ROOT.V379.original-root-linear-target-supervisor.v1','ROOT.V457.own80-whole-model-supervisor.v1'),
  ('ROOT385 supervisor received','ROOT457 supervisor received'),
  ('real current M120 layer0 payloads; original128-epilogue TP3 graph qualifier and timings; no profile installed or full463 performance',
   'true M80/S16 using unchanged original profile versus independent ordinary M80 full-model computation and original M120; explicit cross-M120 observations retained; all128 query KV and post-timing checks; no new serving or AR permission'),
  ("ACTION='real-M120-deferred-chain-original-graph-qualifier'","ACTION='M80-own-geometry-whole-model-reference-cost'"),
 ),dict(HERE=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit,
  MAX_SECONDS=MAX_SECONDS,MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES))
 ns=factory();ns['SCOPE'].update(finite_cases=1,independent_prefixes=72)
 if ns['execute_locked'].__globals__ is not ns:raise RuntimeError('ROOT457 actual owned callbacks')
 return ns
def main():
 global EXPECTED_FREEZE,EXPECTED_ARCHIVE
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--expected-freeze-sha256',required=True);p.add_argument('--expected-archive-sha256',required=True);a=p.parse_args()
 EXPECTED_FREEZE=a.expected_freeze_sha256;EXPECTED_ARCHIVE=a.expected_archive_sha256
 if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)or any(not re.fullmatch('[a-f0-9]{64}',v)for v in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):raise RuntimeError('ROOT457 safe tag and exact source pins')
 prepared();ns=execution_namespace()
 with LOCK.open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](a.tag)
if __name__=='__main__':main()
