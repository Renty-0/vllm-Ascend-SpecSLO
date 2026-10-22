"""Exact frozen423 full guard after owned424 cleanup; no other process control."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-parent4-selective-full-integration-v423'
D=B/'budget-diagnostics'
TAG='selective-parent4-E4-global64-root-v423-current75-full463-ownAR-v3'
old_tag='parent4-child4-N24-SLO-mixed-root-v424-complete75-v1'
old_path=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'/f'receipt-{old_tag}.json'
old=json.loads(old_path.read_text())
assert old['status']=='failed' and old['sources_unchanged'] and old['owned_remaining']==[] and old['devices_after']['all_idle']
assert (D/'ROOT424_INTENTIONAL_CONSOLIDATION_TO423_REQUEST.json').is_file()
terminal=json.loads((D/'ROOT424_INTENTIONAL_CONSOLIDATION_TERMINAL_AUDIT.json').read_text())
assert terminal['status']=='terminal_incomplete_finite_evidence_retained'
assert terminal['receipt_sha256']==hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ('ROOT423_REAL_PRECONSTRUCTOR_RESULT.json','ROOT423_FROZEN_COLD_PARENT_RESULT.json'):
 host=json.loads((D/name).read_text());assert host['status'].startswith('PASS_HOST423') and host['NPU_used'] is False
record=json.loads((P/'SOURCE423_RECEIPT.json').read_text())
assert record['freeze_sha256']=='ebf13f6899ee749c2ed7c7132f097c692223e246a20cd4c5c7cfad5596861749'
assert record['archive_sha256']=='a46c14e2d6f2bb46682e951349ae3e5c35e2955dc1b5ff533814349b79531725'
env=dict(os.environ)
env.update(dict(HOME='/root',USER='root',LOGNAME='root',SHELL='/bin/bash',LANG='C.UTF-8',PATH='/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin',PYTHONDONTWRITEBYTECODE='1'))
for line in (P/'launch423.sh').read_text().splitlines():
 for value in shlex.split(line.rstrip(' \\')):
  if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
   key,setting=value.split('=',1);env[key]=setting
env['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
import subprocess
probe=subprocess.run(['npu-smi','info'],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
assert probe.returncode==0, 'existing driver runtime required: '+probe.stdout[-1000:]
args=[sys.executable,'-B',str(P/'guard423.py'),'--tag',TAG,'--expected-freeze-sha256',record['freeze_sha256'],'--expected-archive-sha256',record['archive_sha256']]
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=args,
 source_freeze=record['freeze_sha256'],source_archive=record['archive_sha256'],source_auth_environment={k:v for k,v in env.items() if k.startswith('SPECSLO_') or k=='PATH'},
 parent_existing_driver_runtime_environment_inherited=True,
 original_global64_and_private_caps_retained=True,all_original_and227_and681_and75_current_cases_required=True,
 warm_twozero_seal_full463_all3ownAR_required=True,NN_permission_inherited=False,github_commit_requested=False)
with (D/'ROOT423_FULL_CURRENT75_LAUNCH_REQUEST_V3.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
os.execve(sys.executable,args,env)
