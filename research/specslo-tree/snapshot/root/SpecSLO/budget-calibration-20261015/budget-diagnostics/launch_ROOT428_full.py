"""Launch frozen425 after actual423 cleanup and all frozen HOST prerequisites."""
import datetime,hashlib,json,os,re,shlex,subprocess,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-parent4-current-state-metadata-binding-v428'
TAG='reference-selector-parent4-E4-global64-root-v428-current75-full463-ownAR-v1'
old=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-canonical-tail-binding-parent4-E4-global64-root-v425-current75-full463-ownAR-v1.json'
prior=json.loads(old.read_text())
assert prior['status']=='failed' and prior['sources_unchanged'] and prior['owned_remaining']==[] and prior['devices_after']['all_idle']
terminal=json.loads((D/'ROOT425_TERMINAL_CURRENT_STATE_FIELD_FAILURE_AUDIT.json').read_text())
assert terminal['original_receipt_sha256']==hashlib.sha256(old.read_bytes()).hexdigest()
for name in ('ROOT428_REAL_PRECONSTRUCTOR_RESULT.json','ROOT428_FROZEN_COLD_PARENT_RESULT.json'):
 host=json.loads((D/name).read_text());assert host['status'].startswith('PASS_HOST428') and host['NPU_used'] is False
record=json.loads((P/'SOURCE428_RECEIPT.json').read_text())
assert record['freeze_sha256']=='d484d2cdb3387abb487c2a44ce865571ebb8d3047aeef2c77965fb866b7d0a4a'
assert record['archive_sha256']=='8e145c79e496d096faa598bc59dd30bf70521a10c34a73c51f3e0cc186041fd1'
env=dict(os.environ)
env.update(HOME='/root',USER='root',LOGNAME='root',SHELL='/bin/bash',LANG='C.UTF-8',PATH='/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin',PYTHONDONTWRITEBYTECODE='1')
for line in (P/'launch428.sh').read_text().splitlines():
 for value in shlex.split(line.rstrip(' \\')):
  if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
   key,setting=value.split('=',1);env[key]=setting
env['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
probe=subprocess.run(['npu-smi','info'],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
assert probe.returncode==0,'actual driver tool runtime needed: '+probe.stdout[-1000:]
args=[sys.executable,'-B',str(P/'guard428.py'),'--tag',TAG,'--expected-freeze-sha256',record['freeze_sha256'],'--expected-archive-sha256',record['archive_sha256']]
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=args,
 source_freeze=record['freeze_sha256'],source_archive=record['archive_sha256'],
 source_auth_environment={k:v for k,v in env.items() if k.startswith('SPECSLO_') or k=='PATH'},
 parent_existing_driver_runtime_inherited=True,child_existing_env_i_and_runtime_sh_unchanged=True,
 all_original_components_and227_and681_and75_required=True,
 warm_twozero_seal_full463_all3ownAR_required=True,global_active_limit=64,
 private_tag_MiB=256,aggregate_MiB=3584,NN_permission_inherited=False,github_commit_requested=False)
with(D/'ROOT428_FULL_CURRENT75_LAUNCH_REQUEST.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
os.execve(sys.executable,args,env)
