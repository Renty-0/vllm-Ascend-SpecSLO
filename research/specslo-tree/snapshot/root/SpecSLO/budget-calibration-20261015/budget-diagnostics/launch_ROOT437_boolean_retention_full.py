"""Fresh all75+6/full463 run after an independently reproduced retention bug.

No prior engine permission, Native validation flag, result, or Git operation
is inherited. All original owned supervision and private limits remain.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-refill-witness-boolean-retention-full-v437'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='boolean-witness-retention-parent4-E4-global64-root-v437-fresh75-plus6-full463-ownAR-v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    terminal=json.loads((D/'ROOT436_TERMINAL_INDEPENDENT_ACTUAL_FAILURE.json').read_text())
    assert terminal['errors']==[] and terminal['metrics']is None and terminal['supervisor_status']=='failed'
    assert terminal['original_receipt_sha256']=='3baa7f94960380c3b8fab6de032dec7452df414f0946827efca433cfef019f1c'
    retained=json.loads((D/'ROOT436_COMPLETED_75_PLUS6_INDEPENDENT_COMPONENT_AUDIT.json').read_text())
    assert retained['status'].startswith('PASS_HOST436')and retained['NPU_used']is False
    assert retained['Raw']is retained['Goodput']is None and retained['new_engine_permission_inherited']is False
    bound={}
    for path in(P/'HOST437_RETENTION_RESULT.json',P/'HOST437_CONSTRUCTION_RESULT.json',
        D/'ROOT437_FROZEN_ACTUAL_TRACE_PRECONSTRUCTOR_HOST.json',D/'ROOT437_FROZEN_COLD_PARENT_RESULT.json',
        D/'ROOT437_TERMINAL_ADMISSION_HOST_RESULT.json'):
        row=json.loads(path.read_text());assert row['status'].startswith('PASS_HOST437')and row['NPU_used']is False
        bound[str(path)]=sha(path)
    regression=json.loads((P/'HOST437_RETENTION_RESULT.json').read_text())
    assert regression['actual_CPU_admission_to_saved_trace_cases']==regression['old_bug_reproduced']==4
    assert regression['controlled_missing_wrong_type_or_generation_negatives']==16
    assert json.loads((D/'ROOT437_TERMINAL_ADMISSION_HOST_RESULT.json').read_text())['terminal_auditor_sha256']==sha(D/'audit_ROOT437_terminal_full_HOST.py')
    record=json.loads((P/'SOURCE437_RECEIPT.json').read_text())
    assert record['freeze_sha256']=='8ecd9f97279d1ae6023e4ca14188954b34575b26940ec64d1848c8b93a99f8d8'
    assert record['archive_sha256']=='1553265a38fa19caf8bcb32067e9e921627f97f9c892429d68bd891e97aa366e'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for line in(P/'launch437.sh').read_text().splitlines():
        for value in shlex.split(line.rstrip(' \\')):
            if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
                key,setting=value.split('=',1);env[key]=setting;os.environ[key]=setting
    env['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
    os.environ['SPECSLO_V257_ROOT_FREEZE_SHA']=env['SPECSLO_V257_ROOT_FREEZE_SHA']
    line=next(l for l in(P/'startup437.sh').read_text().splitlines()if l.startswith('export PYTHONPATH='))
    path=line.removeprefix('export PYTHONPATH=');F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
    for old,new in(('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(DONOR)),('${PYTHONPATH:-}','')):path=path.replace(old,new)
    assert '$'not in path
    env['PYTHONPATH']=path;sys.path[:0]=path.split(':')
    import guard437
    guard437.EXPECTED_FREEZE=record['freeze_sha256'];guard437.EXPECTED_ARCHIVE=record['archive_sha256']
    guard437.prepared()
    storage=guard437.storage()
    assert storage['aggregate_private_bytes']+320*1024**2<=3584*1024**2,'Original256MiB tag plus64MiB source overhead must fit'
    argv=[sys.executable,'-B',str(P/'guard437.py'),'--tag',TAG,'--expected-freeze-sha256',record['freeze_sha256'],'--expected-archive-sha256',record['archive_sha256']]
    request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=argv,tag=TAG,
        source_freeze=record['freeze_sha256'],source_archive=record['archive_sha256'],HOST_prerequisites=bound,
        exact_same436_neural_callbacks=True,only_fix='Boolean type of post-generation witness-retention predicate',
        all_original227_681_75_plus6_warm_twozero_seal_full463_all3ownAR_fresh_required=True,
        prior_NN_permission_inherited=False,Native_flags_assigned=False,global_active_limit=64,
        prelaunch_private_storage=storage,private_tag_MiB=256,aggregate_MiB=3584,
        original_run_seconds=18000,new_Git_commits=0,new_Git_pushes=0,
        Git_only_after_confirmed_complete_effective_improvement=True)
    with(D/'ROOT437_FRESH_FULL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
    os.execve(sys.executable,argv,env)


if __name__=='__main__':main()
