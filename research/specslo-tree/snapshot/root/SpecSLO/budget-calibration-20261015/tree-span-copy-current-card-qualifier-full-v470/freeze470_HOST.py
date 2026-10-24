"""Freeze new full candidate only while all owned devices and prior work are idle."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import sys
import tarfile

P=Path(__file__).resolve().parent
B=P.parent
R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
PRIOR=R/'receipt-span-KV-copy-cohort48-current0127-root-v469-full463-ownAR-v1.json'
COPY_PROOF=R/'receipt-parallel-common-root-fork-current0127-span-copy-cost-v1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with (B/'.ROOT_NPU_EXECUTION.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads(PRIOR.read_text())
        if (previous['status']!='failed'or previous.get('sources_unchanged')is not True
                or previous.get('owned_remaining')!=[]or previous['devices_after']['all_idle']is not True):
            raise RuntimeError('Actual prior run and owned cleanup must end first')
        proof=json.loads(COPY_PROOF.read_text())
        if proof['status']!='complete' or proof['sources_unchanged'] is not True:
            raise RuntimeError('Original copy numerical proof remains required')
        import check470,guard470
        from check470 import PARENT_FREEZE,PARENT_ARCHIVE
        if guard470.execution_namespace()['original_helpers']()['helper']().device_state()['all_idle']is not True:
            raise RuntimeError('Actual current0127 devices must be idle')
        host=json.loads((P/'HOST470_RESULT.json').read_text())
        if (host['status']!='PASS_HOST470_ORIGINAL_QUALIFIERS_AND_DEEPEST_TERMINAL_ABI'
                or host['terminal_interfaces']['actual_deepest_current_import_called']is not True
                or host['actual_current_card_actor']['positive'] != 4 or host['actual_current_card_actor']['negative'] != 20
                or host['NPU_used']is not False or host['NN_permission']is not False):
            raise RuntimeError('Complete HOST integration check required')
        for path,wanted in host['source_sha256'].items():
            if sha(path)!=wanted:raise RuntimeError('Changed HOST-checked source: '+path)
        paths=[f for f in P.iterdir()if f.is_file()and(f.suffix in('.py','.sh')or f.name in('PLAN470.json','HOST470_RESULT.json'))]
        paths.extend((PRIOR,COPY_PROOF,B/'budget-diagnostics/ROOT_BRANCH_SPAN_COPY_CONCURRENT_COST_COMPLETE_HOST.json',
                      B/'budget-diagnostics/ROOT466_FINAL_COMMON_CLOCK_HOST.json'))
        files={str(f):sha(f)for f in sorted(paths)}
        kernel_root=B/'tree-branch-KV-span-copy-HOST-next'
        for path in kernel_root.iterdir():
            if path.is_file() and path.suffix in('.py','.sh','.json'):
                files[str(path)]=sha(path)
        frozen=dict(schema='ROOT470.target-span-copy-full.freeze.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            files=files,parent466_freeze=PARENT_FREEZE,parent466_archive=PARENT_ARCHIVE,
            physical_cards=[0,1,2,7],global_active_limit=64,maximum_new_cohort=48,
            maximum_empty_prefill_cohort=48,maximum_nonempty_prefill_cohort=32,
            target_copy_programs=64,original_model_weights_admission_and_clocks_unchanged=True,
            original_full_numerical_and_AR_checks_retained=True,NN_permission_inherited=False)
        path=P/'FREEZE470.json'
        with path.open('x')as stream:json.dump(frozen,stream,indent=2);stream.write('\n')
        members=dict(files);members[str(path)]=sha(path)
        archive=P/'SOURCE470.tar.gz'
        with tarfile.open(archive,'x:gz')as stream:
            for name in sorted(members):stream.add(name,arcname=name.lstrip('/'),recursive=False)
        receipt=dict(status='HOST470_SOURCE_FROZEN_NO_NN_PERMISSION',source_freeze_SHA256=sha(path),
            source_archive_SHA256=sha(archive),own_source_members=len(members),NPU_model_launched=False,NN_permission_inherited=False)
        with (P/'SOURCE470_RECEIPT.json').open('x')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
        print(json.dumps(receipt))


if __name__=='__main__':main()
