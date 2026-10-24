"""Freeze the actual coupled experiment after the preceding owned run exits."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import tarfile

P=Path(__file__).resolve().parent
B=P.parent
PRIOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-parallel-common-root-fork-current0127-concurrent-cost-v1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with(B/'.ROOT_NPU_EXECUTION.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads(PRIOR.read_text())
        if previous['status']!='complete' or previous.get('sources_unchanged')is not True or previous.get('devices_after',{}).get('all_idle')is not True:
            raise RuntimeError('Actual target proof and owned cleanup must complete first')
        from host_env_span import setup
        setup()
        from guard_span import execution_namespace
        if execution_namespace()['original_helpers']()['helper']().device_state()['all_idle']is not True:
            raise RuntimeError('All current cards must be idle')
        result=json.loads((P/'HOST_SPAN_RESULT.json').read_text())
        if result['status']!='PASS_HOST_SPAN_COPY_MAPPING_AND_ORIGINAL_COUPLED_LIFECYCLE' or result['NPU_used']is not False:
            raise RuntimeError('Actual original CPU methods and declared bindings required')
        for path,wanted in result['source_sha256'].items():
            if sha(path)!=wanted:raise RuntimeError('HOST-checked code changed: '+path)
        files={str(p):sha(p)for p in P.iterdir()if p.is_file()and p.suffix in('.py','.sh','.json')and not p.name.startswith(('FREEZE_','SOURCE_'))}
        for folder in ('tree-parallel-root-fork-overlap-cost-HOST-next',):
            parent=B/folder
            frozen=json.loads((parent/'FREEZE_PROBE.json').read_text())
            for path,wanted in frozen['files'].items():
                if sha(path)!=wanted or path in files and files[path]!=wanted:
                    raise RuntimeError('Unchanged actual producer and target source closure required: '+path)
                files[path]=wanted
            for name in ('FREEZE_PROBE.json','SOURCE_PROBE.tar.gz','SOURCE_PROBE_RECEIPT.json'):
                path=parent/name;files[str(path)]=sha(path)
        files[str(PRIOR)]=sha(PRIOR)
        pins=json.loads((B/'tree-M80-own-geometry-whole-model-v457/SOURCE457_RECEIPT.json').read_text())
        value=dict(schema='ROOT.parallel-fork-span-copy-concurrent-cost.freeze.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
            parent_freeze_sha256=pins['freeze_sha256'],parent_archive_sha256=pins['archive_sha256'],
            physical_cards=[0,1,2,7],global_active_ceiling=64,real_draft_cases=21,target_passes=42,timed_concurrent_calls_each_rank=840,postcost_node_checks=14,
            NPU_launched=False,NN_permission=False,
            scope='New raw16 span copy precheck; original21 coupled proof plus840 concurrent API timing calls and14 postcost independent target-node checks; no fullserving Goodput')
        freeze=P/'FREEZE_PROBE.json'
        with freeze.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
        members=dict(files);members[str(freeze)]=sha(freeze)
        archive=P/'SOURCE_PROBE.tar.gz'
        with tarfile.open(archive,'x:gz')as stream:
            for path in sorted(members):stream.add(path,arcname=path.lstrip('/'),recursive=False)
        receipt=dict(freeze_sha256=sha(freeze),archive_sha256=sha(archive),own_members=len(members),NPU_launched=False,NN_permission=False)
        with(P/'SOURCE_PROBE_RECEIPT.json').open('x')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
        print(json.dumps(receipt))


if __name__=='__main__':main()
