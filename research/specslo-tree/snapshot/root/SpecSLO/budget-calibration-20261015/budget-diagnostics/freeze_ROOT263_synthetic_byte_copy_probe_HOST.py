"""Freeze a synthetic copy-only probe; no backend or device accesses."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-fused-byte-branch-KV-copy-v263'
OLD=B/'tree-fused-branch-KV-row-copy-v262/FREEZE262.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(P/'FREEZE263.json').exists()and not(P/'SOURCE263.tar.gz').exists()
    assert sha(OLD)=='4ffc9f1205e060fa8be48e5e7acae6b47d84b04b494fdab4a0f3098de1a87ca8'
    evidence=json.loads((B/'tree-fused-branch-KV-row-copy-v262/HOST262_ALL227_NATIVE_COPY_MAPS_RESULT.json').read_text())
    assert evidence['cases']==227 and evidence['branch_cases']==172 and evidence['actual_NPU_used']is False
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT263.synthetic-copy-probe-review.v1',
        existing_native_environment_only=True,existing_Triton_and_TritonAscend=True,
        scope='synthetic K/V copy only, no real weights/attention/decoding/Goodput',
        actual172_maps_each_width256_384=True,old4ops_and_new1op_ACLGraph=True,
        all65536_raw16_bit_payloads_changed_restored_graph_replays=True,
        HOST_map_checks_before_device_index_upload=True,Triton_warm_outside_graph_capture=True,
        scalar_address_uint8_copy_no_BF16_math_or_Torch_device_dtype_view=True,
        private_Triton_cache_only=True,original_source_verifier_and_ROOT_lock_and_all4idle_required=True,
        neural_qualified=False,performance_qualified=False,production_installed=False)
    (P/'REVIEW263.json').write_text(json.dumps(review,indent=2)+'\n')
    files=dict(json.loads(OLD.read_text())['files']);files[str(OLD)]=sha(OLD)
    extra=[*P.glob('*'),Path(__file__),D/'ROOT262_FAILED_OPERATOR_COMPILE_CLASSIFICATION.json',Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/ops/triton/batch_memcpy.py')]
    for p in extra:
        if p.is_file():
            if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
            files[str(p)]=sha(p)
    for path,wanted in files.items():assert sha(path)==wanted,path
    freeze=P/'FREEZE263.json'
    with freeze.open('x')as f:json.dump(dict(schema='ROOT263.synthetic-copy-only-source-freeze.v1',UTC=review['UTC'],files=files,
        actual_NPU_executed=False,neural_qualified=False,production_installed=False),f,indent=2);f.write('\n')
    members={**files,str(freeze):sha(freeze)}
    with tarfile.open(P/'SOURCE263.tar.gz','x:gz')as archive:
        for path,wanted in sorted(members.items()):
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
            info=tarfile.TarInfo(path.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE263.tar.gz','r:gz')as archive:
        assert len(archive.getmembers())==len(members)
        for member in archive:assert member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name]
    result=dict(status='HOST_synthetic_copy_probe_frozen_and_archive_verified',freeze_sha256=sha(freeze),archive_sha256=sha(P/'SOURCE263.tar.gz'),members=len(members),NPU_used=False)
    with(P/'SOURCE263_RECEIPT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
