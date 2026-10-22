"""Freeze the synthetic same-Q/K/V FULL FIA capacity experiment."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-lifetime-KV-bucket-synthetic-invariance-v267'
OLD=B/'tree-FIA-fixed-q5-mask-capacity-cost-v264/FREEZE264.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(P/'FREEZE267.json').exists()and not(P/'SOURCE267.tar.gz').exists()
    assert sha(OLD)=='1b8a93b23f231f4d570c7cb5fbe8face5f8098ddf849b9c69fc85b62b8207216'
    host=json.loads((P/'HOST267_ARGUMENTS_RESULT.json').read_text());assert len(host['cases'])==256 and host['NPU_used']is False
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT267.synthetic-fixed-q5-FIA-capacity-review.v1',
        existing_native_environment_only=True,real_model_weights_loaded=False,
        scope='Conditional only: constant4096 sentinel plus private per-request lifetimeKV buckets; test per-bucket position/future/collateral-context invariance and synthetic cost',
        prefixes=[127,511,1505,2555],heads_and_KVheads=[[24,3],[16,2]],real_groups=[16,23],mask_capacities=[10,11],rotations=[0,1,3,7],future_variants=[0,1],one_unused4096_sentinel_required=True,
        original_FIA_FULL_validator_and_argument_preparation_prefix_unchanged=True,
        original_sparse_mode1_inner_precise1=True,workspace_reused_serially_as_original_Native_FULL_path=True,
        only_after266_completes_and_all4_idle=True,current266_sources_and_process_untouched=True,
        distinct_outputs_for64_repeated_FIA_calls_per_graph=True,every_output_must_equal_eager=True,
        rotated_order80rounds_per_configuration=True,all4idle_and_original_source_verifiers_required=True,
        neural_qualified=False,Goodput=None,production_installed=False)
    (P/'REVIEW267.json').write_text(json.dumps(review,indent=2)+'\n')
    files=dict(json.loads(OLD.read_text())['files']);files[str(OLD)]=sha(OLD)
    extra=[*P.glob('*'),Path(__file__),D/'ROOT263_INDEPENDENT_SYNTHETIC_COPY_GRAPH_AUDIT.json',
        B/'tree-permanent-KV-bucket-synthetic-FIA-probe-v261/synthetic261.py',B/'tree-permanent-KV-bucket-synthetic-FIA-probe-v261/HOST_SYNTHETIC261_GEOMETRY_RESULT.json',D/'ROOT264_INDEPENDENT_SYNTHETIC_FIA_COST_AUDIT.json',Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')]
    for p in extra:
        if p.is_file():
            if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
            files[str(p)]=sha(p)
    for path,wanted in files.items():assert sha(path)==wanted,path
    freeze=P/'FREEZE267.json'
    with freeze.open('x')as f:json.dump(dict(schema='ROOT267.synthetic-FIA-capacity-frozen-source.v1',UTC=review['UTC'],files=files,NPU_used=False,neural_qualified=False,production_installed=False),f,indent=2);f.write('\n')
    members={**files,str(freeze):sha(freeze)}
    with tarfile.open(P/'SOURCE267.tar.gz','x:gz')as archive:
        for path,wanted in sorted(members.items()):
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
            info=tarfile.TarInfo(path.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE267.tar.gz','r:gz')as archive:
        assert len(archive.getmembers())==len(members)
        for member in archive:assert member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name]
    result=dict(status='HOST_synthetic_FIA_capacity_sources_archive_verified',freeze_sha256=sha(freeze),archive_sha256=sha(P/'SOURCE267.tar.gz'),members=len(members),NPU_used=False)
    with(P/'SOURCE267_RECEIPT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
