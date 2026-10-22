"""Freeze the synthetic same-Q/K/V FULL FIA capacity experiment."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-FIA-fixed-q5-mask-capacity-cost-v264'
OLD=B/'tree-fused-byte-branch-KV-copy-v263/FREEZE263.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(P/'FREEZE264.json').exists()and not(P/'SOURCE264.tar.gz').exists()
    assert sha(OLD)=='a8b7fb9ae6b2825aef9dd6017c9197b8a4298593061c149e6c59207f7c660f93'
    host=json.loads((P/'HOST264_ARGUMENTS_RESULT.json').read_text());assert len(host['cases'])==42 and host['NPU_used']is False
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT264.synthetic-fixed-q5-FIA-capacity-review.v1',
        existing_native_environment_only=True,real_model_weights_loaded=False,
        scope='syntheticFULLFIA only: samequeries/KVtables/visiblekeys/actualq5/KV2560; comparemask capacities5,8,9,10,11,12,16',
        prefixes=[127,1505,2555],layouts=['all24-spines','12-spines-12-side-groups'],
        original_FIA_FULL_validator_and_argument_preparation_prefix_unchanged=True,
        original_sparse_mode1_inner_precise1=True,workspace_reused_serially_as_original_Native_FULL_path=True,
        distinct_outputs_for64_repeated_FIA_calls_per_graph=True,every_output_must_equal_eager=True,
        rotated_order80rounds_per_configuration=True,all4idle_and_original_source_verifiers_required=True,
        neural_qualified=False,Goodput=None,production_installed=False)
    (P/'REVIEW264.json').write_text(json.dumps(review,indent=2)+'\n')
    files=dict(json.loads(OLD.read_text())['files']);files[str(OLD)]=sha(OLD)
    extra=[*P.glob('*'),Path(__file__),D/'ROOT263_INDEPENDENT_SYNTHETIC_COPY_GRAPH_AUDIT.json',
        Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')]
    for p in extra:
        if p.is_file():
            if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
            files[str(p)]=sha(p)
    for path,wanted in files.items():assert sha(path)==wanted,path
    freeze=P/'FREEZE264.json'
    with freeze.open('x')as f:json.dump(dict(schema='ROOT264.synthetic-FIA-capacity-frozen-source.v1',UTC=review['UTC'],files=files,NPU_used=False,neural_qualified=False,production_installed=False),f,indent=2);f.write('\n')
    members={**files,str(freeze):sha(freeze)}
    with tarfile.open(P/'SOURCE264.tar.gz','x:gz')as archive:
        for path,wanted in sorted(members.items()):
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
            info=tarfile.TarInfo(path.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE264.tar.gz','r:gz')as archive:
        assert len(archive.getmembers())==len(members)
        for member in archive:assert member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name]
    result=dict(status='HOST_synthetic_FIA_capacity_sources_archive_verified',freeze_sha256=sha(freeze),archive_sha256=sha(P/'SOURCE264.tar.gz'),members=len(members),NPU_used=False)
    with(P/'SOURCE264_RECEIPT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
