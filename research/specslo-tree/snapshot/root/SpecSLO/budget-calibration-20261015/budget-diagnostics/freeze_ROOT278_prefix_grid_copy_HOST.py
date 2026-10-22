"""Freeze the operator-only three-arm raw-byte tiling experiment."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-prefix-grid-byte-KV-copy-v278'
OLD=B/'tree-native-rank3-KV-copy-cost-v265/FREEZE265.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(P/'FREEZE278.json').exists()
    assert sha(OLD)=='abb76be9ea44170b0837a1fc5f7b65cf19c538f3a75702764337294a93a6f219'
    host=json.loads((P/'HOST278_RESULT.json').read_text())
    assert host['actual_original_Native_maps']==172 and host['all_nonidentity_programs_preserved']is True
    files=dict(json.loads(OLD.read_text())['files']);files[str(OLD)]=sha(OLD)
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        schema='ROOT278.tiled-raw-byte-copy-review.v1',
        mechanism='Reuse exact263 one-row kernel binary, grid selected from128/256/512/1024/1792; only proved identity suffix omitted; fixed1792 indexbuffers unchanged',
        device_arms=['original rank3 fourgather/scatters','retained263 one-row fused rawbyte kernel','new278 prefix grid, same263 kernel binary'],
        gates='All172 actual Native maps at widths256/384; wholecache K/V equal; all65536 raw16-bit payloads with changed/restored ACLGraph controls;80 rotated pairs per five maps per width',
        original10496_15_212_pre_post=True,owned_ROOT_lock=True,all4idle_pre_post=True,
        models_loaded=False,NN_qualified=False,Goodput=None,adopted=False)
    (P/'REVIEW278.json').write_text(json.dumps(review,indent=2)+'\n')
    for p in [*P.glob('*'),Path(__file__),B/'budget-diagnostics/ROOT275_INDEPENDENT_TILED_COPY_AUDIT_AND_REJECTION.json']:
        if p.is_file():
            if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
            files[str(p)]=sha(p)
    for p,h in files.items():assert sha(p)==h,p
    freeze=P/'FREEZE278.json'
    with freeze.open('x')as f:json.dump(dict(schema='ROOT278.frozen-operator-only.v1',UTC=review['UTC'],files=files),f,indent=2);f.write('\n')
    members={**files,str(freeze):sha(freeze)}
    with tarfile.open(P/'SOURCE278.tar.gz','x:gz')as archive:
        for p,h in sorted(members.items()):
            data=Path(p).read_bytes();assert hashlib.sha256(data).hexdigest()==h
            info=tarfile.TarInfo(p.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    result=dict(status='HOST_prefix_grid_operator_sources_frozen',freeze_sha256=sha(freeze),archive_sha256=sha(P/'SOURCE278.tar.gz'),members=len(members),NPU_used=False)
    with(P/'SOURCE278_RECEIPT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
