"""Freeze only a metadata probe; no inference or device accesses."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-packed-FULL-host-vector-metadata-v258'
PREVIOUS=B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257/FREEZE257.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(P/'FREEZE258.json').exists()and not(P/'SOURCE258.tar.gz').exists()
    assert sha(PREVIOUS)=='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
    files=dict(json.loads(PREVIOUS.read_text())['files']);files[str(PREVIOUS)]=sha(PREVIOUS)
    additional=[*P.glob('*.py'),P/'HOST_PACKET258_RESULT.json',Path(__file__),
        B/'tree-static-q5-immutable-index-cache-host-v247/guard247.py',
        Path('/root/SpecSLO/resume-20261012/run_baseline_calibration-v1.py')]
    for p in additional:
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
        files[str(p)]=sha(p)
    for path,wanted in files.items():assert sha(path)==wanted,path
    row=dict(schema='ROOT258.actual-NPU-metadata-only-private-source-freeze.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
        scope='Original checked HOST factory versus one integer packet; device values and paired timings only',
        original_native_full_source_verifier_required=True,models_loaded=False,
        actual_NPU_executed=False,neural_qualified=False,performance_qualified=False,adopted=False)
    with(P/'FREEZE258.json').open('x')as f:json.dump(row,f,indent=2);f.write('\n')
    with tarfile.open(P/'SOURCE258.tar.gz','x:gz')as archive:
        for path,wanted in {**files,str(P/'FREEZE258.json'):sha(P/'FREEZE258.json')}.items():
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
            info=tarfile.TarInfo(path.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE258.tar.gz','r:gz')as archive:
        expected={**files,str(P/'FREEZE258.json'):sha(P/'FREEZE258.json')}
        assert len(archive.getmembers())==len(expected)
        for member in archive:
            assert member.isfile()and member.name=='/'.join(Path('/'+member.name).parts[1:])
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==expected['/'+member.name]
    out=dict(status='HOST_frozen_source_archive_verified_NO_NPU',directory=str(P),
        freeze_sha256=sha(P/'FREEZE258.json'),archive_sha256=sha(P/'SOURCE258.tar.gz'),members=len(files)+1,
        actual_NPU_executed=False,production_installed=False)
    with(P/'SOURCE258_RECEIPT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
