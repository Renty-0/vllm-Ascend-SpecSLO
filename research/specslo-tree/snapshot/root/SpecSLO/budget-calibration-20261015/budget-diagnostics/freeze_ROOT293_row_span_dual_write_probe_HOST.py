"""Freeze a synthetic-only experiment; no model qualification is inherited."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-prefix-row-and-span-cache-probe-v293'
K=B/'tree-prefix-precopy-dual-cachewrite-host-v288'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not(P/'FREEZE293.json').exists()
    assert read(P/'HOST293_RESULT.json')['raw16_input_coverage']=={'256':65536,'384':65536}
    assert read(K/'HOST_SPLIT288_RESULT.json')['case_count']==70
    span=B/'tree-contiguous-prefix-span-copy-host-v292'
    assert read(span/'HOST292_RESULT.json')['exhaustive_cases']==1664
    files={}
    for directory,epoch in [('tree-dynamic-root-alternate-eager-compat-full-v287',287),('tree-tiled-byte-branch-KV-copy-v275',275)]:
        f=B/directory/f'FREEZE{epoch}.json'
        for path,h in read(f)['files'].items():
            if path in files:assert files[path]==h
            files[path]=h
        files[str(f)]=sha(f)
    for root in(P,K,span):
        for f in root.iterdir():
            if f.is_file():
                if f.suffix=='.py':compile(f.read_text(),str(f),'exec')
                files[str(f)]=sha(f)
    files[str(Path(__file__))]=sha(__file__)
    for path,h in files.items():assert sha(path)==h,path
    review=dict(schema='ROOT293.frozen-synthetic-cache-operators.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_only_HOST_gates_passed=True,NPU_qualification_inherited=False,models_loaded=False,
        kernels='three-way comparison: retained Native per-layer cache write/copy, one all-layer row prefix copy plus dual writes, one all-layer contiguous span copy plus dual writes',
        actual_rank3_cache_shape=True,changed_restored_full_cache_raw_bits_required=True,
        all65536_raw16_patterns_required=True,both_contiguous_and_strided_QKV_inputs_required=True,
        paired_rotated_isolated_cost=True,methods_per_rotated_round=3,rotated_rounds_per_case=42,real_runtime_and_attention_not_executed=True,
        actual_Native408_math_and_full463_AR_still_required=True,Goodput=None,adopted=False)
    f=P/'FREEZE293.json'
    with f.open('x')as out:json.dump(dict(**review,files=files),out,indent=2);out.write('\n')
    inventory={**files,str(f):sha(f)}
    with tarfile.open(P/'SOURCE293.tar.gz','x:gz')as ar:
        for path,h in sorted(inventory.items()):
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==h
            info=tarfile.TarInfo(path.lstrip('/'));info.mode=0o600;info.size=len(data);info.mtime=0;ar.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE293.tar.gz','r:gz')as ar:
        assert len(ar.getmembers())==len(inventory)
        for m in ar:assert hashlib.sha256(ar.extractfile(m).read()).hexdigest()==inventory['/'+m.name]
    pins=dict(freeze_sha256=sha(f),archive_sha256=sha(P/'SOURCE293.tar.gz'),members=len(inventory))
    (P/'SOURCE293_RECEIPT.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps(pins))
if __name__=='__main__':main()
