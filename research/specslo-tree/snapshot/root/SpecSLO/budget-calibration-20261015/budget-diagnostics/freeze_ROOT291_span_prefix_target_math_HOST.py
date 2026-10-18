"""Freeze fresh target-only NN qualification after isolated295 operator proof."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-all-layer-prefix-dual-cache-target-cost-v291'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not(P/'FREEZE291.json').exists()
    assert read(P/'HOST291_RESULT.json')['cases']==408
    assert read(P/'HOST_GRID291_RESULT.json')['case_count']==408
    assert read(P/'HOST_SPLIT291_SPANS_RESULT.json')['actual_cases']==408
    op=read(D/'ROOT295_INDEPENDENT_OPERATOR_AUDIT.json')
    assert op['status'].startswith('PASS_')and op['full_cache_raw16_controls']==120
    assert all(t['span_relative']<1 for t in op['timings'])
    backend=(P/'static_state291.py').read_text()
    assert '.prefix291_copies.run('not in backend and 'SpanCopies' in backend
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT291.span-prefix-dual-write-target-NN.v1',
        actual_geometry='unchangedM120/S24/uniformq5/KV2560, C96 andprimary+sidegroups24',
        operator='one768-program all64-layer contiguous committed-prefix memcpy, per-layer raw dual cache write replaces Native reshape-and-cache; original attention retained',
        actual_operator295_fresh_proof=True,NN_math_qualification_inherited=False,
        source_prefix_maps='actual Native HOST schedules and physical row bijection; no device metadata D2H',
        original_cache_addresses_retained=True,no_float_arithmetic_in_new_KV_operators=True,
        independent_math='fresh72 own prefixes and408 target cases, bitwise hidden/all128KV/greedy, graph/eager, committed prefixes protected',
        actual_predeclared_cold_families=6,extra_changed_restored_dispatches=12,
        paired_component_cost='11 N/fork configurations, B4..8,20rotated rounds; math mustfinish first',
        BF16_models_and_original_sampler=True,quantization=False,
        no_full_workload_or_serving_scope=True,whole463_AR=False,Goodput=None,adopted=False)
    (P/'REVIEW291.json').write_text(json.dumps(review,indent=2)+'\n')
    files={}
    for directory,epoch in [('tree-prefix-grid-target-replay-node-cost-v280',280),('tree-safe-grid-span-dual-cache-probe-v295',295)]:
        f=B/directory/f'FREEZE{epoch}.json'
        for path,h in read(f)['files'].items():
            if path in files:assert files[path]==h
            files[path]=h
        files[str(f)]=sha(f)
    for f in P.iterdir():
        if f.is_file():
            if f.suffix=='.py':compile(f.read_text(),str(f),'exec')
            files[str(f)]=sha(f)
    for f in [D/'ROOT295_INDEPENDENT_OPERATOR_AUDIT.json',Path(__file__)]:files[str(f)]=sha(f)
    for path,h in files.items():assert sha(path)==h,path
    f=P/'FREEZE291.json'
    with f.open('x')as out:json.dump(dict(schema='ROOT291.frozen-fresh-target-numeric-diagnostic.v1',files=files,review=review,NPU_used=False,model_qualification_inherited=False),out,indent=2);out.write('\n')
    inventory={**files,str(f):sha(f)}
    with tarfile.open(P/'SOURCE291.tar.gz','x:gz')as ar:
        for path,h in sorted(inventory.items()):
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==h
            info=tarfile.TarInfo(path.lstrip('/'));info.mode=0o600;info.size=len(data);info.mtime=0;ar.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE291.tar.gz','r:gz')as ar:
        assert len(ar.getmembers())==len(inventory)
        for m in ar:assert m.isfile()and hashlib.sha256(ar.extractfile(m).read()).hexdigest()==inventory['/'+m.name]
    pins=dict(freeze_sha256=sha(f),archive_sha256=sha(P/'SOURCE291.tar.gz'),members=len(inventory))
    (P/'SOURCE291_RECEIPT.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps(pins))
if __name__=='__main__':main()
