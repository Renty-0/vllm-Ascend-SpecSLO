"""Reopen exact operator result, frozen sources, archive and cleanup receipt."""
import argparse,hashlib,json,statistics,tarfile
from collections import Counter
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-prefix-row-and-span-cache-probe-v293'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    root=P/'actual'/args.tag;r=read(root/'receipt.json');pins=read(P/'SOURCE293_RECEIPT.json')
    assert r['status']=='complete'and r['errors']==[]and r['owned_remaining']==[]and r['sources_unchanged']is True
    assert r['devices_before']['all_idle']is True and r['devices_after']['all_idle']is True
    assert sha(P/'FREEZE293.json')==pins['freeze_sha256']and sha(P/'SOURCE293.tar.gz')==pins['archive_sha256']
    files=read(P/'FREEZE293.json')['files']
    for path,h in files.items():assert sha(path)==h,path
    inventory={**files,str(P/'FREEZE293.json'):pins['freeze_sha256']}
    with tarfile.open(P/'SOURCE293.tar.gz','r:gz')as ar:
        assert len(ar.getmembers())==len(inventory)==pins['members']
        for m in ar:assert m.isfile()and hashlib.sha256(ar.extractfile(m).read()).hexdigest()==inventory['/'+m.name]
    for name in('pre_verifier','post_verifier'):
        v=r[name];assert v['exit_code']==0 and sha(v['path'])==v['sha256']
        v=v['actual_result'];assert v['status']=='pass'and v['errors']==[]
        assert(v['checked_frozen_files'],v['checked_external_files'],v['checked_packages'])==(10496,15,212)
    path=Path(r['paired_cost_path']);assert sha(path)==r['paired_cost_sha256'];v=read(path)
    assert v['status']=='complete'and v['models_loaded']is False and v['real_model_weights_loaded']is False
    assert v['attention_executed']is False and v['synthetic_KV_copy_executed']is True
    assert v['candidate_source_sha256']==files[str(B/'tree-contiguous-prefix-span-copy-host-v292/kernels292.py')]
    assert v['row_and_dual_source_sha256']==files[str(B/'tree-prefix-precopy-dual-cachewrite-host-v288/kernels288.py')]
    assert v['source_sha256']==files[str(P/'probe293.py')]
    assert v['cache_layers']==64 and v['raw_bit_coverage']=={'256':65536,'384':65536}
    checks=v['map_cases'];assert len(checks)==120 and checks==v['raw_bit_replays']
    grouped={}
    for row in checks:
        assert row['all128_full_cache_storages_raw16_equal']is True and row['both_candidates_equal']is True
        assert row['same_buffers_changed_and_restored']is True and row['actual_ACLGraph_replayed']is True
        key=(row['width'],row['style'],row['case']);grouped.setdefault(key,[]).append(row['variant'])
    assert len(grouped)==40 and all(variants==[0,1,0]for variants in grouped.values())
    timings=[]
    for row in v['timings']:
        assert len(row['pairs'])==42
        by_method={k:[]for k in('baseline','row_prefix','span_prefix')}
        orders=Counter()
        for pair in row['pairs']:
            assert set(pair['order'])==set(by_method)and set(pair['ms'])==set(by_method)
            orders[tuple(pair['order'])]+=1
            for name,ms in pair['ms'].items():assert ms>0;by_method[name].append(ms)
        assert sorted(orders.values())==[14,14,14]
        medians={k:statistics.median(values)for k,values in by_method.items()}
        assert medians==row['median_ms']
        timings.append(dict(width=row['width'],style=row['style'],case=row['case'],copy_grid=row['copy_programs'],median_ms=medians,
            row_relative=medians['row_prefix']/medians['baseline'],span_relative=medians['span_prefix']/medians['baseline']))
    assert len(timings)==16
    out=dict(status='PASS_independent_operator293_raw_bits_graphs_rotated_cost_and_owned_cleanup',
        tag=args.tag,receipt_sha256=sha(root/'receipt.json'),result_sha256=sha(path),
        full_cache_raw16_controls=120,all65536_words_each_width=True,cache_layers=64,
        original_Native_cache_writer_reference=True,source_and_archive_verified=True,
        original_environment_prepost_verified=True,all4_idle_and_owned_cleanup_complete=True,
        timings=timings,scope='isolated cache operators only; no attention or model',
        whole463_AR=False,Goodput=None,adopted=False)
    dest=B/'budget-diagnostics/ROOT293_INDEPENDENT_OPERATOR_AUDIT.json'
    with dest.open('x')as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
