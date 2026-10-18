"""Prepare nine ordered fresh-process commands from an already qualified tree.

This program checks source/evidence bytes and writes a plan. It launches no
process and queries no device. ROOT executes and audits each cell separately.
"""
import argparse,datetime,hashlib,json,re,tarfile
from pathlib import Path
from final_candidate_eligibility import qualified,sha,need
from final_common_comparison import ORDERS,collect

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
FIXED=B/'final-fixed-gamma4-stream-witness-v1'
OFFICIAL=D/'final-official-tp4-publication-rebind-v2'


def verify_prepared(directory,manifest_path,expected_manifest_sha=None):
    if expected_manifest_sha is not None:
        need(sha(manifest_path)==expected_manifest_sha,'exact prepared source manifest')
    data=json.loads(manifest_path.read_text())
    # Both prepared baseline manifests retain their original key names.
    groups=[data[k]for k in ('files','static_files','HOST_evidence_files','original_pinned_files')if k in data]
    need(bool(groups),'explicit original prepared source inventory')
    for group in groups:
        for name,expected in group.items():
            path=Path(name);path=path if path.is_absolute()else directory/path
            need(path.is_file()and not path.is_symlink()and sha(path)==expected,'unchanged prepared baseline '+str(path))
    if 'archive'in data:
        archive=data['archive'];path=Path(archive['path'])
    else:
        path=directory/'PREPARED_PUBLICATION_V2_SOURCE.tar.gz'
        archive=dict(path=str(path),sha256='e0ad778bfc7ea66b144c720b83fe966697a7a3b9b0cee03997cc65c33f759632')
    need(sha(path)==archive['sha256'],'unchanged prepared source archive')
    return dict(manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),
                archive_path=str(path),archive_sha256=archive['sha256'])


def build(selection_path,directory,prefix):
    selection_path=Path(selection_path).resolve();directory=Path(directory).resolve()
    need(directory.is_relative_to(B)and not directory.exists(),'fresh private comparison directory')
    need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,60}',prefix)is not None,'short safe fresh run-tag prefix')
    selected=json.loads(selection_path.read_text())
    for path,expected in selected['source_and_evidence_hashes'].items():
        need(sha(path)==expected,'qualified candidate source/evidence unchanged')
    receipt=json.loads(Path(selected['eligible_receipt']).read_text())
    audit=json.loads(Path(selected['eligible_audit']).read_text())
    metrics=collect('qualified_dynamic_tree',selected['eligible_raw_trace'])
    need(qualified(receipt,audit,metrics)==selected['eligibility'],'actual qualified candidate remains eligible')
    freeze=sha(selected['freeze_manifest']);archive=sha(selected['source_archive'])
    fixed_binding=verify_prepared(FIXED,FIXED/'PREPARED_SOURCE_MANIFEST.json',
        'b9a20faad5dba0bbb14d3d533bc1295dd0ccdca2608f9431bf1466b798684e08')
    # Official v2 uses a source binding inventory plus a separate frozen manifest.
    official_manifest=OFFICIAL/'PREPARED_PUBLICATION_V2_SOURCE_MANIFEST.json'
    official_binding=verify_prepared(OFFICIAL,official_manifest,
        '3c03fc5f3ca4348fb94dc31d0701c0a892fefe54ece53f78fa9df28979528936')
    bindings=json.loads((OFFICIAL/'SOURCE_BINDINGS.json').read_text())
    for group in ('local_files','original_rebind_sources','publication_witness_sources','official_runtime_and_sources'):
        for path,expected in bindings[group].items():
            need(sha(path)==expected,'unchanged official baseline dependency '+path)
    inventory={str(f)for root in bindings['Python_source_roots']for f in Path(root).rglob('*.py')}
    need(inventory==set(bindings['Python_source_inventory']),'exact official Python source inventory')
    cells=[]
    for block,order in enumerate(ORDERS,1):
        for position,method in enumerate(order):
            tag=f'{prefix}-b{block}-p{position}-{method}'
            if method=='fixed_gamma4':
                argv=['bash',str(FIXED/'run.sh'),str(directory/'fixed-results'),tag]
                root=directory/'fixed-results/runs'/('candidate-on-'+tag)
                outputs=dict(result=str(root/'result.json'),mechanism_gate=str(root/'mechanism-gate.json'),
                             witness_directory=str(root/'host-stream'))
            elif method=='official_MC2_ON':
                argv=['bash',str(OFFICIAL/'run.sh'),'1',tag]
                root=OFFICIAL/'future-results/runs'/tag
                outputs=dict(result=str(root/'result.json'),original_audit=str(root/'independent-audit.json'),
                    receipt=str(OFFICIAL/'actual-metadata'/tag/'receipt.json'),
                    witness_directory=str(B/'final-official-publication-witness-v1/actual'/tag))
            else:
                argv=['python3','-B',selected['guard_script'],'--tag',tag,
                      '--expected-freeze-sha256',freeze,'--expected-archive-sha256',archive]
                root=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
                outputs=dict(receipt=str(root/('receipt-'+tag+'.json')),actual=str(root/'actual'/tag),
                    terminal_auditor=selected['terminal_auditor_script'],
                    raw_trace=str(root/'actual'/tag/selected['full_trace_relative_path']))
            cells.append(dict(block=block,position=position,method=method,tag=tag,argv=argv,outputs=outputs,
                execution_lock='tree guard owns ROOT lock itself; ROOT locks baselines only',
                source_and_environment_checks_required=True,owned_cleanup_and_four_idle_required=True,
                independent_actual_result_audit_required=True))
    result=dict(schema='SpecSLO.prepared-final-nine-cell-command-plan.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_path=str(selection_path),
        selection_sha256=sha(selection_path),candidate_freeze_sha256=freeze,
        fixed_prepared_binding=fixed_binding,official_prepared_binding=official_binding,
        cells=cells,all_cells_required=True,statistics='method medians, all cells retained; no fastest selection',
        model_processes_launched=0,NPU_used=False,goal_achieved=False)
    directory.mkdir(mode=0o700)
    with(directory/'run-plan.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--prefix',required=True)
    args=p.parse_args();value=build(args.selection,args.directory,args.prefix)
    print(json.dumps(dict(cells=len(value['cells']),model_processes_launched=0,NPU_used=False)))
