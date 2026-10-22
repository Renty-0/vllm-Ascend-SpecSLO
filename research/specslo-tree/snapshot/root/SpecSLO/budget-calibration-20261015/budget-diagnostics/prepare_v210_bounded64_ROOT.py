"""Freeze ROOT's bounded normal64 source after exact projections and HOST groups."""
import ast
import contextlib
import datetime
import hashlib
import io
import json
import runpy
import tarfile
import time
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=D/'v210-source-design-only'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
def merge(*maps):
    result={}
    for values in maps:
        for path,digest in values.items():
            assert path not in result or result[path]==digest,path
            result[path]=digest
    return result


def frozen_map(folder,number):
    path=folder/f'FROZEN_SOURCE{number}.json';f=read(path)
    values=merge(*(f[k]for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews')),
                 {str(folder/f'CPU_RESULT{number}.json'):f['CPU_sha256'],str(path):sha(path),
                  f['root_review']['path']:f['root_review']['sha256']})
    for p,h in values.items():assert sha(p)==h,p
    return values


def main():
    for name in('CPU_RESULT210.json','FROZEN_ROOT210.json','PREPARED_SOURCE_ROOT210.tar.gz'):
        assert not(P/name).exists(),'fresh preparation only'
    own={str(p):sha(p)for p in sorted(P.iterdir())if p.is_file()and p.suffix in('.py','.sh')}
    for name in('guard210.py','check210.py','runtime210.py','component210.py','result210.py','geometry210.py',
                'actor210.py','bank210.py','target210.py','audit3210.py'):
        compile((P/name).read_text(),str(P/name),'exec')
    folders=('HOST-attempts','runtime-HOST-attempts','guard-HOST-attempts');results=[]
    for name in folders:
        values=[p/'RESULT.json'for p in(P/name).iterdir()if(p/'RESULT.json').exists()]
        passed=[p for p in values if read(p)['status'].startswith('passed_')]
        latest=max(passed,key=lambda p:p.stat().st_mtime);r=read(latest)
        assert r['errors']==[]and r['NPU_used']is r['model_constructed']is r['Torch_library_imported']is False
        for path,digest in own.items():
            if path.endswith('.py')or name=='guard-HOST-attempts':
                assert r['source_before'][path]==r['source_after'][path]==digest,(name,path)
        results.append(ref(latest))
    # Rebuild all four complete-source donor projections without editing the
    # actual candidate, including the additional role-None ownership checks.
    namespace=runpy.run_path(str(D/'build_v210_normal64_source_prototype_ROOT.py'),run_name='_ROOT210_SOURCE_ONLY')
    projected=P/'projection-HOST-attempts'/str(time.time_ns())/'source';projected.parent.mkdir(parents=True)
    namespace['main'].__globals__['OUT']=projected
    with contextlib.redirect_stdout(io.StringIO()):namespace['main']()
    for name in('bank210.py','target210.py','actor210.py','audit3210.py'):
        assert (projected/name).read_bytes()==(P/name).read_bytes(),name
    # Reproduce both launcher projections, so all original inference flags,
    # environment exports, extension DSO/required op and thread policy remain.
    launch=read(P/'LAUNCH_PROJECTION210.json')
    for row in launch['projections']:
        source=Path(row['source']);assert sha(source)==row['source_sha256']
        text=source.read_text()
        for before,after,count in row['changes']:
            assert text.count(before)==count;text=text.replace(before,after)
        assert text==Path(row['output']).read_text()and sha(row['output'])==row['output_sha256']
    external=merge(frozen_map(B/'tree-bounded-parent4-shared4-eager-diagnostic-v199',199),
                   frozen_map(B/'tree-normal-cap3-original-geometries-full-v207',207))
    evidence={str(p):sha(p)for name in(*folders,'projection-HOST-attempts')
              for p in(P/name).rglob('*')if p.is_file()}
    for path in(P/'LAUNCH_PROJECTION210.json',Path(__file__),
                 D/'build_v210_normal64_source_prototype_ROOT.py',
                 D/'check_v210_normal64_source_HOST_ROOT.py',D/'check_v210_runtime_source_HOST_ROOT.py',
                 D/'check_v210_guard_control_HOST_ROOT.py'):
        evidence[str(path)]=sha(path)
    cpu=dict(schema='ROOT.V210.actual-three-HOST-groups.v1',status='passed',tests_run=3,
             errors=0,failures=0,Torch_library_imported=False,NPU_used=False,model_constructed=False,
             results=results,current_sources=own,source_groups=['writer/auditor/mapper','runtime/spawn-boundary','supervision-control'],
             carrier_results_are_not_numerical_qualification=True,old_suites_rerun=False)
    write(P/'CPU_RESULT210.json',cpu)
    inputs=merge(own,external,evidence,{str(P/'CPU_RESULT210.json'):sha(P/'CPU_RESULT210.json')})
    review=dict(schema='ROOT.V210.bounded64-source-and-guard-review.v1',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed_source_review',
                candidate=str(P),reviewed_inputs=inputs,current_source_peer_review=False,
                mechanical_whole_source_projections=4,launch_projections=2,
                actual_HOST_groups=3,models_launched=False,NPU_used=False,
                reviewed_semantics=[
                    'Bounded normal-only B3 Q64/M64/S16. There is no production permission or full463 path.',
                    'Original199 constructor/IO/startup source stays frozen. Only component call, source-first gate, truthful labels/reply key and result auditor differ.',
                    'New top-level worker is pickleable; its child recreates the exact projected worker before any Native model import.',
                    'Original current201/44/114 and B4/M80 hard gates precede new normal64 model work.',
                    'Current same-M full64 target hidden/head, all128 target queryKV, own bank compaction, original protocol and natural d3 repair/next; discovery plus two complete zero traces required.',
                    'Both cold G64 and H64 are diagnostic keys. Actual flags must close naturally; original entry objects remain.',
                    'Finally-revoked geometry delegates all other domains and restores control/layout owner on errors.',
                    'Original verifier/owned cleanup/supervision bodies retained, clean env, ROOT self-lock, private bounds and source pre/post checks.',
                ],limitations=[
                    'No independent agent source review is claimed for these new files. The frozen donors have their own source reviews, which are not this model qualification.',
                    'HOST flags/tensors/processes are explicit carriers or stubs, not numerical/device/baseline evidence.',
                    'Actual ROOT210 records use a fresh tag under frozen199 allowed private actual root; donor sources/existing actual are unchanged.',
                    'Mixed promoted-child origin, optional budget full geometry, cost and complete workload performance remain unqualified.',
                ],formal_Goodput=False,adopted=False)
    root=D/'ROOT_V210_BOUNDED_NORMAL64_SOURCE_AND_GUARD_REVIEW.json';write(root,review)
    frozen=dict(schema='ROOT.V210.normal64-bounded-source-freeze.v1',source_files=own,external_files=external,
                retained_HOST_evidence=evidence,CPU_sha256=sha(P/'CPU_RESULT210.json'),root_review=ref(root),
                ROOT_source_reviewed=True,current_source_peer_review=False,
                production_issuer=False,full463=False,formal_Goodput=False,adopted=False)
    write(P/'FROZEN_ROOT210.json',frozen)
    selected=merge(inputs,{str(P/'FROZEN_ROOT210.json'):sha(P/'FROZEN_ROOT210.json'),str(root):sha(root)})
    archive=P/'PREPARED_SOURCE_ROOT210.tar.gz'
    with tarfile.open(archive,'x:gz')as saved:
        for path,digest in selected.items():
            assert sha(path)==digest,path;saved.add(path,arcname=path.lstrip('/'),recursive=False)
    result=dict(schema='ROOT.V210.frozen-bounded64-source-preparation.v1',status='source_frozen_unrun',
                candidate=str(P),freeze=ref(P/'FROZEN_ROOT210.json'),archive=ref(archive),
                archive_members=len(selected),source_files=len(own),external_files=len(external),
                retained_HOST_files=len(evidence),ROOT_review=ref(root),HOST_results=results,
                current_source_peer_review=False,NPU_used=False,model_launched=False,
                production_issuer=False,full463=False,Goodput=None,adopted=False)
    out=D/'ROOT_V210_PREPARED_BOUNDED64_SOURCE.json';write(out,result)
    print(json.dumps(dict(prepared=ref(out),freeze=result['freeze'],archive=result['archive'],members=len(selected))))


if __name__=='__main__':main()
