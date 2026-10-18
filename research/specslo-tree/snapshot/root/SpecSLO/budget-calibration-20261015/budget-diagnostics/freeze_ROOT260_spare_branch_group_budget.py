"""Freeze the LIVE budget candidate with all prior model proofs as inputs only."""
import ast,datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'
P=B/'tree-spare-branch-group-budget-full-v260'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
    if not v:raise RuntimeError(m)
def main():
    need(not(P/'FREEZE260.json').exists()and not(P/'SOURCE260.tar.gz').exists(),'fresh immutable epoch')
    summary=D/'ROOT257_COMPLETE_RESULT_FOR_NEXT_OPTIMIZATION.json';prior=json.loads(summary.read_text())
    need(prior['status']=='complete_independently_audited'and prior['all463_equal_on_all3targets'],'complete previous correct full model result')
    for name in ['HOST_FULL260_PROTOCOL_RESULT.json','HOST_ADMISSION260_RESULT.json','HOST_PRIORITY260_RESULT.json','HOST_BUDGET260_FINAL_RESULT.json']:
        v=json.loads((P/name).read_text());need(v['status'].startswith('HOST_'),'actual HOST gate '+name)
    v=json.loads((P/'HOST_BUDGET260_FINAL_RESULT.json').read_text())
    need(v['grant_cases']==160 and v['controller_cases']==144 and v['actual_policy_closure_helper_executed']
        and v['irreversible_over_capacity_reservation_rejected']and v['eager_only_original_grants_unchanged'],'new actual allocator/controller scope')
    f=OLD/'FREEZE257.json';need(sha(f)=='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2','exact prior freeze')
    files=dict(json.loads(f.read_text())['files']);files[str(f)]=sha(f)
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT260.spare-branch-group-budget-integration.v1',
        original257_correct_full_summary=str(summary),original257_summary_sha256=sha(summary),
        actual_model_backend='unchanged immutable255 M120/S24/KV2560 and all original BF16 weights/greedy',
        LIVE_shape='Original typed split allocator with individual cap5 eligibility chosen from spare24-N groups by class and positive g5-g4, all other normal owners cap4',
        LIVE_target='Original full controller with added N+B5<=24 admission; whole tickets, original SLO-first order and deferred records retained',
        original_class_shape_and_controller_methods_unmodified=True,
        scalar_gain_need_charge_refinement_and_typed_publication_unchanged=True,
        N24plus_grant_equivalence_HOST_checked=True,C96=True,global_draft192=True,active48=True,
        original_global_environment_TARGET0_unchanged=True,actual_LIVE_target_SLO_first=True,
        packet258_and_deferred259_not_installed=True,original463_arrivals_SLO_and_output256=True,
        all_original_startup_components_required=True,fresh_same_instance227_and72_prefixes_required=True,
        original_warm_twozero_seal_full463_and_complete_ownAR_required=True,
        actual_full_N_gt8_B5_must_be_exercised_naturally=True,NPU_qualification_inherited=False,
        neural_qualified=False,performance_qualified=False,adopted=False)
    (P/'INTEGRATION_REVIEW260.json').write_text(json.dumps(review,indent=2)+'\n')
    for p in P.rglob('*'):
        if p.is_file():
            if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
            files[str(p)]=sha(p)
    for p in[summary,D/'build_ROOT260_spare_branch_group_budget.py',D/'ROOT260_GENERATION_TEMPLATE.py',Path(__file__)]:files[str(p)]=sha(p)
    for path,wanted in files.items():need(sha(path)==wanted,'all exact source bytes '+path)
    f=P/'FREEZE260.json'
    with f.open('x')as stream:json.dump(dict(schema='ROOT260.spare-branch-group-full-ownAR-frozen-sources.v1',UTC=review['UTC'],files=files,
        prior_frozen_sources_edited=False,original_models_sampler_workload_unchanged=True,NPU_qualification_inherited=False,
        same_instance227_plus72_independent_prefixes_required=True,full463_ownAR_required=True,adopted=False),stream,indent=2);stream.write('\n')
    members={**files,str(f):sha(f)}
    with tarfile.open(P/'SOURCE260.tar.gz','x:gz')as archive:
        for path,wanted in sorted(members.items()):
            data=Path(path).read_bytes();need(hashlib.sha256(data).hexdigest()==wanted,'source content during archiving')
            info=tarfile.TarInfo(path.lstrip('/'));info.mode=0o600;info.size=len(data);info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE260.tar.gz','r:gz')as archive:
        need(len(archive.getmembers())==len(members),'complete exact source inventory')
        for member in archive:
            need(member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name],'exact archived source bytes')
    print(json.dumps(dict(freeze=sha(f),archive=sha(P/'SOURCE260.tar.gz'),members=len(members))))
if __name__=='__main__':main()
