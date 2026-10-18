"""Freeze the LIVE budget candidate with all prior model proofs as inputs only."""
import ast,datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-spare-branch-group-budget-full-v260'
P=B/'tree-restored-cost-repair-byte-KV-full-v266'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
    if not v:raise RuntimeError(m)
def main():
    need(not(P/'FREEZE266.json').exists()and not(P/'SOURCE266.tar.gz').exists(),'fresh immutable epoch')
    summary=D/'ROOT260_COMPLETE_COMPUTE_AND_FAILED_COST_REPAIR_HANDOFF.json'
    prior=json.loads(summary.read_text())
    for name in ['HOST_FULL266_PROTOCOL_RESULT.json','HOST_ADMISSION266_RESULT.json','HOST_PRIORITY266_RESULT.json','HOST_COST266_RESULT.json','HOST_PREPARE266_RESULT.json']:
        v=json.loads((P/name).read_text());need(v['status'].startswith('HOST_'),'actual HOST gate '+name)
    v=json.loads((P/'HOST_COST266_RESULT.json').read_text());need(v['case_count']==160 and v['all_per_generation_audits_passed'] and v['actual_repaired_cases'],'actual bound cost repair')
    v=json.loads((P/'HOST_PREPARE266_RESULT.json').read_text());need(v['cases']==227 and v['actual_checked_branch_layouts']==172,'all real private prepare cases')
    audit=D/'ROOT265_INDEPENDENT_RANK3_THREE_ARM_COPY_AUDIT.json'
    need(json.loads(audit.read_text())['status']=='independent_actual_rank3_three_arm_KV_copy_audit_passed','actual rank3 copy probe and independent audit')
    f=OLD/'FREEZE260.json';need(sha(f)=='64c57c99cbb7026aa4f92e18c61dd2fe087aec975a697da0a9cb46c0184960e1','exact prior freeze')
    files=dict(json.loads(f.read_text())['files']);files[str(f)]=sha(f)
    for name in ['tree-fused-byte-branch-KV-copy-v263','tree-native-rank3-KV-copy-cost-v265']:
        num='263' if name.endswith('263') else '265';q=B/name/('FREEZE'+num+'.json')
        files.update(json.loads(q.read_text())['files']);files[str(q)]=sha(q)
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT266.spare-branch-group-budget-integration.v1',
        previous260_failed_cost_repair_summary=str(summary),previous260_summary_sha256=sha(summary),
        actual_model_backend='Original255 M120/S24/KV2560 BF16 neural maths and greedy; branch gather/scatter replaced by checked263 raw-byte copy',
        LIVE_shape='Actual original218 bound repair solver with only its original_solver cell replaced by proven260 group-cap allocator; original control/state/code retained',
        LIVE_target='Original full controller with added N+B5<=24 admission; whole tickets, original SLO-first order and deferred records retained',
        original_class_shape_and_controller_methods_unmodified=True,
        scalar_gain_need_charge_refinement_and_typed_publication_unchanged=True,
        actual_cost_repair_160_HOST_cases=True, actual_prepare_227_HOST_cases=True, same_current_218_generation_state=True, raw_byte_copy_no_arithmetic=True,C96=True,global_draft192=True,active48=True,
        original_global_environment_TARGET0_unchanged=True,actual_LIVE_target_SLO_first=True,
        packet258_and_deferred259_not_installed=True,original463_arrivals_SLO_and_output256=True,
        all_original_startup_components_required=True,fresh_same_instance227_and72_prefixes_required=True,
        original_warm_twozero_seal_full463_and_complete_ownAR_required=True,
        actual_full_N_gt8_B5_must_be_exercised_naturally=True,NPU_qualification_inherited=False,
        neural_qualified=False,performance_qualified=False,adopted=False)
    (P/'INTEGRATION_REVIEW266.json').write_text(json.dumps(review,indent=2)+'\n')
    for p in P.rglob('*'):
        if p.is_file():
            if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
            files[str(p)]=sha(p)
    for p in[summary,audit,D/'ROOT264_INDEPENDENT_SYNTHETIC_FIA_COST_AUDIT.json',D/'audit_ROOT264_265_operator_probes_HOST.py',D/'build_ROOT266_restore_repair_and_byte_copy_HOST.py',Path(__file__)]:files[str(p)]=sha(p)
    for path,wanted in files.items():need(sha(path)==wanted,'all exact source bytes '+path)
    f=P/'FREEZE266.json'
    with f.open('x')as stream:json.dump(dict(schema='ROOT266.spare-branch-group-full-ownAR-frozen-sources.v1',UTC=review['UTC'],files=files,
        prior_frozen_sources_edited=False,original_models_sampler_workload_unchanged=True,NPU_qualification_inherited=False,
        same_instance227_plus72_independent_prefixes_required=True,full463_ownAR_required=True,adopted=False),stream,indent=2);stream.write('\n')
    members={**files,str(f):sha(f)}
    with tarfile.open(P/'SOURCE266.tar.gz','x:gz')as archive:
        for path,wanted in sorted(members.items()):
            data=Path(path).read_bytes();need(hashlib.sha256(data).hexdigest()==wanted,'source content during archiving')
            info=tarfile.TarInfo(path.lstrip('/'));info.mode=0o600;info.size=len(data);info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    with tarfile.open(P/'SOURCE266.tar.gz','r:gz')as archive:
        need(len(archive.getmembers())==len(members),'complete exact source inventory')
        for member in archive:
            need(member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name],'exact archived source bytes')
    print(json.dumps(dict(freeze=sha(f),archive=sha(P/'SOURCE266.tar.gz'),members=len(members))))
if __name__=='__main__':main()
