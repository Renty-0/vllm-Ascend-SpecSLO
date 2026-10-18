"""Freeze the scheduling-only candidate after its source and controller checks."""
import ast,datetime,hashlib,json,pathlib,tarfile
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-static-q5-M120-full463-own-AR-v256'
P=B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def need(v,m):
    if not v:raise RuntimeError(m)
summary=D/'ROOT256_COMPLETE_RESULT_FOR_NEXT_POLICY.json';prior=json.loads(summary.read_text())
need(prior['status']=='complete_independently_audited'and prior['all463_equal_on_all3targets'],'complete prior correct full model and ownAR')
for name in ['HOST_FULL257_PROTOCOL_RESULT.json','HOST_ADMISSION257_RESULT.json','HOST_PRIORITY257_RESULT.json']:
    v=json.loads((P/name).read_text());need(v['status'].startswith('HOST_'),'actual HOST gate '+name)
v=json.loads((P/'HOST_PRIORITY257_RESULT.json').read_text());need(v['new_SLO_first_selected']==[0,1]and v['whole_ready_tickets_preserved']and v['live_projection']['truthful_actual_policy_counter']==1,'actual original controller enforces SLO-first within same budget')
f=OLD/'FREEZE256.json';files=dict(json.loads(f.read_text())['files']);files[str(f)]=sha(f)
review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT257.live-SLO-first-integration-review.v1',
    original256_full_correct_result_summary=str(summary),original256_full_summary_sha256=sha(summary),
    actual_model_implementation='immutable255 M120/S24/KV2560; identical weights/sampler/geometry',
    original_global_environment_TARGET0_unchanged=True,only_LIVE_build_plan_target_slo_first_and_counter_changed=True,
    actual_LIVE_target_SLO_first=True,actual_LIVE_policy_counter=1,whole_ticket_and_budget_checks_preserved=True,
    real_N24_C96=True,active48=True,global_draft192=True,original463_arrivals_SLO_and_output256_unchanged=True,
    all_original_startup_and_components_required=True,fresh_same_instance227_and72_independent_prefixes_required=True,
    original_warm_twozero_seal_full463_and_complete_ownAR_required=True,NPU_qualification_inherited=False,
    performance_qualified=False,whole_workload_AR_qualified=False,adopted=False)
(P/'INTEGRATION_REVIEW257.json').write_text(json.dumps(review,indent=2)+'\n')
for f in P.iterdir():
    need(f.name not in('FREEZE257.json','SOURCE257.tar.gz'),'fresh source epoch, no overwrite')
    if f.suffix=='.py':ast.parse(f.read_text(),filename=str(f))
    if f.is_file():files[str(f)]=sha(f)
for f in[summary,D/'build_ROOT257_live_target_SLO_first.py',D/'ROOT257_GENERATION_TEMPLATE.py',pathlib.Path(__file__)]:files[str(f)]=sha(f)
f=P/'FREEZE257.json';f.write_text(json.dumps(dict(schema='ROOT257.live-SLO-first-full-ownAR-frozen-sources.v1',UTC=review['UTC'],files=files,
    prior_frozen_sources_edited=False,original_models_sampler_workload_unchanged=True,NPU_qualification_inherited=False,
    same_instance227_plus72_independent_prefixes_required=True,full463_ownAR_required=True,adopted=False),indent=2)+'\n')
with tarfile.open(P/'SOURCE257.tar.gz','w:gz')as archive:
    for name in sorted([*files,str(f)]):archive.add(name,arcname=name.lstrip('/'),recursive=False)
print(json.dumps(dict(freeze=sha(f),archive=sha(P/'SOURCE257.tar.gz'),members=len(files)+1)))
