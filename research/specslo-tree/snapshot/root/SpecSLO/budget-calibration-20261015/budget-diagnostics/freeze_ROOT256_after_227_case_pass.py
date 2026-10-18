"""Freeze the fresh full epoch only after the bounded227-case actual result."""
import ast,datetime,hashlib,json,pathlib,tarfile
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-static-q5-M120-full463-own-AR-v256'
MATH=B/'tree-static-q5-M120-node-math-v255'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='static-M120-root-v255-all227-node-cases-v1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def need(v,m):
    if not v:raise RuntimeError(m)
receipt=DONOR/('receipt-'+TAG+'.json');r=json.loads(receipt.read_text())
need(r['status']=='complete'and not r['errors']and r['sources_unchanged']and r['devices_after']['all_idle'],'bounded actual227 math must finish and four cards idle')
audit=D/'ROOT255_INDEPENDENT_ALL227_NODE_AUDIT.json';a=json.loads(audit.read_text())
need(a['receipt_sha256']==sha(receipt)and a['all3_targets_all227_bitwise']is True and a['all72_prefixes_unchanged']is True,'exact independently audited227 result')
for n in ['HOST_FULL256_PROTOCOL_RESULT.json','HOST_ADMISSION256_RESULT.json']:
    v=json.loads((P/n).read_text());need(v['status'].startswith('HOST_'),'HOST source/Native/admission gate')
protocol=json.loads((P/'HOST_FULL256_PROTOCOL_RESULT.json').read_text())
need(len(protocol['cases'])==227 and all(protocol[k]is True for k in ['actual_observer_cases227','draft_None_protocol','independent_AR_mask_reconstruction','complete_runtime_inverse','original_trace_defaults_preserved']),'complete full observer/default/reference integration')
f=MATH/'FREEZE255.json';files=dict(json.loads(f.read_text())['files']);files[str(f)]=sha(f)
review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT256.M120-full-integration-review.v1',
    prior_bounded227_audit=str(audit),prior_bounded227_audit_sha256=sha(audit),NPU_qualification_inherited=False,
    all_original_startup_and_components_required=True,new_same_instance227_node_cases_required=True,
    independent_prefill_owners=72,physical_M=120,actual_FIA_groups=24,actual_query_partition=[5]*24,
    linear_mask_capacity=10,branch_copy_capacity=11,KV_execution_extent=2560,
    real_target_N24_C96=True,original_target_N32_C96_Q128=True,global_draft_budget192=True,
    active48_with_page_reservation=True,original_workload463_and_models_arrivals_SLO_sampling_unchanged=True,
    original_warm_fixedpoint8_twozero_seal_full463_and_complete_ownAR_required=True,
    all_other_files_private=True,per_tag_limit_MiB=256,aggregate_private_guard_limit_MiB=1536,
    performance_qualified=False,whole_workload_AR_qualified=False,adopted=False)
(P/'INTEGRATION_REVIEW256.json').write_text(json.dumps(review,indent=2)+'\n')
for f in P.iterdir():
    need(f.name not in('FREEZE256.json','SOURCE256.tar.gz'),'fresh256 freeze, no rewrite')
    if f.suffix=='.py':ast.parse(f.read_text(),filename=str(f))
    if f.is_file():files[str(f)]=sha(f)
for f in(D/'build_ROOT256_M120_full.py',D/'audit_ROOT255_bounded_nodes_HOST.py',pathlib.Path(__file__),audit):files[str(f)]=sha(f)
f=P/'FREEZE256.json';f.write_text(json.dumps(dict(schema='ROOT256.M120-full-own-AR-frozen-sources.v1',UTC=review['UTC'],files=files,
    prior_frozen_sources_edited=False,original_models_sampler_workload_unchanged=True,
    same_instance227_plus72_independent_prefixes_required=True,full463_ownAR_required=True,
    NPU_qualification_inherited=False,adopted=False),indent=2)+'\n')
with tarfile.open(P/'SOURCE256.tar.gz','w:gz')as archive:
    for name in sorted([*files,str(f)]):archive.add(name,arcname=name.lstrip('/'),recursive=False)
print(json.dumps(dict(freeze=sha(f),archive=sha(P/'SOURCE256.tar.gz'),members=len(files)+1)))
