"""Frozen diagnostic copies of231 primitives, before full-run integration."""
import ast,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';OLD=B/'tree-M80-FULL9-numeric-isolation-diagnostic-v230';PRIMITIVES=B/'tree-canonical-M80-FULL9-full463-own-AR-v231';P=B/'tree-canonical-M80-independent-branch-math-v232'
P.mkdir(exist_ok=False)
ACTION='M80-FULL9-independent-branch-math'
for name in('check230.py','runtime230.py','guard230.py','shutdown230.py','launch230.sh','startup230.sh'):
    source=(OLD/name).read_text().replace(str(OLD),str(P)).replace('230','232').replace('canonical-FULL-numeric-body-cost',ACTION)
    (P/name.replace('230','232')).write_text(source)
helpers=('mapper231.py','padding231.py','root_padding_forward231.py','metadata231.py','query1_reference231.py','reference_schedule231.py','bank231.py','neural_oracle231.py')
bindings={}
for name in helpers:
    p=PRIMITIVES/name;raw=p.read_bytes();(P/name).write_bytes(raw);bindings[name]=dict(original=str(p),sha256=hashlib.sha256(raw).hexdigest())
for name in('HOST_MAPPER231_RESULT.json','CPU_REFERENCE_SCHEDULE231_RESULT.json','CPU_BANK_PADDING231_RESULT.json'):
    (P/name).write_bytes((PRIMITIVES/name).read_bytes())
fixture=json.loads((OLD/'FIXTURE230.json').read_text());prompts=fixture['history']['cases'][0]['prompts'][:16]
prefixes=[(list(row)*((127+len(row)-1)//len(row)))[:127]for row in prompts]
(P/'FIXTURE232.json').write_text(json.dumps(dict(schema='ROOT232.synthetic-neural-math-fixture.v1',prefixes=prefixes,source=str(OLD/'FIXTURE230.json'),source_sha256=hashlib.sha256((OLD/'FIXTURE230.json').read_bytes()).hexdigest(),
    scope='sixteen valid synthetic127-token prefixes, truncating/repeating original tokenized prompts; not workload requests or a gain calibration dataset'),indent=2)+'\n')
(P/'PRIMITIVE_SOURCE_BINDING232.json').write_text(json.dumps(dict(status='copied_source_identity_checked',helpers=bindings,source_originals_need_not_be_mutable_runtime_dependencies=True,actual_runtime_uses_copied_private232_files=True),indent=2)+'\n')
p=P/'runtime232.py';s=p.read_text();a="            entry['install'](fast=True,host=True,transport=True,replay=True,repair=True,prefill=False)";b="            from install232 import install_entry\n            install_entry(entry,native)";assert s.count(a)==1;s=s.replace(a,b).replace('numeric_cases=4','numeric_cases=151');p.write_text(s)
p=P/'guard232.py';s=p.read_text();a=s.index('def actual_audit(');z=s.index('\ndef execution_namespace()',a)
audit='''def actual_audit(directory):
    from neural_oracle231 import cases
    expected=cases();directory=Path(directory);reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    require(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all four bounded mathematical diagnostic roles exited')
    require(read(directory/'completed.json')==dict(replies=[['calibrated',r,dict(numeric_cases=151,performance=False)]for r in range(4)],performance=False,adopted=False),'all four mathematical role replies')
    heads=[]
    for rank,report in enumerate(reports):
        require(report['rank']==rank and report['original_greedy_sampler_unchanged']is True and report['whole_workload_AR_qualified']is False,'explicit original-sampler bounded-only report')
        math=report['math'];require(len(math['cases'])==len(expected)==151,'complete predeclared151 neural cases')
        if rank:
            bank=report['bank'];require(bank['independent_neural_prefills']==48 and bank['all_triplet_roots_equal']and bank['all_triplet_prefix_KV_equal']and bank['candidate_KV_copied']is False,'all48 neural prefills and independent triplet prefix identity')
            require(bank['lease']['unique_pages']==160 and math['all_independent_prefixes_unchanged']is True,'bounded actual160page lease and all independent prefixes protected')
            for record,case in zip(math['cases'],expected,strict=True):
                require(record['case']==case and record['exact_greedy']is True and record['all128_KV_original_matcher_passed']is True and record['candidate_KV_copied']is False,'actual independent case math')
                require(record['candidate_tokens']==record['reference_tokens']and len(record['candidate_tokens'])==case['query_count'],'complete actual real predictions')
            heads.append([v['candidate_tokens']for v in math['cases']])
    require(heads[0]==heads[1]==heads[2],'all three target ranks agree on every qualified real-node head')
    return dict(status='bounded_M80_independent_branch_math_passed',cases=151,ranks=4,real_N=list(range(1,17)),original_sampler_unchanged=True,teacher_forced_neural_fixture_only=True,whole_workload_AR_qualified=False,production_qualified=False,formal_Goodput=False,adopted=False),'diagnostic_only',dict(independent_node_math_only=True,original_startup201_44_114_not_run=True)
'''
s=s[:a]+audit+s[z:];s=s.replace('four contexts, canonical M80 FULL9 segments','151 independent tree-node math cases').replace('existing q5 FULL9 body replay cost and eager M/Q ablations','independent own-prefix branch math; no performance timing')
p.write_text(s)
for p in P.glob('*.py'):ast.parse(p.read_text())
(P/'ROOT_SOURCE_REVIEW232.json').write_text(json.dumps(dict(status='source_prepared_CPU_proofs_copied_device_unrun',independent_target_math_cases=151,NN_prefixes=48,real_candidate_N=list(range(1,17)),
    model_rows=80,physical_segments=[14,15,16,17,18,22],new_root_only_dummy_for_Q79=True,real_candidate_budgets_unchanged=True,
    original_Native_capture_and_matcher_retained=True,original_sampler_unchanged=True,original_startup201_44_114_not_run=True,
    no_serving_workload=True,no_Goodput_or_latency_measurement=True,production_qualified=False,adopted=False),indent=2)+'\n')
print(json.dumps(dict(directory=str(P),source_prepared=True,NPU_started=False)))
