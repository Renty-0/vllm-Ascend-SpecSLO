"""New LIVE budget delegates; retain every frozen M120 model/reference gate."""
import ast,collections,json,re
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'
P=B/'tree-spare-branch-group-budget-full-v260';P.mkdir(exist_ok=True)
prefixes={'ROOT','_ROOT','V','SPECSLO_V','_canonical','_root','full','v','FREEZE','SOURCE','FIXTURE','HOST_FULL','HOST_ADMISSION','HOST_PRIORITY','INTEGRATION_REVIEW'}
prefixes.update(f.stem.removesuffix('257')for f in OLD.iterdir()if f.suffix in('.py','.sh'))
pattern=re.compile(r'\b('+ '|'.join(re.escape(s)for s in sorted(prefixes,key=len,reverse=True))+r')257(?=\b|_)')
def renamed(source):return pattern.sub(lambda m:m.group(1)+'260',source.replace(str(OLD),str(P)))
def ints(source):return collections.Counter(n.value for n in ast.walk(ast.parse(source))if isinstance(n,ast.Constant)and type(n.value)is int)
for f in OLD.iterdir():
    if not f.is_file()or f.suffix not in('.py','.sh')or f.name.startswith('check_'):continue
    name=f.name.replace('257','260');s=renamed(f.read_text())
    if f.suffix=='.py'and ints(s)!=ints(f.read_text()):raise RuntimeError('version rename changed numerical constants '+name)
    if name=='generation260.py':continue
    target=P/name;assert not target.exists();target.write_text(s)
(P/'generation260.py').write_bytes((D/'ROOT260_GENERATION_TEMPLATE.py').read_bytes())
(P/'FIXTURE260.json').write_bytes((OLD/'FIXTURE257.json').read_bytes())
for name in ('runtime260.py','guard260.py'):
    f=P/name;s=f.read_text();assert 'sys.path.extend(['in s;s=s.replace('sys.path.extend([','sys.path.extend([str(B/\'tree-static-q5-M120-SLO-first-full463-own-AR-v257\'),',1);f.write_text(s)
f=P/'startup260.sh';s=f.read_text().replace('export PYTHONPATH=','export PYTHONPATH='+str(OLD)+':');f.write_text(s)
for stem in ('full','admission','priority'):
    s=renamed((OLD/f'check_{stem}257_HOST.py').read_text())
    anchor='P=Path(__file__).resolve().parent;B=P.parent'
    assert anchor in s;s=s.replace(anchor,anchor+'\nsys.path.append('+repr(str(OLD))+')',1)
    if stem=='priority':
        s=s.replace("ast.unparse(n.func)=='controller.build_plan'","ast.unparse(n.func)=='_root260_build_plan'")
    (P/f'check_{stem}260_HOST.py').write_text(s)
f=P/'scope260.py';s=f.read_text();anchor="    permission = getattr(engine, '_canonical260_permission', None)";assert s.count(anchor)==1
s=s.replace(anchor,anchor+"\n    import budget260\n    if permission is not None:\n        need(permission['budget_shape'] is budget260.shape and permission['target_plan'] is budget260.build_plan, 'same LIVE group-budget delegates')",1)
anchor="greedy=engine.graph_runner.run_target_greedy)";assert s.count(anchor)==1
s=s.replace(anchor,"greedy=engine.graph_runner.run_target_greedy,\n                                         budget_shape=__import__('budget260').shape, target_plan=__import__('budget260').build_plan)",1)
anchor='canonical_same_instance_cases=227, canonical_M=120,';assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n                actual_LIVE_spare_B5_group_policy=True, actual_target_primary_side_group_budget=24,\n                inherited_original_cap4_wrapper_not_called_by_this_LIVE_shape=True,',1);f.write_text(s)
f=P/'audit260.py';s=f.read_text();anchor="                need(trace['decode_counters']['spec_rhythm_target_slo_first_enabled']==1, 'actual LIVE SLO-first policy counter')";assert s.count(anchor)==1
s=s.replace(anchor,anchor+"\n                need(trace['decode_counters']['spec_rhythm_ROOT260_group_budget_enabled']==1, 'actual LIVE spare branch group policy counter')",1)
anchor="                calls = trace['normal_target_calls']";assert s.count(anchor)==1
s=s.replace(anchor,anchor+"\n                if trace['scope_count']==463:\n                    need(any(call['N']>8 and 5 in call['budgets'] for call in calls), 'actual larger-N B5 scope exercised naturally')",1);f.write_text(s)
(P/'DECLARED_GROUP_BUDGET260.json').write_text(json.dumps(dict(schema='ROOT260.spare-branch-group-budget.v1',
    same_model_backend='M120/S24/KV2560, original BF16 weights, original greedy and own-AR',
    changed='LIVE-only original typed allocator per-owner caps and whole-ticket physical group budget',
    original_scalar_gains_needs_priority_charges_and_publication_preserved=True,
    C96=True,global_draft192=True,active48=True,N_plus_B5_le24=True,
    original_startup_components_and_fresh227_required=True,full463_independent_ownAR_required=True,
    packet258_or_deferred259_installed=False,performance_qualified=False,adopted=False),indent=2)+'\n')
for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
print(P)
