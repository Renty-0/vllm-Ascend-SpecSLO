"""Fresh scheduling candidate; keep all M120 mathematics and original gates."""
import ast,collections,json,pathlib,re
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-static-q5-M120-full463-own-AR-v256'
P=B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'
P.mkdir(exist_ok=True)
prefixes={'ROOT','_ROOT','V','SPECSLO_V','_canonical','_root','full','v',
    'FREEZE','SOURCE','FIXTURE','HOST_FULL','HOST_ADMISSION','INTEGRATION_REVIEW'}
prefixes.update(f.stem.removesuffix('256')for f in OLD.iterdir()if f.suffix in('.py','.sh'))
pattern=re.compile(r'\b('+ '|'.join(re.escape(s)for s in sorted(prefixes,key=len,reverse=True))+r')256(?=\b|_)')
def renamed(source):
    return pattern.sub(lambda match:match.group(1)+'257',source.replace(str(OLD),str(P)))
def ints(source):
    return collections.Counter(n.value for n in ast.walk(ast.parse(source))if isinstance(n,ast.Constant)and type(n.value)is int)
for f in OLD.iterdir():
    if not f.is_file()or f.suffix not in('.py','.sh')or f.name.startswith('check_'):continue
    name=f.name.replace('256','257')
    s=renamed(f.read_text())
    if f.suffix=='.py'and ints(s)!=ints(f.read_text()):raise RuntimeError('numeric constants changed by version rename '+name)
    # Retain the unchanged source implementation of page-reserved admission.
    # The new LIVE-only generation projection below adds just the SLO policy.
    if name=='generation257.py':continue
    (P/name).write_text(s)
(P/'FIXTURE257.json').write_bytes((OLD/'FIXTURE256.json').read_bytes())
# Link original256 module only where source routines need its pure projector.
for name in ('runtime257.py','guard257.py'):
    f=P/name;s=f.read_text();anchor="sys.path.extend(["
    if anchor not in s:raise RuntimeError('declared dependency import anchor')
    s=s.replace(anchor,anchor+"str(B/'tree-static-q5-M120-full463-own-AR-v256'),",1);f.write_text(s)
f=P/'startup257.sh';s=f.read_text();s=s.replace('export PYTHONPATH=','export PYTHONPATH='+str(OLD)+':');f.write_text(s)
# Same real CPU/Native/own-history integration checks, with actual LIVE flag.
s=renamed((OLD/'check_full256_HOST.py').read_text())
s=s.replace("P=Path(__file__).resolve().parent;B=P.parent", "P=Path(__file__).resolve().parent;B=P.parent\nsys.path.append(str(B/'tree-static-q5-M120-full463-own-AR-v256'))")
(P/'check_full257_HOST.py').write_text(s)
s=(OLD/'check_admission256_HOST.py').read_text().replace('generation256','generation257').replace('window256','window257').replace('HOST_ADMISSION256','HOST_ADMISSION257')
s=s.replace("P=Path(__file__).resolve().parent;B=P.parent", "P=Path(__file__).resolve().parent;B=P.parent\nsys.path.append(str(B/'tree-static-q5-M120-full463-own-AR-v256'))")
(P/'check_admission257_HOST.py').write_text(s)
f=P/'audit257.py';s=f.read_text();anchor="                rows = geometry[str(trace['source_generation'])]"
if s.count(anchor)!=1:raise RuntimeError('actual LIVE audit anchor')
s=s.replace(anchor,"                need(trace['decode_counters']['spec_rhythm_target_slo_first_enabled']==1, 'actual LIVE SLO-first policy counter')\n"+anchor);f.write_text(s)
f=P/'scope257.py';s=f.read_text();anchor='canonical_same_instance_cases=227, canonical_M=120,'
if s.count(anchor)!=1:raise RuntimeError('actual LIVE scope report anchor')
s=s.replace(anchor,anchor+'\n                original_global_environment_TARGET0_unchanged=True, actual_LIVE_target_SLO_first=True,');f.write_text(s)
(P/'generation257.py').write_bytes((B/'budget-diagnostics/ROOT257_GENERATION_TEMPLATE.py').read_bytes())
(P/'DECLARED_TARGET_SLO_FIRST257.json').write_text(json.dumps(dict(schema='ROOT257.live-target-SLO-first-policy-candidate.v1',
    model_backend_unchanged='M120/S24/KV2560, all original BF16 weights and sampler',
    original_startup_global_environment_TARGET0_unchanged=True,
    actual_live_build_plan_target_slo_first=True,actual_live_counter1_required=True,
    same_N24_C96_active48_global192=True,all_original_and_fresh227_gates_required=True,
    full463_ownAR_required=True,performance_qualified=False,adopted=False),indent=2)+'\n')
print(P)
