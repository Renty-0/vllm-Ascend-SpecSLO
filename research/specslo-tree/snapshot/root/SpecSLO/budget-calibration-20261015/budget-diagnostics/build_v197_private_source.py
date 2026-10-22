from pathlib import Path
import ast,difflib,hashlib,json
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';S=B/'tree-causal80-real16-bankfix-full-v173';M=B/'tree-vacancy-cohort-prefill-cap4-active32-full-v190';P=B/'tree-empty-active-cohort-causal80-active32-full-v197';G=D/'v197-root-execution'
P.mkdir(exist_ok=False);G.mkdir(exist_ok=False)
for oldname in ('base','source','progress','runtime','trace','audit','check','scope','component','extension','publication','prefill'):
 (P/f'{oldname}197.py').write_text((M/f'{oldname}190.py').read_text().replace('190','197'))
(P/'next4_197.py').write_text((M/'next4_190.py').read_text().replace('190','197'))
def edit(name,old,new,count=1):
 p=P/name;s=p.read_text();assert s.count(old)==count,(name,old,s.count(old));p.write_text(s.replace(old,new))
edit('base197.py',"CORE185=B/'tree-vacancy-cohort-prefill-full-v185'\n",'')
edit('base197.py','(BASE,ACTIVE,P173,CORE185)','(BASE,ACTIVE,P173)')
edit('base197.py',"ACTION='vacancy-cohort-prefill-cap4-active32-full463'","ACTION='empty-active-cohort-causal80-active32-full463'")
s=(P/'base197.py').read_text();s=s[:s.index('\ndef core_module(')];(P/'base197.py').write_text(s)
(P/'config197.py').write_text('''"""Exact original173 constructor; no admission/coalesce configuration mutation."""\nfrom startup_config_v132 import make_config\n''')
edit('trace197.py',"from base197 import core_module\n  state['source']=core_module('source_patch185').patch(adapt_source(builder(),active_window))","from empty197 import patch\n  state['source']=patch(adapt_source(builder(),active_window))")
edit('trace197.py'," from trace_admission197 import wrap\n return wrap(ns['run_trace']),state"," return ns['run_trace'],state")
(P/'extension197.py').write_text('''"""Only173 production parent; new empty-active admission source and current math."""\ndef prepare_install(output_directory):\n from base197 import parent_module\n from source197 import entry_namespace\n from progress197 import configure\n configure(output_directory)\n return entry_namespace(output_directory),parent_module('mapper173').controls_class()\ndef after_install(native,graph,live):\n from base197 import parent_module\n return dict(parent_module('extension173').after_install(native,graph,live),empty_active_refill_only=True,production_parent='frozen173',new_hot_collectives=0)\ndef loaded_origins():\n from base197 import parent_module,P,sha,read,require\n from pathlib import Path\n import sys\n original=parent_module('extension173').loaded_origins();pins=read(P/'FROZEN_SOURCE197.json')['files'];own={}\n for name,module in tuple(sys.modules.items()):\n  path=vars(module).get('__file__')if hasattr(module,'__dict__')else None\n  if type(path)is str and Path(path).is_absolute()and Path(path).resolve().is_relative_to(P):\n   p=Path(path).resolve();require(p.is_file()and pins.get(str(p))==sha(p),'ROOT-frozen own source '+name);own[str(p)]=sha(p)\n return dict(original=original,private197_sources=own,no_parent_source_mutation=True)\ndef qualify_component(engine,native,stage,policy,output_directory):\n from component197 import run\n return run(engine,native,stage,policy,output_directory)\n''')
# Exact original publication exposes original post-root scheduler states only to this component.
edit('publication197.py','  bank.contexts=[list(v.token_ids) for v in local];bank.reference_contexts=[p+[t] for p,t in zip(bank.prompts,tokens,strict=True)]','  bank.contexts=[list(v.token_ids) for v in local];bank.reference_contexts=[p+[t] for p,t in zip(bank.prompts,tokens,strict=True)]\n  bank.runtime_states=states;bank.publication_active=tuple(active)')
# Keep original same-M4-step draft oracle, explicitly map subset owner IDs.
edit('next4_197.py',"n=case['N'];require(n in (2,4) and case['budgets']==[4]*n,'finite true normalB4 N2/4 next scope')\n budgets=dict(enumerate(case['budgets']))","n=case['N'];require(n in (1,16) and case['budgets']==[4]*n and len(case['ids'])==n,'finite real-planner N1/16 B4 next subset')\n budgets=dict(zip(case['ids'],case['budgets'],strict=True))")
edit('next4_197.py','requested_normal=tuple(range(n))',"requested_normal=tuple(case['ids'])")
edit('next4_197.py','owner=i+n;p=len(values)-1;',"owner=bank.reference_ids[i];p=len(values)-1;")
edit('next4_197.py',"require(n in (2,4) and physical==n,'exact first next same physical M for N2/4')","require(n in (1,16) and physical==n,'exact first next same physical M for actual planner subset')")
edit('next4_197.py','bank.reference_contexts[i-n]','bank.reference_contexts[bank.reference_ids.index(i)]')
edit('next4_197.py',"before[i]['safe_before']","before[bank.ids[i]]['safe_before']")
edit('next4_197.py',"before[i]['tokens']","before[bank.ids[i]]['tokens']")
edit('next4_197.py',"a['ledger'][i]==r['ledger'][i+n]","a['ledger'][bank.ids[i]]==r['ledger'][bank.reference_ids[i]]")
edit('next4_197.py','independent_owners=list(range(n,2*n))','independent_owners=list(bank.reference_ids)')
# Wide real prompt batch: reuse 32 manifest prompts with different lengths, each independently prefetched.
edit('prefill197.py',"self.rows=rows[:n];self.prompts=", "self.rows=rows[:n];self.ids=list(range(n));self.reference_ids=list(range(n,2*n));self.prompts=")
edit('prefill197.py',"self.incumbent_tokens=list(rows[4]['token_ids'])","self.incumbent_tokens=list(rows[-1]['token_ids'])")
a=(P/'prefill197.py').read_text();start=a.index(' result=[];lengths=set()');end=a.index('\nclass Bank:',start)
a=a[:start]+''' result=[dict(request_id=r['request_id'],token_ids=list(t),length=len(t),original_prompt=r['prompt']) for r,t in zip(rows,values,strict=True)]
 require(len(result)==65 and len({v['length']for v in result[:32]})>1 and max(v['length']for v in result)<=2301,'real mixed-length original prompt owners')
 return result
''' + a[end:];(P/'prefill197.py').write_text(a)
# Original after-prefill nextdraft now runs only original controller's selected home, not an enlarged N32 target.
start=a.index(' # Each oracle prefix was computed');end=a.index('\n def final():',start)
a=a[:start]+''' # Planner-generated subset only: wide prefill does not enlarge target/draft scope.
 from subset197 import prepare,guard_rest,finish
 selected=lane('original-planner-next-subset',lambda:prepare(engine,native,bank))
 protected_rest=lane('protect-all-nonselected-owner-prefixes',lambda:guard_rest(engine,bank,selected))
 work=__import__('next4_197').generate(engine,native,selected,dict(N=selected.n,budgets=[4]*selected.n,C=4*selected.n,ids=selected.ids),1000+n+pass_index,lane)
 planning=lane('original-ready-ticket-target-admission',lambda:finish(engine,native,bank,selected,work,protected_rest))
''' +a[end:]
a=a.replace("next_draft=dict(rows=work['rows'],selected=work['selected'],draft_proof=work['draft_proof'])","next_draft=dict(rows=work['rows'],selected=work['selected'],draft_proof=work['draft_proof'],planner=planning)")
a=a.replace('for n in (2,4):','for n in (2,32):').replace('representative_sizes=[2,4]','representative_sizes=[2,32]')
# audit source literals adjusted below after inspection.
(P/'prefill197.py').write_text(a)
edit('scope197.py','representative_cohort_sizes=[2,4]','representative_cohort_sizes=[2,32]')
edit('scope197.py','minimum_requests=4,maximum_wait_ms=150.,minimum_is_not_maximum=True','empty_active_only=True,active_window32=True,constructor_coalesce_unchanged=True')
# Own original audit retains parent173 algorithm; append the new actual empty-cohort check.
p=P/'audit197.py';a=p.read_text();a=a[:a.index('\n\ndef audit(directory,selected_windows):')]+'''\n\ndef audit(directory,selected_windows):
 value=original_audit(directory,selected_windows)
 from prefill197 import audit as audit_prefill
 from empty197 import audit_trace
 extra=audit_prefill(Path(directory)/'multiowner-eager-prefill-own-current-process')
 reports=[read(Path(directory)/f'rank{r}-local-diagnostic.json') for r in range(4)]
 require(all(r['component']['multiowner_prefill']['audit']==extra for r in reports),'same current new multiowner math')
 decisions=[]
 from audit_production_v101 import load_artifact
 for i,env in enumerate(reports[1]['windows']['traces']):
  traces=[load_artifact(r['windows']['traces'][i]['original_trace']['artifact'],directory)for r in reports]
  for t in traces[1:]:require(t['standalone_prefill_calls']==traces[0]['standalone_prefill_calls'],'same all4 genuine prefill owner batches')
  decisions.append(dict(window=env['window'],role=env['role'],actual_empty_cohorts=audit_trace(traces[1])))
 return dict(value,current_multiowner_eager_component=extra,empty_active_cohorts=decisions,whole_AR_qualified=False)
''';p.write_text(a)
# All shell/supervisor context remains original190 reviewed independent guard, only private identity and truthful scope labels.
for name in ('guard190.py','launch190.sh','startup190.sh','GUARD_INPUTS190.json'):
 text=(D/'v190-root-execution'/name).read_text().replace('190','197').replace('tree-vacancy-cohort-prefill-cap4-active32-full-v197',P.name)
 if name.endswith('.py'):
  text=text.replace('vacancy_min_requests=4,vacancy_wait_ms=150.0,','empty_active_only=True,')
  text=text.replace('one full463 active32 cap4 new4/150 vacancy-cohort after current original B4,causal80 and multiowner eager/next4 math/twozero','one full463 active32 cap4 empty-active-cohort after current original B4,causal80 and N2/32 multiowner eager/planner-subset next4 math/twozero')
 (G/name.replace('190','197')).write_text(text)
(P/'SCOPE197.txt').write_text('Sole production parent173. Only new scheduling predicate: after original live receive, nonempty active returns before ordinary admission. Native configured64/queue463/storage527/pool544; actual active32; C96/E96/global192; target16; no SLO flag/budget/arrival/clock/publication changes. New multiowner N2/32 same-process finite math, planner-derived N1/16 next subsets; no wholeAR or alltuple claim. ROOT freeze and peer required before execution.\n')
print(P)
