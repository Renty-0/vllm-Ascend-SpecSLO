"""Create only private190 source; frozen173/185/common sources stay read-only."""
from pathlib import Path
import difflib,hashlib,json
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-vacancy-cohort-prefill-cap4-active32-full-v190';G=D/'v190-root-execution'
S=B/'tree-causal80-real16-bankfix-full-v173';V=B/'tree-vacancy-cohort-prefill-full-v185';BASE=B/'tree-high-normal-cap4-native-families-v148'
rows=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,text):
 p=P/name
 if p.exists():raise RuntimeError('private source already exists '+name)
 p.write_text(text)
def derive(root,old,new,oldns,newns,changes=()):
 original=(root/old).read_text();text=original.replace(oldns,newns)
 for before,after,count in changes:
  if text.count(before)!=count:raise RuntimeError((new,before,text.count(before),count))
  text=text.replace(before,after)
 transformed=original.replace(oldns,newns);ops=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(None,transformed,text,autojunk=False).get_opcodes():
  if tag!='equal':ops.append(dict(start=a,end=b,old=transformed[a:b],new=text[c:d]))
 write(new,text);rows.append(dict(file=new,parent=str(root/old),parent_sha256=sha(root/old),namespace=[oldns,newns],deltas=ops))

# Explicit original173 production namespace; only new private IO and admission.
derive(V,'base185.py','base190.py','185','190',(
 ("P180=B/'tree-uniform-b3-three-bucket-full-v180'","P173=B/'tree-causal80-real16-bankfix-full-v173'\nCORE185=B/'tree-vacancy-cohort-prefill-full-v185'",1),
 ('(BASE,ACTIVE,P180)','(BASE,ACTIVE,P173,CORE185)',1),
 ("P180/(name+'.py')","P173/(name+'.py')",2),('readonly parent180 module','readonly production parent173 module',1),
 ("ACTION='vacancy-cohort-prefill-active32-full463'","ACTION='vacancy-cohort-prefill-cap4-active32-full463'",1)))
for name in ('source','progress'):
 derive(S,name+'173.py',name+'190.py','173','190')
derive(S,'runtime173.py','runtime190.py','173','190',(
 ('engine._v190_actual_startup=startup','engine._v190_actual_startup=startup\n     engine._v173_actual_startup=startup',1),
 ('from startup_config_v132 import make_config','from config190 import make_config',1)))
derive(S,'trace173.py','trace190.py','173','190',(
 ("state['source']=adapt_source(builder(),active_window)","from base190 import core_module\n  state['source']=core_module('source_patch185').patch(adapt_source(builder(),active_window))",1),
 ("return ns['run_trace'],state","from trace_admission190 import wrap\n return wrap(ns['run_trace']),state",1)))
derive(V,'config185.py','config190.py','185','190')
derive(V,'publication185.py','publication190.py','185','190')
derive(V,'trace_admission185.py','trace_admission190.py','185','190',(
 ('engine._v190_admission_generations','engine._v185_admission_generations',1),
 ('engine._v190_current_cohort','engine._v185_current_cohort',1),
 ('from cohort190 import VacancyCohort',"from base190 import core_module\n VacancyCohort=core_module('cohort185').VacancyCohort",1)))
derive(V,'prefill185.py','prefill190.py','185','190',(
 ("parent_module('draft180').generate(engine,native,bank,dict(N=n,budgets=[3]*n)","__import__('next4_190').generate(engine,native,bank,dict(N=n,budgets=[4]*n,C=4*n)",1),
 ('real B3 chain','real B4 chain',1),
 ("nextrow['selected']==[list(range(3))]*n","nextrow['selected']==[list(range(4))]*n",1),
 ('all(len(v)==3 for v in nextrow[\'rows\'])','all(len(v)==4 for v in nextrow[\'rows\'])',1),
 ('genuine next B3 publication','genuine next B4 publication',1),
 ("d['exact_full6_encoded']","d['exact_full8_encoded']",1),
 ("d['actual_draft_model_stages']==3 and d['raw_encoded_shape'][1:]==[3,2] and d['canonical8_zero_slots']==[3,7]","d['actual_draft_model_stages']==4 and d['raw_encoded_shape'][1:]==[4,2]",1),
 ('actual sameM three-step next','actual sameM four-step next',1)))

# Exact original B4 actor math, parameterized only for the two fresh prompt
# cohort sizes. This first nextdraft has no missing committed suffix.
derive(BASE,'draft_v146.py','next4_190.py','from scope_v146 import require,digest','from base190 import require\nfrom scope_v146 import digest',(
 ("budgets=dict(enumerate(case['budgets']))","n=case['N'];require(n in (2,4) and case['budgets']==[4]*n,'finite true normalB4 N2/4 next scope')\n budgets=dict(enumerate(case['budgets']))",1),
 ('requested_normal=tuple(range(32))','requested_normal=tuple(range(n))',1),
 ('result=[]\n for i,values','result=[];n=bank.n\n for i,values',1),
 ('owner=i+32','owner=i+n',1),
 ('entries=reference_entries(bank);repair=','n=bank.n;entries=reference_entries(bank);repair=',1),
 ('physical=native._next_linear_draft_graph_bucket(sum(len(v) for _,_,_,v in entries),128) if repair else 32',
  "require(not repair,'fresh full-prefill next has no missing committed suffix')\n physical=native._next_linear_draft_graph_bucket(n,engine.config.max_num_seqs)",1),
 ("require(physical==32 if not repair else physical in (36,40,44,48,64),'declared real grouped repair geometry')","require(n in (2,4) and physical==n,'exact first next same physical M for N2/4')",1),
 ('bank.reference_contexts[i-32]','bank.reference_contexts[i-n]',1),
 ('first.new_zeros((32,','first.new_zeros((physical,',1),
 ('[0]*32,3,32','[0]*n,3,physical',1),
 ("source=stage('source',bank.source);plans=","n=case['N'];source=stage('source',bank.source);plans=",1),
 ('[False]*32,plans','[False]*n,plans',1),
 ("tuple(a['encoded'].shape)==(32,4,2)","tuple(a['encoded'].shape)==(n,4,2)",1),
 ("r['ledger'][i+32]","r['ledger'][i+n]",1),
 ('raw_encoded_shape=[32,4,2],raw_pool_nodes=256','raw_encoded_shape=[n,4,2],raw_pool_nodes=8*n',1),
 ('independent_owners=list(range(32,64))','independent_owners=list(range(n,2*n))',1),
 ('[None]*32 if engine.is_draft','[None]*n if engine.is_draft',1)))

# Audit keeps the original173 causal80 geometry/accounting, then reopens new
# multiowner files and recorded decision inputs using unchanged scalar policy.
derive(S,'audit173.py','audit190.py','173','190',(
 ('def audit(directory,selected_windows):','def original_audit(directory,selected_windows):',1),
 ('from audit80190 import audit as audit80','from audit80173 import audit as audit80',1),
 ('from mapper190 import expected_geometry','from mapper173 import expected_geometry',1)))
with (P/'audit190.py').open('a')as f:f.write("\n\ndef audit(directory,selected_windows):\n value=original_audit(directory,selected_windows)\n from prefill190 import audit as audit_prefill\n from trace_admission190 import audit_traces\n extra=audit_prefill(Path(directory)/'multiowner-eager-prefill-own-current-process')\n reports=[read(Path(directory)/f'rank{r}-local-diagnostic.json') for r in range(4)]\n require(all(r['component']['multiowner_prefill']['audit']==extra for r in reports),'same current new multiowner math')\n return dict(value,current_multiowner_eager_component=extra,admission190=audit_traces(reports),minimum_requests_is_not_maximum=True)\n")
rows[-1]['append_text']=(P/'audit190.py').read_text().split('\n\ndef audit(directory,selected_windows):',1)[1]
rows[-1]['append_text']='\n\ndef audit(directory,selected_windows):'+rows[-1]['append_text']
write('PROJECTIONS190.json',json.dumps(dict(schema='V190.explicit-source-projections.v1',production_parent=str(S),readonly_admission_core=str(V),projection=rows),sort_keys=True,indent=2)+'\n')
