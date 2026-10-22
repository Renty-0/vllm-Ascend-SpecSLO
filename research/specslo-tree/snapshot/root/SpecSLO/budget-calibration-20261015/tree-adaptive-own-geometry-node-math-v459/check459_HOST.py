"""HOST symbolic schedules and actual source construction; no neural result."""
from pathlib import Path
from types import SimpleNamespace,CodeType
import argparse,ast,copy,datetime,importlib.abc,json,shlex,sys
P=Path(__file__).resolve().parent;B=P.parent
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST459 no backend import')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env459 import setup
setup()
for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
import oracle459,oracle377,oracle255,reference_schedule459,reference_schedule255,frames459
from prefixes376 import mid_prefixes,short_prefixes
fixture=json.loads((B/'tree-single-prepare-target-math-cost-v311/FIXTURE311.json').read_text())
cases=oracle459.cases();assert len(cases)==231 and cases==oracle377.cases()
class HostList(list):
    def tolist(self):return list(self)
def plans_for(case,prefixes):
    plans=[];rows=[]
    for i,b in enumerate(case['budgets']):
        prefix=len(prefixes[i]);parents=[-1,0,1,2][:min(b,4)]
        positions=[prefix]+[prefix+j+1 for j in range(min(b,4))]
        chosen=list(range(min(b,4)))
        if b==5:
            parents.append(case['anchor']-1);positions.append(prefix+case['anchor']+1);chosen.append(4+case['anchor'])
        plans.append(SimpleNamespace(parent_indices=HostList(parents),positions=HostList(positions),candidate_budget=b))
        rows.append([103+17*i+29*j for j in chosen])
    return plans,rows
family_counts={80:0,120:0};schedules=0
for name,prefixes in [('mixed',fixture['prefixes']),('mid',mid_prefixes(fixture['prefixes'])),('short',short_prefixes(fixture['prefixes']))]:
    for case in cases:
        plans,rows=plans_for(case,prefixes);roots=[1000+i for i in range(case['N'])]
        expected={(i,q)for i,b in enumerate(case['budgets'])for q in range(b+1)}
        own=reference_schedule459.schedule(plans,roots,rows,physical_M=case['M'])
        draft=reference_schedule255.schedule(plans,[0]*case['N'],rows)
        assert[(f['depth'],f['chunk'])for f in own]==[(f['depth'],f['chunk'])for f in draft]
        observed=[]
        for frame in own:
            assert frame['physical_M']==case['M']and len(frame['tokens'])==case['M']
            assert len(frame['selected'])==case['M']//5 and len(set(frame['selected']))==len(frame['selected'])
            assert frame['actual_seq_lengths_q']==list(range(5,case['M']+1,5))
            for row in frame['real_queries']:
                assert 24<=row['reference_owner']<72
                assert frame['ids'][row['head_row']]==row['reference_owner']
                assert frame['tokens'][row['head_row']]==row['token']
                if row['candidate_query']is not None:observed.append(tuple(row['candidate_query']))
        assert set(observed)==expected and len(observed)==len(expected)==case['query_count']
        schedules+=1;family_counts[case['M']]+=1
for rows in(80,120):frames459.framed(rows);reference_schedule459.framed(rows)
for rows in(0,79,81,160):
    try:frames459.framed(rows)
    except ValueError:pass
    else:raise AssertionError('HOST459 invalid reference geometry accepted')
bad=dict(N=17,budgets=[4]*17,anchor=0);plans,rows=plans_for(bad,fixture['prefixes'])
try:reference_schedule459.schedule(plans,[0]*17,rows,physical_M=80)
except ValueError:pass
else:raise AssertionError('HOST459 reference capacity overflow accepted')
oracle459.reference_function()
before=oracle377.specialized_run();after=oracle459.specialized_run()
def same_code(a,b):
    assert a.co_code==b.co_code and a.co_names==b.co_names and a.co_freevars==b.co_freevars
    assert len(a.co_consts)==len(b.co_consts)
    for x,y in zip(a.co_consts,b.co_consts,strict=True):
        if isinstance(x,CodeType):same_code(x,y)
        else:assert x==y
same_code(before.__code__,after.__code__)
assert after.__globals__['reference']is oracle459.reference and after.__globals__['candidate']is oracle377.candidate
import benchmark459,probe459,audit459,probe378
benchmark459.projected()
from source377 import function
function(probe378.run,(
    ('from oracle377 import run as neural, cases','from oracle459 import run as neural, cases'),
    ('from benchmark378 import run as costs','from benchmark459 import run as costs'),
    ('independent_reference_FULL120=True,','independent_reference_FULL120=False,independent_reference_same_physical_M=True,'),
))
function(audit459.original.panel,(("report['outputs_equal_between_modes'] is True","report['outputs_equal_between_modes'] is None"),))
import runtime459,runtime311,guard459
tree=ast.parse(runtime459.worker_source());nodes=[copy.deepcopy(n)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
namespace=dict(vars(runtime311));exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(P/'runtime459.py')+'::HOST-actual-lifecycle','exec'),namespace)
class StopAfterParse(Exception):pass
old_parse=argparse.ArgumentParser.parse_args;old_argv=sys.argv[:];seen=[]
def stop(parser,*args,**kwargs):
    value=old_parse(parser,*args,**kwargs);seen.append(vars(value));raise StopAfterParse()
try:
    argparse.ArgumentParser.parse_args=stop;sys.argv=['runtime459.py','--action',runtime459.ACTION,'--tag','HOST459-ACTUAL-PARSER']
    try:namespace['main']()
    except StopAfterParse:pass
finally:argparse.ArgumentParser.parse_args=old_parse;sys.argv=old_argv
assert seen==[dict(action=runtime459.ACTION,tag='HOST459-ACTUAL-PARSER')]
worker=next(n for n in nodes if n.name=='worker')
predicates=[n.test for n in ast.walk(worker)if isinstance(n,ast.If)and'completed or'in ast.unparse(n.test)]
assert len(predicates)==1
predicate=compile(ast.Expression(body=predicates[0]),'<HOST459-actual-worker-predicate>','eval')
for completed,mode,action,rejected in[(False,'calibrate',runtime459.ACTION,False),(True,'calibrate',runtime459.ACTION,True),(False,'calibrate','M80-own-geometry-whole-model-reference-cost',True),(False,'wrong',runtime459.ACTION,True)]:
    assert eval(predicate,dict(completed=completed,mode=mode,action=action))is rejected
supervisor=guard459.execution_namespace();assert supervisor['execute_locked'].__globals__ is supervisor
assert supervisor['ACTION']==runtime459.ACTION and supervisor['SCOPE']['finite_cases']==693
assert supervisor['SCOPE']['independent_prefixes']==216
def exports(path):
    result={}
    for line in path.read_text().splitlines():
        if line.startswith('export '):
            for item in shlex.split(line[7:]):
                if item.startswith('VLLM_ASCEND_')and'='in item:
                    k,v=item.split('=',1);result[k]=v
    return result
assert exports(P/'startup459.sh')==exports(B/'tree-M80-own-geometry-whole-model-v457/startup457.sh')
assert'448'in exports(P/'startup459.sh')['VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS'].split(',')
result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST459_SYMBOLIC693_REFERENCE_SOURCE_AND_ACTUAL_LIFECYCLE',
    symbolic_schedules=schedules,family_counts=family_counts,original231_tree_cases_preserved=True,
    original_greedy_Native_and_all_bitwise_comparator_code_identical=True,all_original_bank_offsets24_48_preserved=True,
    all_independent_reference_nodes_exactly_once=True,draft_stage_enumeration_matches_all_cases=True,
    reference_body_full_AST_inverse_passed=True,original_numerical_startup_exports_equal=True,
    actual_main_parser_exercised=True,actual_worker_predicates_exercised=4,actual_owned_supervisor_constructed=True,
    model_constructors=0,NPU_used=False,NN_permission=False,Raw=None,Goodput=None)
with(P/'HOST459_CONSTRUCTION_RESULT.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result))
