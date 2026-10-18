"""Create an explicit M120/24q5 prototype from immutable prior sources."""
import ast,datetime,hashlib,json,pathlib
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-static-q5-M128-full463-own-AR-v251'
MAP=B/'tree-static-q5-M128-projection-host-v249'
P=B/'tree-static-q5-M120-node-math-v255'
P.mkdir(exist_ok=False)
changes=[]
def write(name,s,replacements=()):
    for before,after in replacements:
        count=s.count(before)
        if count<1:raise RuntimeError('missing explicit anchor '+name+' '+before)
        s=s.replace(before,after);changes.append(dict(file=name,before=before,after=after,count=count))
    ast.parse(s,filename=name);(P/name).write_text(s)
write('mapper255.py',(MAP/'mapper249.py').read_text(),[('128','120')])
write('padding255.py',(MAP/'padding249.py').read_text(),[('mapper249','mapper255'),('128','120')])
s=(MAP/'projection249.py').read_text()
write('projection255.py',s,[
    ('enumerate([5]*25+[3])','enumerate([5]*24)'),
    ('list(range(5,126,5))+[128]','list(range(5,121,5))'),
    ('tuple(range(5,126,5))+(128,)','tuple(range(5,121,5))'),
    ('mask[n:25,:,:5,0]=False;mask[25,:,:3,0]=False','mask[n:24,:,:5,0]=False'),
    ('(26,32)','(24,32)'),('(26,1,mask_capacity,4096)','(24,1,mask_capacity,4096)'),
    ('(extent,)*26','(extent,)*24'),('len(groups)>25','len(groups)>24'),('128','120')])
s=(MAP/'branch_tail249.py').read_text()
write('branch_tail255.py',s,[('padding249','padding255'),
    ('!=128','!=120'),('range(128)','range(120)'),('physical_queries=128','physical_queries=120'),
    ('dict(M=128','dict(M=120')])
s=(OLD/'static_state251.py').read_text()
write('static_state255.py',s,[('branch_tail249','branch_tail255'),('projection249','projection255'),
    ('M=128','M=120'),('attention_segments=26','attention_segments=24'),('fixed_q5_groups=25','fixed_q5_groups=24')])
s=(OLD/'frames251.py').read_text()
a=s.index('    if rows==128:');z=s.index('    x,pos,meta=',a);s=s[:a]+s[z:]
s=s.replace(',tail_tokens=(0,0,0)','').replace("    if len(tail_tokens)!=3:raise ValueError('complete nonreal three-query tail tokens')\n",'')
write('frames255.py',s,[('rows not in(80,128)','rows!=120'),
    ('q5groups=16 if rows==80 else 25','q5groups=24'),
    ('tuple(range(5,q5groups*5+1,5))+((128,)if rows==128 else())','tuple(range(5,121,5))')])
s=(OLD/'metadata251.py').read_text()
a=s.index("    if frame['ids'][125:]");z=s.index('    ensure_zero(engine)',a);s=s[:a]+s[z:]
write('metadata255.py',s,[('frames251','frames255'),('static_state251','static_state255'),
    ('len(owners)!=25','len(owners)!=24'),("==128","==120"),('range(25)','range(24)'),
    ("rows=128,reference=True,tail_tokens=frame['tokens'][125:]","rows=120,reference=True"),
    ('(26,1,10,4096)','(24,1,10,4096)'),('(2560,)*26','(2560,)*24')])
s=(OLD/'reference_schedule251.py').read_text()
write('reference_schedule255.py',s,[
    ('range(0,len(active),25)','range(0,len(active),24)'),('active[begin:begin+25]','active[begin:begin+24]'),
    ('range(25-len(selected))','range(24-len(selected))'),
    ('            tokens.extend([0,future,future]);ids.extend([488]*3);positions.extend(range(3))\n',''),
    ('len(tokens)!=128','len(tokens)!=120'),('chunk=begin//25','chunk=begin//24'),
    ('[positions[5*j+4]+1 for j in range(25)]+[3]','[positions[5*j+4]+1 for j in range(24)]'),
    ('list(range(5,126,5))+[128]','list(range(5,121,5))')])
s=(OLD/'oracle251.py').read_text()
s=s.replace('mapper249','mapper255').replace('metadata251','metadata255').replace('reference_schedule251','reference_schedule255').replace('baseline251','baseline255').replace('branch_tail249','branch_tail255')
s=s.replace('if n.value==80:n.value=128','if n.value==80:n.value=120').replace('>25','>24')
a=s.index("    for label,budgets in [('rootonly'");z=s.index('    return result',a)
s=s[:a]+'''    for label,budgets in [('rootonly',[4]*23+[1]),('fork-rootonly',[5]+[4]*20+[1,3])]:
        n=len(budgets);family='full9'if 5 in budgets else'causal';m,s,padding=envelope(n,n+sum(budgets),family)
        for anchor in(range(4)if 5 in budgets else(0,)):
            result.append(dict(name=f'N{n}-{label}-anchor{anchor}',N=n,budgets=budgets,anchor=anchor,M=m,S=s,
                padding=padding,plan_family=family,query_count=n+sum(budgets),fixture_budget_reduction=None))
'''+s[z:]
write('oracle255.py',s)
write('baseline255.py',(OLD/'baseline251.py').read_text(),[('oracle251','oracle255'),('reference_schedule251','reference_schedule255'),('128','120')])
(P/'FIXTURE255.json').write_bytes((OLD/'FIXTURE251.json').read_bytes())
# Keep source changes inspectable; no claim that HOST checks grant device scope.
(P/'DECLARED_SOURCE_CHANGES255.json').write_text(json.dumps(dict(changes=changes,M=120,q5_groups=24,no_nonreal_tail3=True,real_N24_C96_unchanged=True,production_qualification=False),indent=2)+'\n')
print(P)
