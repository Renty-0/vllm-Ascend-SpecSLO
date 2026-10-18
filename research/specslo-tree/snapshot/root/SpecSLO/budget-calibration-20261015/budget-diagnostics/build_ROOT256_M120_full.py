"""Full M120 epoch; every original startup/full/independent-AR gate retained."""
import ast,datetime,json,pathlib
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-static-q5-M128-full463-own-AR-v251'
MATH=B/'tree-static-q5-M120-node-math-v255'
P=B/'tree-static-q5-M120-full463-own-AR-v256'
P.mkdir(exist_ok=False)
shared={'static_state','metadata','frames','oracle','reference_schedule','baseline'}
changes=[]
def replace(s,before,after,name):
    count=s.count(before)
    if count<1:raise RuntimeError('missing '+name+' '+before)
    changes.append(dict(file=name,before=before,after=after,count=count))
    return s.replace(before,after)
for file in OLD.iterdir():
    if not file.is_file()or file.suffix not in('.py','.sh')or file.name.startswith('check_'):continue
    stem=file.stem.removesuffix('251');name=file.name.replace('251','256')
    if stem in shared:
        (P/name).write_text('"""Use exact immutable M120 math implementation; fresh proof required."""\nfrom '+stem+'255 import *\n')
        continue
    if stem=='bank':
        (P/name).write_text('"""Exact original72 independent neural prefixes; fresh each run."""\nfrom bank251 import allocate,build\n');continue
    s=file.read_text().replace(str(OLD),str(P)).replace('251','256')
    s=s.replace("str(B/'tree-static-q5-M128-projection-host-v249')","str(B/'tree-static-q5-M120-node-math-v255'),str(B/'tree-static-q5-M128-full463-own-AR-v251'),str(B/'tree-static-q5-M128-projection-host-v249')")
    # Both launch-time sys.path and standalone auditors need the frozen M120
    # implementation. Source validation still hashes every prior dependency.
    if stem=='startup':
        s=s.replace('export PYTHONPATH=', 'export PYTHONPATH='+str(MATH)+':'+str(OLD)+':')
    if stem in ('extension','scope','audit','guard'):
        s=s.replace('233','227')
    if stem in('extension','scope','audit','audit_static_calls'):
        for before,after in[
            ('M=128','M=120'),("['M']==128","['M']==120"),("['M'] == 128","['M'] == 120"),
            ("== call['physical_M'] == 128","== call['physical_M'] == 120"),
            ('attention_segments=26','attention_segments=24'),("['attention_segments']==26","['attention_segments']==24"),
            ('(26,10),(26,11)','(24,10),(24,11)'),('(128,)','(120,)'),
            ('[2560]*26','[2560]*24'),('list(range(5,126,5))+[128]','list(range(5,121,5))'),
            ('[26,1,capacity,4096]','[24,1,capacity,4096]'),('[26, 1, 10, 4096]','[24, 1, 10, 4096]'),
            ('<=25','<=24'),('from projection249','from projection255'),('from mapper249','from mapper255'),
            ('canonical_M=128','canonical_M=120'),('physical_primary_plus_side_groups=25','physical_primary_plus_side_groups=24'),
            ('physical_queries=128','physical_queries=120'),('physical_primary_side_groups=25','physical_primary_side_groups=24'),
            ('M128_S26','M120_S24')]:s=s.replace(before,after)
    if stem=='reference':
        for before,after in[
            ("data['ids'][:125:5]","data['ids'][:120:5]"),('(128,)','(120,)'),
            ('(2560,)*26','(2560,)*24'),('(26,32)','(24,32)'),('(26,1,10,4096)','(24,1,10,4096)'),
            ('len(values)==128','len(values)==120'),('M128_S26','M120_S24'),('M128/S26','M120/S24'),
            ('used+9','used+8'),('list(range(5,126,5))+[128]','list(range(5,121,5))'),
            ("len(data['tokens'])!=128","len(data['tokens'])!=120"),("len(data['sequence_lens'])!=26","len(data['sequence_lens'])!=24")]:
            s=replace(s,before,after,name)
        s=replace(s,"    data['tokens'].extend([0,future,future]);data['ids'].extend([488]*3)\n    data['positions'].extend(range(3));data['sequence_lens'].append(3)\n",'',name)
        s=s.replace('append nine independent','append eight independent').replace('followed by the always-nonreal three-query tail','without a short tail').replace('explicit128 physical rows','explicit120 physical rows')
    if stem=='audit_reference':
        for before,after in[
            ('len(positions)==128','len(positions)==120'),('(26*10*4096)','(24*10*4096)'),('range(26)','range(24)'),
            ('range(1,3 if group==25 else 5)','range(1,5)'),('len(heads)==128','len(heads)==120'),
            ('range(25-len(selected))','range(24-len(selected))'),
            ('                    positions.extend(range(3));ids.extend([488]*3);lengths.append(3)\n',''),
            ('list(range(5,126,5))+[128]','list(range(5,121,5))'),('[2560]*26','[2560]*24'),
            ('selected+list(range(463,463+25-len(selected)))+[488]','selected+list(range(463,463+24-len(selected)))'),
            ('M128_S26','M120_S24'),('[26,1,10,4096]','[24,1,10,4096]'),
            ('zero,3 if index==25 else 5','zero,5'),('queries in(3,5)','queries==5')]:
            s=replace(s,before,after,name)
    if stem=='guard':
        s=s.replace('M128','M120').replace('max25 primary/side groups','max24 primary/side groups')
    (P/name).write_text(s)
    if file.suffix=='.py':ast.parse(s,filename=name)
(P/'FIXTURE256.json').write_bytes((OLD/'FIXTURE251.json').read_bytes())
(P/'DECLARED_FULL256_SOURCE_CHANGES.json').write_text(json.dumps(dict(changes=changes,
    implementation=str(MATH),original_configured_target_N32_C96_Q128_unchanged=True,
    effective_N24_C96_unchanged=True,actual_M120_S24=True,active48_and_page_reservation_unchanged=True,
    all_original_startup_and_component_gates=True,new_same_instance227_and72_independent_prefixes_required=True,
    full463_ownAR_required=True,NPU_qualification_inherited=False),indent=2)+'\n')
print(P)
