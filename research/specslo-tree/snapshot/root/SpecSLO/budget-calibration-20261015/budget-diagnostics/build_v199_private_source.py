from pathlib import Path
import json,hashlib,difflib
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199';G=D/'v199-root-execution'
A=B/'tree-empty-active-cohort-causal80-active32-full-v197';OLD=B/'tree-linear-parent4-child3-bank-binding-full-v179'
assert P.is_dir() and not list(P.iterdir())
projection=[]
def put(name,text):
 (P/name).write_text(text.rstrip()+'\n')
def projected(name,source,old,new,transform=lambda x:x):
 raw=source.read_text();base=raw.replace(old,new);out=transform(base);put(name,out)
 ops=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(a=base,b=out,autojunk=False).get_opcodes():
  if tag!='equal':ops.append(dict(start=a,end=b,old=base[a:b],new=out[c:d]))
 projection.append(dict(file=name,parent=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),namespace=[old,new],deltas=ops))
for stem in ['base','source','progress','config','check','host_full']:
 projected(stem+'199.py',A/(stem+'197.py'),'197','199')
p=P/'base199.py';text=p.read_text().replace("ACTION='empty-active-cohort-causal80-active32-full463'","ACTION='bounded-parent4-child4-shared4-diagnostic'")
text+='\nV151=B/\'tree-normal4-parent3-shared4-eager-v151\'\nV152=B/\'tree-union-shared4-protocol-local-v152\'\nV159=B/\'tree-live-shared4-union-v159\'\nV161=B/\'tree-union-own-epoch-metadata-fix-local-v161\'\nV179=B/\'tree-linear-parent4-child3-bank-binding-full-v179\'\nfor root in (V151,V152,V159,V161,B/\'tree-natural-positive-union-discovery-local-v154\',B/\'tree-repair-frontier-fusion-local-v149\'):\n if str(root) not in sys.path:sys.path.append(str(root))\n'
# Re-record the complete sole parent projection after added constants.
projection=[r for r in projection if r['file']!='base199.py'];(P/'base199.py').unlink();projected('base199.py',A/'base197.py','197','199',lambda x:text)
# Selection excludes old receipts as qualification, includes exact source/input bytes.
inputs=json.loads((A/'INPUTS197.json').read_text());oldinputs=json.loads((OLD/'INPUTS179.json').read_text());inputs['files'].update(oldinputs['files'])
for directory in [OLD,B/'tree-normal4-parent3-shared4-eager-v151',B/'tree-live-shared4-union-v159',B/'tree-union-own-epoch-metadata-fix-local-v161']:
 for q in directory.glob('*.py'):inputs['files'][str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
for q in [OLD/'FIXTURE_INPUT179.json',A/'next4_197.py']:
 inputs['files'][str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
put('INPUTS199.json',json.dumps(inputs,indent=2,sort_keys=True));put('FIXTURE_INPUT199.json',(OLD/'FIXTURE_INPUT179.json').read_text())
# Original provider uses only prefixes through current root, never old target future.
projected('provider199.py',OLD/'provider179.py','179','199')
projected('observe199.py',OLD/'observer179.py','179','199',lambda x:x.replace('context_lens=metadata.context_lens.index_select(0,index)','context_lens=metadata.context_lens.index_select(0,index.to(metadata.context_lens.device))'))
# Original generic observer defs are imported into own namespace; no install on import.
put('SCOPE199.txt','SOURCE ONLY. Finite P4->child4 shared4 diagnostic. N16 parent/normal; E1R0,E1R1,E2R1,E3R1,E4R1. Original173 current baseline first. No production issuer, full463, GP, AR or inherited numerical qualification. Child3 is source contrast only. No shifted ABI/no no-BONUS.')
put('PROJECTIONS199.json',json.dumps(dict(projection=projection),indent=2,sort_keys=True))
# Template manifest identifies the new guard itself; no permission/freeze created.
for name in ['guard197.py','startup197.sh','launch197.sh','GUARD_INPUTS197.json']:
 q=G/name.replace('197','199');q.write_text((D/'v197-root-execution'/name).read_text().replace('197','199').replace('tree-empty-active-cohort-causal80-active32-full-v199','tree-bounded-parent4-shared4-eager-diagnostic-v199').replace('empty-active-cohort-causal80-active32-full463','bounded-parent4-child4-shared4-diagnostic'))
print(P)
