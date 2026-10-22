from pathlib import Path
import ast,difflib,hashlib,json
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-normal-cap3-original-geometries-full-v207';G=D/'v207-root-execution'
OLD=B/'tree-empty-active-cohort-causal80-active32-full-v197';P180=B/'tree-uniform-b3-three-bucket-full-v180';P176=B/'tree-high-normal-cap3-active32-full-v176'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
f=read(OLD/'FROZEN_SOURCE197.json');pins={}
for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):pins.update(f[k])
pins.update({str(OLD/'CPU_RESULT197.json'):f['CPU_sha256'],str(OLD/'FROZEN_SOURCE197.json'):sha(OLD/'FROZEN_SOURCE197.json'),f['root_review']['path']:f['root_review']['sha256']})
assert len(pins)==294
for p in [P180/'bank180.py',P180/'draft180.py',P180/'target180.py',P180/'test_host180.py',P176/'shape176.py',P176/'observe176.py',P176/'host176.py']:
 assert str(p)not in pins or pins[str(p)]==sha(p);pins[str(p)]=sha(p)
for p in [B/'tree-high-normal-cap4-native-families-v148/fast_tree_draft_v5.py',B/'tree-high-normal-cap4-native-families-v148/parallel_steps.py',B/'tree-high-normal-cap4-native-families-v148/original_native_protocol_v75.py',*[Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')/n for n in ('spec_rhythm.py','native_model.py','native_graph.py','native_cache.py')]]:pins[str(p)]=sha(p)
for path in read(P/'INPUTS207.json')['files']:
 if '/tree-normal4-parent3-shared4-eager-v151/'in path or '/tree-parent3-union3-padding-observer-devicefix-full-v188/'in path or '/tree-bounded-parent4-shared4-eager-diagnostic-v199/'in path or '/tree-short-parent-eager-v134/'in path or '/tree-packed-eager01-cost-v102/'in path or '/tree-natural-positive-union-discovery-local-v154/'in path or '/tree-packed-eager01-cost-v107/'in path:pins[path]=sha(path)
for p,h in pins.items():assert sha(p)==h,p
oldinput=read(OLD/'INPUTS197.json')
value=dict(schema='V207.readonly-sole197-source-inputs.v1',files=pins,parent_archive=oldinput['parent_archive'],P160_archive=oldinput['P160_archive'],P173=oldinput['P173'],
 parent197=dict(freeze_sha256=sha(OLD/'FROZEN_SOURCE197.json'),CPU_sha256=sha(OLD/'CPU_RESULT197.json'),archive=dict(path=str(OLD/'PREPARED_SOURCE_ARCHIVE197.tar.gz'),sha256='a5e7c4811ef714176cd988639b50ca4ba5f7d32a834fc812cb08bc6109a0c35b0')),
 old_device_qualification_inherited=False,production_delta='normalN>8 cap4->3; lowerNtrueB5 and allg/BONUS/original mapper remain197',
 ROOT_diagnostic_key_authorization='H96/S32 and H128/S38 diagnostic-only counterparts; current two-layer P3E3 strict domain')
# Parent archive pin is read from immutable ROOT preparation, not guessed.
rootreview=read(f['root_review']['path'])
# Root prepared descriptor can be located in immutable INPUTS or own archive SHA:
# SOURCE archive reading by this preparation helper computes bytes only.
value['parent197']['archive']['sha256']=sha(OLD/'PREPARED_SOURCE_ARCHIVE197.tar.gz')
(P/'INPUTS207.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
record=read(P/'_BUILD_DONORS.json');rows=[]
for dest,(source,epoch)in sorted(record.items()):
 p=Path(dest)
 if not p.is_file()or p.parent!=P:continue
 a=Path(source).read_text();b=p.read_text();ops=[]
 for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
  if tag!='equal':ops.append(dict(start=i,end=j,old=a[i:j],new=b[k:l]))
 rows.append(dict(file=p.name,parent=source,parent_sha256=sha(source),namespace_epoch=epoch,namespace_Sha_literals_preserved=True,deltas=ops))
(P/'PROJECTIONS207.json').write_text(json.dumps(dict(schema='V207.exact-whole-projections.v1',projection=rows,
  production_single_delta='highN>8 individualcap4->3; mapper197 unchanged',diagnostic_helper_donors='180B3 originalengine pad;176 originalcap3 shape/snapshot only; current device evidence never inherited'),indent=2,sort_keys=True)+'\n')
for p in P.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
print(json.dumps(dict(local_py=len(list(P.glob('*.py'))),readonly=len(pins),whole_projections=len(rows),P=str(P))))
