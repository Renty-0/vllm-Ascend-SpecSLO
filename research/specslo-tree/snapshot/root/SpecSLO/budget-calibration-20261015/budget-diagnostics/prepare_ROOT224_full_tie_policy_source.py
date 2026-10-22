"""Explicit source-only new sampling-tie policy arm; no model execution."""
from pathlib import Path
import tokenize,io,ast,json,hashlib,datetime,re
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';old=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218';P=B/'tree-FP32-tie-policy-full463-own-AR-v224';assert not P.exists();P.mkdir()
records=[]
def project(text):
 out=[]
 for t in tokenize.generate_tokens(io.StringIO(text).readline):
  value=t.string
  if t.type==tokenize.NAME:value=value.replace('218','224')
  elif t.type==tokenize.STRING:
   try:literal=ast.literal_eval(value)
   except Exception:literal=None
   if not(isinstance(literal,str)and re.fullmatch('[a-f0-9]{64}',literal)):value=value.replace('218','224')
  out.append(tokenize.TokenInfo(t.type,value,t.start,t.end,t.line))
 return tokenize.untokenize(out)
for path in sorted([*old.glob('*.py'),*old.glob('*.sh')]):
 text=project(path.read_text())if path.suffix=='.py'else path.read_text().replace('218','224');new=P/path.name.replace('218','224');new.write_text(text)
 records.append(dict(original=str(path),original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),new=str(new),namespace_only=True))
# Exactly the tested helper; no eager/install behavior hidden in the copy.
helper=B/'tree-FP32-BF16-tie-refinement-diagnostic-v223/precision_ties223.py';(P/'precision_ties224.py').write_bytes(helper.read_bytes())
# Explicit pre-constructor install and post-run report; all original startup/zero/full/ref gates stay.
p=P/'runtime224.py';s=p.read_text();anchor="    for before,after,count in changes:";assert s.count(anchor)==1
extra='''    changes=changes+(
        ("   state=capture_before_native_constructor(torch)",
         "   from vllm_ascend.spec_decode.pearl import native_model\\n   from tie224 import install as install_tie_policy\\n   overlay['ROOT224_target_tie_policy']=install_tie_policy(native_model,rank)\\n   state=capture_before_native_constructor(torch)",1),
        ("     report['Native_final']=", "     from tie224 import snapshot as tie_snapshot\\n     report['ROOT224_tie_policy']=stage('ROOT224/tie-policy-final',lambda:tie_snapshot(engine))\\n     report['Native_final']=",1))
'''
s=s.replace(anchor,extra+anchor);p.write_text(s)
# Provenance of the actual new private head policy before/after constructor.
p=P/'extension224.py';s=p.read_text();anchor="    return dict(extension209.loaded_origins(),ROOT224_explicit_overlay_sources={";assert s.count(anchor)==1
s=s.replace(anchor,"    from tie224 import source_report\n"+anchor.replace('ROOT224_explicit_overlay_sources={','ROOT224_tie_policy_sources=source_report(),ROOT224_explicit_overlay_sources={'));p.write_text(s)
# Reference explicitly declares that both paths now use the refined maximum-tie rule.
p=P/'reference224.py';s=p.read_text();anchor="        completed=True,backend='original target TP3 BF16 causal FIA Q80/S16; only first query per owner consumed',";assert s.count(anchor)==1
s=s.replace(anchor,"        completed=True,backend='target TP3 BF16 body causal FIA Q80/S16 with explicitly refined FP32 maximum-tie rule; only first query per owner consumed',\n        target_greedy_tie_policy='ROOT224 complete BF16 top ties up to4 per shard refined in FP32; original choice on overflow',")
p.write_text(s)
for path in P.glob('*.py'):ast.parse(path.read_text())
(P/'PROJECTION224.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace_projection=records,original223_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),production_delta='target greedy tie policy explicitly changed before fresh original qualification; weights/model body/clients/arrivals/budgets/clock unchanged',new_NPU_qualification_required=True),indent=2)+'\n')
print(str(P))
