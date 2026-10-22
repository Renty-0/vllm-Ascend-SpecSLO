from pathlib import Path
import json,hashlib,datetime
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-same-prefix-logits-KV-numeric-forensics-v220';P=B/'tree-same-prefix-logits-KV-numeric-forensics-v221';assert not P.exists();P.mkdir()
for src in[*OLD.glob('*.py'),*OLD.glob('*.sh'),OLD/'FIXTURE220.json']:
 text=src.read_text().replace(str(OLD),str(P)).replace('220','221')
 (P/src.name.replace('220','221')).write_text(text)
r=P/'runtime221.py';text=r.read_text();old="    with(Path(folder)/name).open('x')as stream:json.dump(value,stream,indent=2)";assert text.count(old)==1
text=text.replace(old,"    encoded=json.dumps(value,indent=2)\n    with(Path(folder)/name).open('x')as stream:stream.write(encoded)")
old="                    report['MC2_routes']=engine.mc2_static_route_manifest";assert text.count(old)==1
text=text.replace(old,"                    manifest=engine.mc2_static_route_manifest\n                    report['MC2_routes']=(dict(profile_sha256=manifest.profile_sha256,digest=manifest.digest,\n                        routes=[route.consensus_document()for route in manifest.routes])if manifest is not None else None)")
r.write_text(text)
p=P/'probes221.py';text=p.read_text();anchor='            pairs=[]';assert text.count(anchor)==1
text=text.replace(anchor,"            def kv_difference(a,b):\n                different=[i for i,(x,y)in enumerate(zip(a,b,strict=True))if not torch.equal(x,y)]\n                return dict(different_tensor_indices=different,first_different_layer=(different[0]//2 if different else None),\n                    first_different_kind=('key'if different[0]%2==0 else 'value')if different else None)\n            pairs=[]")
anchor="                        graph_query_KV_equal=all(torch.equal(a,b)for a,b in zip(private[i]['graph_root'],private[j]['graph_root'],strict=True))))";assert text.count(anchor)==1
text=text.replace(anchor,"                        graph_query_KV_equal=all(torch.equal(a,b)for a,b in zip(private[i]['graph_root'],private[j]['graph_root'],strict=True)),\n                        eager_query_KV_difference=kv_difference(private[i]['eager_root'],private[j]['eager_root']),\n                        graph_query_KV_difference=kv_difference(private[i]['graph_root'],private[j]['graph_root'])))")
p.write_text(text)
# Original218 SHA is a hex digest and contains no decimal220 substring; assert exact inherited binding.
from ast import parse
for p in P.glob('*.py'):parse(p.read_text())
assert "fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260"in(P/'check221.py').read_text()
report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_failed_job='same-prefix-logits-KV-root-v220-three-cases-v1',old_sources_unchanged=True,
 change='Serialize original consensus documents and JSON encode before exclusive open; add only bounded per-layer query-KV difference indices',model_or_sampler_changed=False,new_performance_run=False,NPU_used=False)
(P/'PREPARATION221.json').write_text(json.dumps(report,indent=2)+'\n');print(str(P))
