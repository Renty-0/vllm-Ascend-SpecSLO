"""Prepare216 plus direct packed target mapping read; no model launch."""
import ast,difflib,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
DONOR=B/'tree-cost-aware-N9to16-with-own-FIA-AR-full-v216'
P=B/'tree-packed-KV-preflight-with-own-FIA-AR-full-v217'
assert not P.exists();P.mkdir()
records=[]
for old in sorted(DONOR.iterdir()):
    if old.suffix not in('.py','.sh'):continue
    new=P/old.name.replace('216','217');new.write_text(old.read_text().replace(str(DONOR),str(P)).replace('216','217'))
    records.append(dict(source=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new=str(new)))
def change(name,old,new):
    p=P/name;s=p.read_text();assert s.count(old)==1,(name,old);p.write_text(s.replace(old,new))
change('extension217.py',"result=extension209.after_install(native,graph,live)","result=extension209.after_install(native,graph,live)\n    from slot_preflight217 import install\n    install(native)")
change('extension217.py',"new_model_or_graph=False,budget_grants_may_change=True,", "new_model_or_graph=False,budget_grants_may_change=True,packed_target_KV_preflight=True,")
change('extension217.py',"    return extension209.qualify_component(engine,native,stage,policy,output)","    value=extension209.qualify_component(engine,native,stage,policy,output)\n    from slot_preflight217 import end_shadow\n    stage('ROOT217/complete-actual-preflight-shadow',lambda:end_shadow(engine))\n    return value")
change('runtime217.py',"report[\\'ROOT217_cached_parent_dispatch\\']", "report[\\'ROOT217_cached_parent_dispatch\\']") if False else None
p=P/'runtime217.py';s=p.read_text();old="report['ROOT217_cached_parent_dispatch']=extension.cache_snapshot()";assert s.count(old)==1;s=s.replace(old,"report['ROOT217_slot_preflight']=__import__('slot_preflight217').snapshot()\\n     "+old);p.write_text(s)
change('guard217.py',"        snapshots.append(cache)","        from slot_preflight217 import audit_snapshot\n        audit_snapshot(r['ROOT217_slot_preflight'],rank,trace)\n        snapshots.append(cache)")
change('guard217.py',"cost_aware_N9to16_budget=True,postfull_independent_FIA_AR=True", "cost_aware_N9to16_budget=True,packed_target_KV_preflight=True,postfull_independent_FIA_AR=True")
for r in records:
    old=Path(r['source']).read_text().replace(str(DONOR),str(P)).replace('216','217');new=Path(r['new']).read_text()
    r['new_sha256']=hashlib.sha256(new.encode()).hexdigest();r['patch_after_namespace_projection']=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
    if Path(r['new']).suffix=='.py':ast.parse(new)
(P/'PROJECTION217.json').write_text(json.dumps(dict(schema='ROOT217.packed-preflight-source-projection.v1',donor=str(DONOR),donor_freeze_sha256=hashlib.sha256((DONOR/'FREEZE216.json').read_bytes()).hexdigest(),projection=records,NPU_started=False,source_peer_review=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='source_prepared_unfrozen',directory=str(P),NPU_started=False)))
