"""Saved protocol/structural audit; cannot recompute unretained model/KV tensors."""
import gzip,json
from pathlib import Path
from base199 import require
from scope419 import cases,FAILURES,NATIVE_COUNTERS

def base_audit(output):
 output=Path(output);reports=[json.loads((output/f'rank{i}-local-diagnostic.json').read_text())for i in range(4)];summaries=[]
 for rank,row in enumerate(reports):
  require(row['rank']==rank and row['bounded_diagnostic_completed']is True and row['full463_completed']is row['production_permission']is False and row['cache_released']is True and not row['errors'],'actual bounded worker and release')
  component=row['component'];original=component['original173'];require(original['audit']['component_numerical_qualified']is True and original['causal80']['audit']['math_independent_qualified']is True,'same-process original173 B4 and exactM80 gates')
  bound=component['bounded'];require(bound['two_complete21_zero_traces']is True and bound['production_permission']is bound['full463']is bound['whole_AR']is False and len(bound['traces'])==3*len(cases()),'exact25x3 traces without production permission')
  positive=negative=fullleaf=mixed=False
  per=[]
  for pass_index in range(3):
   for c in cases():
    path=output/'bounded199'/f'rank{rank}-pass{pass_index}-{c.name}.json.gz';v=json.loads(gzip.decompress(path.read_bytes()));require(v['name']==c.name and v['case']['child_B']==4,'actual bounded raw case')
    require(v['diagnostic_slo_inputs']==[dict(owner=i,slo_class='normal'if i<24 else'tight'if i<48 else'loose',slo_tpot_ms=50. if i<24 else 40. if i<48 else 150.,delivered_tokens=1,decode_elapsed_ms=20.,output_token_budget=64)for i in range(64)],'actual fixed diagnostic SLO inputs')
    require(all(row['policy']==dict(priority=True,target_slo_first=True,projected_wait_ms=20.)for row in v['planner']),'actual priority and SLO-first controller calls')
    diff=v['Native_delta'];require(set(diff)==set(NATIVE_COUNTERS)and all(type(x)is int and x>=0 for x in diff.values())and not any(diff[k]for k in FAILURES)and(not pass_index or not any(diff.values())),'actual full21 integer zero/failure gate')
    require(sorted(v['promoted'])==sorted(v['consumed'])and v['rollback']['after']==dict(ready=0,staged=0,payloads=0),'real promoted consume and original invalidation rollback')
    for phase in v['phases']:
     require(len(phase['actual_consensus'])==2 and all(type(x)is int and x==0 for x in phase['actual_consensus'])and len(phase['owners'])<=24,'actual retained integer original commit verdict/capacity')
     if rank:require(phase['target']['allphysical_hidden']is phase['target']['all128query_KV']is True and phase['committed_KV']['all128_committed_KV']is True,'current same-M target query/compacted KV proof')
    if rank==0:
     require(v['math']['sameM']is v['math']['all56_queryKV']is True and v['stale']['rejected']is True and v['next']['all28_written_KV']is True and v['rollback_next']['all28_written_KV']is True,'current independent draft/epoch rollback/ownnext proof')
     fullleaf|=bool(v['leaf']['fullaccept_leaf_repair_observed'])
    positive|=bool(v['promoted']);negative|=bool(v['invalid']);mixed|=bool(v['mixed24Q120_child_observed'])
    per.append(dict(pass_index=pass_index,name=c.name,promoted=v['promoted'],invalid=v['invalid'],consumed=v['consumed'],fingerprint=next(t['fingerprint']for t in bound['traces']if t['case']==c.name and t['pass_index']==pass_index)))
  summaries.append(dict(rank=rank,traces=per,natural_positive=positive,natural_reject=negative,full_leaf_repair=fullleaf if rank==0 else None,mixed24Q120_child=mixed,new_entries=bound['new_entries']))
 # Compare original actual host verdict/ticket histories only across ranks; never TP-private tensors/hashes.
 logical=lambda rows:[{k:v for k,v in r.items()if k!='fingerprint'}for r in rows]
 expected=logical(summaries[0]['traces']);require(all(logical(s['traces'])==expected for s in summaries),'all4 actual normal/eager verdict/commit histories identical')
 covered=all(s['natural_positive']and s['natural_reject']and s['mixed24Q120_child']for s in summaries)and summaries[0]['full_leaf_repair']
 return dict(status='passed_bounded_diagnostic'if covered else'passed_bounded_subset',passed=True,full_required_natural_coverage=bool(covered),ranks=summaries,source_math_reexecution=False,stored_tensor_proofs_structural_only=True,production_permission=False,full463=False,formal_Goodput=False,whole_AR=False)


def audit(output):
    value=base_audit(output);require(value['full_required_natural_coverage']is True,'ROOT419 all natural positive/reject/mixed24 and full leaf repair coverage required');output=Path(output)
    for rank in range(4):
        report=json.loads((output/f'rank{rank}-local-diagnostic.json').read_text())
        binding=report['component']['current_target_binding391']
        require(binding['actual_best376_target_backend']and not binding['best376_draft_claimed'],
            'own finite N24 uses independently bound M120 target')
        if rank:require(report['component']['actual_metadata_matches391']>0,'actual metadata witnessed')
        for path in (output/'bounded199').glob(f'rank{rank}-pass*.json.gz'):
            raw=json.loads(gzip.decompress(path.read_bytes()))
            for phase in raw['phases']:
                if rank:
                    shape=phase['target']['shape']
                    require(shape['M']==120 and shape['S']==24
                        and shape['family']=='static_FULL_M120_Q5_S24','actual current M120 finite phase')
    return dict(value,current_target_backend419='actual376 M120/Q5/S24 uniform KV',
        finite_N=24,finite_E=4,finite_R=list(range(25)),original173_fresh_baseline_retained=True,
        numerical_audit_reexecuted_by_HOST=False,production_permission=False,Raw=None,Goodput=None)
