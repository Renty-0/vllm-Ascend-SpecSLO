"""Conservative exact160 rows from all six actual454 accepted measurements."""
from pathlib import Path
import copy
import hashlib
import json
import math

P=Path(__file__).resolve().parent
B=P.parent
D=B/'budget-diagnostics'
BASE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def p95(values):
    values=sorted(values);position=(len(values)-1)*.95
    lo=math.floor(position);hi=math.ceil(position)
    if lo==hi:return values[lo]
    f=position-lo
    return values[lo]*(1.-f)+values[hi]*f


def build():
    evidence=D/'ROOT454_TERMINAL_INDEPENDENT_REAL32_MC2_AUDIT_HOST.json'
    audited=json.loads(evidence.read_text())
    assert audited['original_supervisor_status']=='complete'and audited['all6_original_operator_gates_accepted']is True
    assert audited['all4idle']is True and audited['owned_remaining']==[]and audited['new_profile_installed']is False
    receipt=Path(audited['receipt']);actual=json.loads(receipt.read_text())
    assert sha(receipt)==audited['receipt_sha256']and actual['status']=='complete'and actual['sources_unchanged']is True
    base=json.loads(BASE.read_text());result=copy.deepcopy(base)
    assert base['minimum_samples']==31 and base['minimum_speedup']==1.02 and base['latency_percentile']==95.
    assert base['metadata']['standalone_chained_flush']is True
    assert not any(row['m']==160 for row in base['measurements'])
    additions=[];sources=[]
    for kind,k in(('attention',3072),('down',8576)):
        records=sorted((row for row in audited['summary']if row['projection']==kind),key=lambda v:v['layer'])
        assert [v['layer']for v in records]==[0,31,63]
        raw=[];profiles=[]
        for record in records:
            path=Path(record['raw']);assert sha(path)==record['raw_sha256']and record['original_gate_accepted']is True
            q=json.loads(path.read_text());profile_path=Path(record['profile']);profile=json.loads(profile_path.read_text())
            assert q['shape']==dict(m=160,k=k,n=5120)
            c=q['correctness'];assert c['exact_all_ranks']is True and c['changed_input_exact_all_ranks']is True
            assert c['changed_input_replays']==1024 and all(c[key]==0. for key in('max_abs_norm','max_abs_added','changed_input_max_abs_norm','changed_input_max_abs_added'))
            assert profile['metadata']['runtime_binding']==base['metadata']['runtime_binding']
            assert profile['metadata']['source_sha256']==base['metadata']['source_sha256']
            assert profile['metadata']['operator']==base['metadata']['operator']and profile['metadata']['standalone_chained_flush']is True
            assert len(profile['measurements'])==1 and profile['minimum_speedup']==1.02
            raw.append(q);profiles.append(profile)
            sources.append(dict(layer=record['layer'],projection=kind,raw=str(path),raw_sha256=sha(path),
                qualified_profile=str(profile_path),qualified_profile_sha256=sha(profile_path)))
        # Each ordinal is bounded by the best observed baseline and worst
        # observed candidate across all three layers; no best layer selected.
        baseline=[min(v['samples']['baseline_latency_ms'][i]for v in raw)for i in range(31)]
        fused=[max(v['samples']['fused_latency_ms'][i]for v in raw)for i in range(31)]
        assert p95(fused)*1.02<=p95(baseline)
        row=copy.deepcopy(profiles[0]['measurements'][0])
        assert(row['m'],row['k'],row['n'])==(160,k,5120)
        row['baseline_latency_ms']=baseline;row['fused_latency_ms']=fused
        row['aggregation_provenance']=dict(method='conservative_cross_layer_ordinal_envelope_v1',layers=[0,31,63],
            exact_shape_only=True,no_M_interpolation=True,source_records=[v for v in sources if v['projection']==kind],
            baseline_p95_ms=p95(baseline),fused_p95_ms=p95(fused),p95_speedup=p95(baseline)/p95(fused),
            real_input_scope='32 separately prefilled owners, all160 B4 query rows real; no payload or weights serialized')
        additions.append(row)
    result['measurements']=base['measurements']+additions
    result['metadata']['ROOT455_exact160_addition']=dict(parent_profile=str(BASE),parent_profile_sha256=sha(BASE),
        actual454_audit=str(evidence),actual454_audit_sha256=sha(evidence),actual454_receipt_sha256=sha(receipt),
        source_records=sources,ordinary_prompt_prefill_unchanged=True,
        original_profile_rows_preserved_byte_values=True,deferred160_enabled=False,whole_model_NN_pending=True)
    path=P/'mc2-profile-with-qualified160.json'
    with path.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    out=dict(status='BUILT_ACTUAL454_CONSERVATIVE_EXACT160_PROFILE_HOST_ONLY',profile=str(path),profile_sha256=sha(path),
        original_profile=str(BASE),original_profile_sha256=sha(BASE),original_rows=len(base['measurements']),added_rows=2,
        additions=[dict(M=v['m'],K=v['k'],N=v['n'],**{k:v['aggregation_provenance'][k]for k in('baseline_p95_ms','fused_p95_ms','p95_speedup')})for v in additions],
        parent_audit=str(evidence),parent_audit_sha256=sha(evidence),source_records=sources,
        NPU_used=False,Native_profile_normalization_pending=True,new_profile_installed=False,
        whole_model_NN_pending=True,production_permission=False,Raw=None,Goodput=None,Git_pushes=0)
    with(P/'HOST455_DERIVED_PROFILE_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':build()
