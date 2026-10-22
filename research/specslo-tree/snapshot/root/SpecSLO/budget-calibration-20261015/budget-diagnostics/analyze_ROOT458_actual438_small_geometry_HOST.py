"""Count true80-capacity calls in original438, directly from its private archive."""
from pathlib import Path
from collections import Counter
import datetime,gzip,hashlib,json,tarfile
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
info=json.loads((D/'ROOT438_ACTUAL_BATCH_AND_CLASS_OWNER_ROUNDS_HOST.json').read_text())
source=Path(info['source_trace']);marker=source.parents[3]/'ARCHIVED.json';archival=json.loads(marker.read_text())
archive=Path(archival['archive']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==archival['archive_sha256']
member=str(source.relative_to(archival['restore_base']))
with tarfile.open(archive,'r:gz')as stream:raw=stream.extractfile(member).read()
assert hashlib.sha256(raw).hexdigest()==info['source_SHA256']=='42d897e4982c829b19c0ccc38ae6d4f64c24c205ab5b74ea9043af2f2de9ad07'
trace=json.loads(gzip.decompress(raw));calls=trace['normal_target_calls'];assert len(calls)==2484
eligible=[];n_hist=Counter();group_hist=Counter();classes=Counter();all_classes=Counter()
for ordinal,row in enumerate(calls):
 budgets=row['budgets'];n=row['N'];assert n==len(budgets)and len(row['sequence_ids'])>=n and all(type(v)is int and 1<=v<=5 for v in budgets)
 owners=row['sequence_ids'][:n];assert all(0<=v<463 for v in owners)and all(v>=463 for v in row['sequence_ids'][n:])
 groups=n+sum(v==5 for v in budgets);queries=n+sum(budgets)
 assert row['physical_M']==120 and queries==row['real_query_count']and groups<=24
 fit=groups<=16 and queries<=80
 for owner in owners:
  cls=trace['requests'][owner]['slo_class'];all_classes[cls]+=1
  if fit:classes[cls]+=1
 if fit:
  eligible.append(ordinal);n_hist[n]+=1;group_hist[groups]+=1
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='READONLY_ACTUAL438_SMALL_GEOMETRY_ELIGIBILITY_ONLY',
 source_trace=str(source),source_trace_sha256=info['source_SHA256'],source_archive=str(archive),source_archive_sha256=archival['archive_sha256'],
 archive_read_in_place=True,files_restored=0,actual_target_calls=len(calls),M80_capacity_eligible_calls=len(eligible),eligible_call_fraction=len(eligible)/len(calls),
 eligible_N_histogram=dict(sorted(n_hist.items())),eligible_group_histogram=dict(sorted(group_hist.items())),
 eligible_owner_verification_rounds_by_class=dict(classes),all_owner_verification_rounds_by_class=dict(all_classes),
 criterion='current positive budgets only: N + count(B==5) <=16 and N+sum(B)<=80; no outcomes or future arrivals',
 speculative_capacity_reduction_rows=40*len(eligible),original_physical_rows=120*len(calls),
 not_a_latency_or_Goodput_projection=True,NPU_used=False,NN_permission=False,Raw=None,Goodput=None)
with(D/'ROOT458_ACTUAL438_SMALL_GEOMETRY_ELIGIBILITY_HOST.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
