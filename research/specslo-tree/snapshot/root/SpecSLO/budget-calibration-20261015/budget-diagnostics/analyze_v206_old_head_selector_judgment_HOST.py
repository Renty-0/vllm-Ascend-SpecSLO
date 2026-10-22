"""Terminal206 old-head conditional count oracle; no target-blind policy fitting."""
import collections,gzip,hashlib,itertools,json,statistics
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-b4-joint-subtree-single-writer-diagnostic-v206';TAG='truefork-single-writer-v206-bounded-gain-v1';F=P/'actual'/TAG/'joint-B4-current-diagnostic'
A=D/'V206_COMPLETED_BOUNDED_GAIN_COST_AND_OLD_HEAD_OPPORTUNITY_HOST_ANALYSIS.json';OUT=D/'V206_CONDITIONAL_OLD_HEAD_SELECTOR_LIMITS_AND_DESIGN_JUDGMENT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
assert sha(A)=='4ab768297ecc5d18bce630d3bd93e2d7e36953ea8e22dd02af143bb7ee253d5a'
assert sha(P/('receipt-'+TAG+'.json'))=='e387a2286f18bba1c4139e32b8845418a2aee7f576500c74d4b2be39a3064e31'
a=json.loads(A.read_text());pins=json.loads(Path(a['raw_pins_file']['path']).read_text());parents=(-1,0,1,2,-1,0,1,2)
layouts=[s for s in itertools.combinations(range(8),4)if all(parents[i]<0 or parents[i]in s for i in s)]
def fork(s):return any([parents[i]for i in s].count(p)>1 for p in {parents[i]for i in s})
def count(chosen,raw,heads):
 accepted=0;parent=-1
 for depth in range(4):
  x=next((i for i in chosen if parents[i]==parent and raw[i]==heads[parent+1]),None)
  if x is None:return accepted+1
  accepted+=1
  if x>=4:return accepted+1
  parent=x
 return accepted+1
rows=[];perclass=[];source={}
for ci,category in((0,'coding'),(2,'chat'),(4,'summarization')):
 dp=F/f'rank0-gain-case{ci}.json.gz';tp=F/f'rank1-gain-case{ci}.json.gz'
 assert sha(dp)==pins[str(dp)]and sha(tp)==pins[str(tp)];source.update({str(dp):sha(dp),str(tp):sha(tp)})
 draft=load(dp);target=load(tp);totals=collections.Counter();oracle=[];forkonly=[];fallback=[];b5=[]
 for i,summary in enumerate(target['rows']):
  if not summary['measured']:continue
  raw=draft['work']['draft_proof']['raw_pool'][i];heads=target['heads'][5*i:5*i+5]
  counts={','.join(map(str,s)):count(s,raw,heads)for s in layouts};totals.update(counts);best=max(counts.values());fo=max(count(s,raw,heads)for s in layouts if fork(s));fb=max(summary['g'],fo)
  scores=draft['work']['draft_proof']['node_confidences'][i];joint=[]
  for parent,p in zip(parents,scores):joint.append(p*(joint[parent]if parent>=0 else 1))
  bestside=max(range(4,8),key=lambda n:(joint[n],-n));b5nodes=(0,1,2,3,bestside);b5g=count(b5nodes,raw,heads)
  old=count((0,1,2,3),raw,heads);assert old==summary['g']
  owner=draft['case']['owners'][i];rows.append(dict(category=category,request_id=owner['request_id'],prefix_offset=owner['output_offset'],old_g=old,conditional_counts_by_layout=counts,
   oracle_all8=best,oracle_sixfork_only=fo,oracle_sixfork_plus_old_spine=fb,conditional_existing_B5_backbone_bestside=bestside,conditional_existing_B5_count=b5g))
  oracle.append(best);forkonly.append(fo);fallback.append(fb);b5.append(b5g)
 perclass.append(dict(category=category,conditional_fixed_layout_means={k:v/12 for k,v in sorted(totals.items())},conditional_all8_oracle_mean=statistics.mean(oracle),conditional_sixfork_only_mean=statistics.mean(forkonly),
  conditional_sixfork_plus_old_spine_oracle_mean=statistics.mean(fallback),conditional_existing_B5_bestside_mean=statistics.mean(b5)))
assert len(rows)==36
v=dict(schema='V206.conditional-old-head-selector-design-limits.v1',status='readonly_conditional_analysis',base_analysis=dict(path=str(A),sha256=sha(A)),authorized_receipt_sha256='e387a2286f18bba1c4139e32b8845418a2aee7f576500c74d4b2be39a3064e31',source_raw_pins=source,
 per_class=perclass,per_prefix_conditional_rows=rows,
 totals=dict(old_delivered=sum(x['old_g']for x in rows),joint_actual_delivered=sum(x['joint_g']for x in a['measured_pairs']),
  all8_oldhead_oracle_delivered=sum(x['oracle_all8']for x in rows),sixfork_plus_old_spine_oldhead_oracle_delivered=sum(x['oracle_sixfork_plus_old_spine']for x in rows),
  forced_sixfork_only_oldhead_oracle_delivered=sum(x['oracle_sixfork_only']for x in rows),existing_B5_bestside_oldhead_conditional_delivered=sum(x['conditional_existing_B5_count']for x in rows),
  actual_selectedfork9_side0=True,conditional_opportunities10=True,k3_alternate_linear0127=3,sixfork_domain_opportunities7=True),
 interpretation=[
  'All8 conditionaloracle =old125+10=135; restricting to6forks+oldspine fallback excludes3 k3/0127 opportunities andgives132. Thisis target-awareold-head arithmetic,notactualnewg.',
  'Forcedfork-only losesprimarydepth andhasno spinesuccessfallback. The oldspine is bestfixed4layout onthese retainedcounts; aalwaysbranch policy isnotjustified.',
  'Naturalgreedyjoint cannotchoose0127: primary3and side7 have sameparent, raw top1p3>=top2p7 andties favor3. 0127mustbe a separatelynamed alternate-linear policy ifever studied,withbranch/leaf KVownnext qualified.',
  'There were 10availabletop2sidematches atfirstoldprimary mismatch butnonechosen byactualjoint. Thisindicates draftjointprobability order missedall10conditionalopportunities; itdoesnot establish aheldouttargetmatchscore.',
  'The 171retainedsharedroot/primary positions happenedtoagree acrossactualoldcausal andjointmask shapes. Thisnever licensesunqueriedsideheads orarbitraryFULL9 numericalequivalence.',
  'ConditionalB5bestside preservesallprimary4 andusesexistingbackbone sidecriterion, butrequires6targetqueries/newphysical geometry/currentmath. ItsoldheadcountisnotactualB5gain andnotapredictedGP.',
 ],
 minimum_selector_design_judgment=dict(
  production_change_now='Not justified by206 alone. Retainexistingbackbone fallback; no210 creation, no newg installation.',
  ex_ante_interface='Anyfuture selector mustchooseonlyfromcurrentdraft roundedprobabilities/rawtokens,class/B andpastcalibration, beforetargetheads; preserveactualB/ancestorclosure/unitcost/normalmin1/eager0 andalloriginalpriority/grants.',
  smallest_hypothesis='A bounded calibratedside-swap rule maycompareexpected extra first-mismatch-side coverage againstlostdeep-primary continuation, usingclass/depth/confidence features. It mustdefault tospine whenevidence isweak. Currentdraftjoint-producttop4 alreadyfailed tofindthese10; relabelingit cannotfixranking.',
  heldout='Do notfit thresholdsorlookup the36current targetaware outcomes thenreport same-source gain. Keep all3prefixes ofarequest inonefold; fouridentityclusters/class areexploratory, and independentidentity validation isrequired.',
  shape_and_state='If future rule choosesB4fork, preserveactualFULL9 currentG/H/ancestry mask/sameM ownKV/realcompaction/currentownnext gates. Ifit chooses0127, separatelyverifylinear-node provenance/side-leaf repair, notassumeallspinesharesdraftledger.',
  next_action='Source-onlytarget-aligned calibration/holdout specification byindependentpeer, then ROOT decides whethera bounded newsource experiment isworthwhile. Nofull-run gainpromise.'),
 measured_g_added=False,new_model_or_NPU=False,Torch_imported=False,candidate_modified=False,old_g_or_frozen_sources_modified=False,formal_Goodput=False,whole_AR=False,adopted=False)
assert not OUT.exists();OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),totals=v['totals'],per_class=perclass)))
