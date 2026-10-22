"""Authorized completed206 bounded gain/cost and conditional-old-head arithmetic."""
import ast,collections,copy,gzip,hashlib,itertools,json,math,statistics,sys,traceback,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-b4-joint-subtree-single-writer-diagnostic-v206'
TAG='truefork-single-writer-v206-bounded-gain-v1';RECEIPT_SHA='e387a2286f18bba1c4139e32b8845418a2aee7f576500c74d4b2be39a3064e31'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
EVIDENCE=D/'v206-completed-gain-cost-analysis-HOST';OUT=D/'V206_COMPLETED_BOUNDED_GAIN_COST_AND_OLD_HEAD_OPPORTUNITY_HOST_ANALYSIS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if ok is not True:raise ValueError(msg)
def raw(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def stats(v):return dict(samples=v,n=len(v),median=statistics.median(v),mean=statistics.mean(v),min=min(v),max=max(v),range=max(v)-min(v),population_stddev=statistics.pstdev(v))
def main():
 need(not EVIDENCE.exists()and not OUT.exists(),'new independent HOST evidence paths');EVIDENCE.mkdir()
 (EVIDENCE/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 result=dict(schema='V206.completed-bounded-gain-cost-readonly-HOST-analysis.v1',status='failed_HOST_analysis',errors=[],tag=TAG,
  Torch_imported=False,NPU_used=False,model_constructed=False,candidate_modified=False,new_gain_table_installed=False,production_qualified=False,whole463_run=False,whole_AR=False,formal_Goodput=False,adopted=False)
 try:
  receipt=P/('receipt-'+TAG+'.json');need(sha(receipt)==RECEIPT_SHA,'exact ROOT-authorized terminal206 receipt');r=json.loads(receipt.read_text());(EVIDENCE/receipt.name).write_bytes(receipt.read_bytes())
  need(r['status']=='complete'and r['actual_bounded_status']=='passed_subset'and r['child']['exit_code']==0 and r['sources_unchanged']is True and r['owned_remaining']==[]and r['devices_after']['all_idle']is True,'complete bounded subset source/cleanup saved receipt')
  result['receipt']=dict(path=str(receipt),sha256=sha(receipt));result['saved_cleanup']=dict(owned_remaining=r['owned_remaining'],all4_idle=r['devices_after']['all_idle'],pre=r['pre_verifier']['exit_code'],post=r['post_verifier']['exit_code'])
  f=json.loads((P/'FROZEN_SOURCE206.json').read_text());source={str(P/name):sha(P/name)for name in('selector206.py','actor206.py','draft206.py','target206.py','bank206.py','audit206.py')};source[str(NATIVE)]=sha(NATIVE)
  for q,h in source.items():need(({**f['files'],**f['external_files']}).get(q)==h,'exact frozen producer or verifier source '+q)
  for name in('selector206.py','actor206.py','audit206.py'):(EVIDENCE/name).write_bytes((P/name).read_bytes())
  result['source_pins']=dict(source,**{str(P/'FROZEN_SOURCE206.json'):sha(P/'FROZEN_SOURCE206.json')})
  folder=P/'actual'/TAG/'joint-B4-current-diagnostic';paths=sorted(folder.glob('*.json.gz'));need(len(paths)==196 and not any(p.is_symlink()for p in paths),'all196 bounded raw files, no full/model access')
  pins={str(p):sha(p)for p in paths};(EVIDENCE/'RAW_PINS.json').write_text(json.dumps(pins,indent=2,sort_keys=True)+'\n')
  data={p.name:raw(p)for p in paths};reports=[data[f'rank{i}-diagnostic.json.gz']for i in range(4)]
  for rank,report in enumerate(reports):need(report['rank']==rank and report['completed']is report['math_independent_qualified']is report['cache_released']is True and report['errors']==[],'all completed current role reports')
  ns=dict(math=math,require=need);tree=ast.parse((P/'selector206.py').read_text())
  for n in tree.body:
   if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in('PARENTS','POLICIES')for t in n.targets):exec(compile(ast.Module(body=[copy.deepcopy(n)],type_ignores=[]),'actual-selector-constants','exec'),ns)
  names={'validate','joint_scores','choose','is_fork'};nodes=[copy.deepcopy(n)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-selector-definitions','exec'),ns)
  parents=ns['PARENTS'];policies=ns['POLICIES'];oldpolicy,jointpolicy=policies[:2]
  at=ast.parse((P/'audit206.py').read_text());nodes=[copy.deepcopy(n)for n in at.body if isinstance(n,ast.ClassDef)and n.name in('Vector','Matrix')];car=dict(copy=copy)
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'original-HOST-integer-carriers','exec'),car)
  nt=ast.parse(NATIVE.read_text());cls=next(n for n in nt.body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine');node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_verify_tree_outputs_host'));node.decorator_list=[];node.returns=None
  for a in(*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs):a.annotation=None
  car.update(torch=types.SimpleNamespace(long='explicit-HOST-integer',tensor=lambda v,**kw:car['Matrix'](v)),TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual-Native-integer-only','exec'),car)
  layouts=[]
  for subset in itertools.combinations(range(8),4):
   if all(parents[i]<0 or parents[i]in subset for i in subset):layouts.append(subset)
  need(len(layouts)==8,'all eight exact4 ancestorclosed layouts')
  def check_case(v,rank):
   w=v['work'];need(len(w['rows'])==16,'actual16owners')
   for i,(chosen,row,summary)in enumerate(zip(w['selected'],w['rows'],v['rows'],strict=True)):
    need(len(chosen)==len(row)==4 and tuple(chosen)in layouts,'actual selected quota/closure')
    accepted=[x for x in v['commit']['accepted_paths'][i]if x>=0];out=v['commit']['committed_tokens'][i]
    need(summary['g']==len(out)==len(accepted)+1 and summary['accepted_count']==len(accepted)and summary['delivered_tokens']==out,'unclipped true original g/path/correction')
    need(summary['accepted_original_nodes']==[chosen[j]for j in accepted]and summary['accepted_side_original_nodes']==[chosen[j]for j in accepted if chosen[j]>=4]and summary['true_fork']==ns['is_fork'](chosen),'actual side/fork bookkeeping')
   if rank==0:
    proof=w['draft_proof'];need(v['heads']is v['target']is v['reference']is None,'draft no invented target evidence')
    for i,(tokens,conf,chosen,row)in enumerate(zip(proof['raw_pool'],proof['node_confidences'],w['selected'],w['rows'],strict=True)):
     need(len(tokens)==len(conf)==8 and row==[tokens[j]for j in chosen],'real raw8 selected input')
     if v['case']['policy']==jointpolicy:need(tuple(chosen)==ns['choose'](conf),'independent actual jointscore selection')
   else:
    need(w['draft_proof']is None and len(v['heads'])==80,'target owns current complete80 head')
    remapped=[]
    for chosen in w['selected']:
     rm={j:i for i,j in enumerate(chosen)};remapped.append([-1 if parents[j]<0 else rm[parents[j]]for j in chosen])
    plans=[types.SimpleNamespace(candidate_budget=4,parent_indices=car['Vector'](p))for p in remapped]
    got=car['_verify_tree_outputs_host'](w['rows'],plans,car['Vector'](v['heads']),4,'HOSTinteger')
    need([[x for x in row if x>=0]for row in got.token_ids.values]==v['commit']['committed_tokens']and got.accepted_node_indices.values==v['commit']['accepted_paths'],'actual original Native integer verdict from retained all80 heads')
   return (v['commit']['committed_tokens'],v['commit']['accepted_paths'])
  integer_count=0
  for ci in range(6):
   same=[]
   for rank in range(4):
    v=data[f'rank{rank}-gain-case{ci}.json.gz'];need(v['rows']==reports[rank]['gain'][ci]['rows'],'raw/role report exactgain')
    same.append(check_case(v,rank));integer_count+=int(rank>0)
   need(all(a==same[0]for a in same),'all4 actualgain verdict agreement')
  geometry_sides=collections.Counter()
  for phase in range(3):
   for ci in range(8):
    same=[]
    for rank in range(4):
     q=data[f'rank{rank}-geometry-p{phase}-case{ci}.json.gz'];same.append(check_case(q,rank));integer_count+=int(rank>0)
    need(all(a==same[0]for a in same),'all4 controlledgeometry verdict agreement')
    if phase==0:
     rows=data[f'rank1-geometry-p0-case{ci}.json.gz']['rows'];geometry_sides[str(ci)]=sum(bool(x['accepted_side_original_nodes'])for x in rows)
  # Old primary-head oracle only counts outputs, never invents a side-node head/token.
  def oldhead_count(chosen,raw8,head):
   accepted=[];parent=-1
   for depth in range(4):
    need(parent<4,'side nodes terminal byraw8 topology')
    pred=head[parent+1];nxt=next((i for i in chosen if parents[i]==parent and raw8[i]==pred),None)
    if nxt is None:return len(accepted)+1,accepted
    accepted.append(nxt)
    if nxt>=4:return len(accepted)+1,accepted
    parent=nxt
   return len(accepted)+1,accepted
  summaries=[];paired=[];score_rows=[];opportunities=[];side_reasons=collections.Counter();shared=collections.Counter()
  for ci,category in((0,'coding'),(2,'chat'),(4,'summarization')):
   od=data[f'rank0-gain-case{ci}.json.gz'];jd=data[f'rank0-gain-case{ci+1}.json.gz'];ot=data[f'rank1-gain-case{ci}.json.gz'];jt=data[f'rank1-gain-case{ci+1}.json.gz']
   rows=[];id_diffs=collections.defaultdict(list);mismatch=collections.Counter();cohortshared=collections.Counter();raw_equal=0
   for i,(a,b)in enumerate(zip(ot['rows'],jt['rows'],strict=True)):
    need(a['measured']==b['measured']and a['source_context_sha256']==b['source_context_sha256'],'actual paired sourceprefix')
    if not a['measured']:continue
    owner=od['case']['owners'][i];need(a['original_request_id']==owner['request_id']and owner['output_offset']in(32,96,192),'retained original sourceidentity/correlatedprefix')
    raw8=od['work']['draft_proof']['raw_pool'][i];scores=od['work']['draft_proof']['node_confidences'][i];jraw=jd['work']['draft_proof']['raw_pool'][i];jscores=jd['work']['draft_proof']['node_confidences'][i]
    raw_equal+=int(raw8==jraw and scores==jscores);joint=ns['joint_scores'](jscores);ranks=sorted(range(8),key=lambda x:(-joint[x],x));bestside=max(range(4,8),key=lambda n:(joint[n],-n))
    oldheads=ot['heads'][5*i:5*i+5];newheads=jt['heads'][5*i:5*i+5];chosen=jt['work']['selected'][i];rm={x:j for j,x in enumerate(chosen)}
    sharedqueries=[('root',oldheads[0],newheads[0])]+[(str(node),oldheads[node+1],newheads[rm[node]+1])for node in range(4)if node in rm]
    for name,x,y in sharedqueries:shared['queries']+=1;cohortshared['queries']+=1;shared['equal']+=int(x==y);cohortshared['equal']+=int(x==y)
    k=next((n for n in range(4)if raw8[n]!=oldheads[n]),4);mismatch['full_primary4'if k==4 else str(k)]+=1
    available=k<4 and raw8[k+4]==oldheads[k]
    variants=[dict(nodes=list(s),g=oldhead_count(s,raw8,oldheads)[0],accepted_original_nodes=oldhead_count(s,raw8,oldheads)[1])for s in layouts];maximum=max(v['g']for v in variants);winners=[v['nodes']for v in variants if v['g']==maximum]
    need(maximum==a['g']+int(available),'finite old-head max bound from firstmismatch top2side only')
    # Dominance-admissible is only a necessary score constraint; not a target-blind policy.
    dom=[v for v in variants if all(node<4 or node-4 in v['nodes']for node in v['nodes'])];dommax=max(v['g']for v in dom)
    for side in[x for x in chosen if x>=4]:
     parent=parents[side];visited=parent<0 or parent in b['accepted_original_nodes']
     prediction=newheads[0 if parent<0 else rm[parent]+1]
     reason='parent_not_reached'if not visited else'primary_child_chosen'if prediction==raw8[side-4]else'neither_top2_child_matches'
     if side in b['accepted_side_original_nodes']:reason='side_accepted'
     side_reasons[reason]+=1
    row=dict(category=category,owner=i,request_id=owner['request_id'],prefix_offset=owner['output_offset'],source_context_sha256=a['source_context_sha256'],source_tokens=a['source_context_tokens'],
     old_g=a['g'],joint_g=b['g'],delta_g=b['g']-a['g'],old_selected=a['selected_original_nodes'],joint_selected=chosen,old_accepted_original_nodes=a['accepted_original_nodes'],joint_accepted_original_nodes=b['accepted_original_nodes'],
     joint_is_fork=b['true_fork'],accepted_side_original_nodes=b['accepted_side_original_nodes'],first_old_primary_mismatch=None if k==4 else k,available_top2_side_match=available,
     conditional_oldhead_oracle_max=maximum,conditional_oldhead_oracle_winner_layouts=winners,dominance_necessary_subset_oracle_max=dommax,
     observed_raw8_and_scores_equal_between_actual_routes=raw8==jraw and scores==jscores,
     best_joint_side=bestside,best_joint_side_global_rank=ranks.index(bestside)+1,best_joint_side_score=joint[bestside],primary3_joint_score=joint[3],
     raw_conditional_probabilities=jscores,joint_scores=list(joint),joint_global_rank_nodes=ranks)
    if available:
     row['matching_side_node']=k+4;row['matching_side_global_joint_rank']=ranks.index(k+4)+1;row['matching_side_conditional_probability']=scores[k+4];row['matching_side_joint_score']=joint[k+4]
     row['matching_side_selected_by_actual_joint']=k+4 in chosen;row['oracle_requires_alternate_linear0127']=k==3
     opportunities.append(row)
    rows.append(row);score_rows.append(row);id_diffs[owner['request_id']].append(row)
   need(len(rows)==12 and len(id_diffs)==4 and all(sorted(v['prefix_offset']for v in group)==[32,96,192]for group in id_diffs.values()),'actual four identity clusters andthree correlatedoffsets')
   summary=dict(category=category,measured_per_policy=12,independent_source_clusters=4,old_g=[x['old_g']for x in rows],joint_g=[x['joint_g']for x in rows],
    old_g_mean=statistics.mean(x['old_g']for x in rows),joint_g_mean=statistics.mean(x['joint_g']for x in rows),paired_mean_delta=statistics.mean(x['delta_g']for x in rows),
    positive_pairs=sum(x['delta_g']>0 for x in rows),negative_pairs=sum(x['delta_g']<0 for x in rows),same_pairs=sum(x['delta_g']==0 for x in rows),natural_joint_forks=sum(x['joint_is_fork']for x in rows),natural_joint_side_accepts=sum(bool(x['accepted_side_original_nodes'])for x in rows),
    selected_layout_histogram=dict(collections.Counter(','.join(map(str,x['joint_selected']))for x in rows)),first_primary_mismatch_histogram=dict(mismatch),available_oldhead_side_matches=sum(x['available_top2_side_match']for x in rows),
    conditional_oldhead_oracle_mean=statistics.mean(x['conditional_oldhead_oracle_max']for x in rows),dominance_necessary_subset_oracle_mean=statistics.mean(x['dominance_necessary_subset_oracle_max']for x in rows),
    identity_cluster_summaries=[dict(request_id=i,prefix_offsets=[x['prefix_offset']for x in group],old_mean=statistics.mean(x['old_g']for x in group),joint_mean=statistics.mean(x['joint_g']for x in group),delta_mean=statistics.mean(x['delta_g']for x in group))for i,group in sorted(id_diffs.items())],
    observed_shared_root_primary_heads=dict(cohortshared),actual_raw8_scores_identical_pairs=raw_equal)
   summaries.append(summary);paired.extend(rows)
  cost=[];proxies=[]
  for category in('coding','chat','summarization'):
   byrank=[]
   for rank,report in enumerate(reports):
    samples={}
    for policy in(oldpolicy,jointpolicy):
     values=[v['diagnostic_whole_protocol_and_next_HOST_seconds']for v in report['cost']if v['category']==category and v['policy']==policy]
     need(len(values)==3 and all(math.isfinite(x)and x>0 for x in values),'three positive finite rolecost samples')
     samples[policy]=stats(values)
    byrank.append(dict(rank=rank,role=report['role'],policies=samples,median_joint_over_old=samples[jointpolicy]['median']/samples[oldpolicy]['median']))
   cost.append(dict(category=category,by_rank=byrank))
   arms={}
   for policy in(oldpolicy,jointpolicy):
    values=[]
    for repeat in range(3):
     actual=[]
     for rank,report in enumerate(reports):
      indexes=[i for i,row in enumerate(report['cost'])if row['category']==category and row['policy']==policy];idx=indexes[repeat];caseindex=idx//3
      value=data[f'rank{rank}-cost-case{caseindex}-repeat{repeat}.json.gz'];need(all(v==0 for v in value['Native']['delta'].values())and len(value['Native']['delta'])==21,'all21zero currentcost raw')
      need(value['diagnostic_whole_protocol_and_next_HOST_seconds']==report['cost'][idx]['diagnostic_whole_protocol_and_next_HOST_seconds'],'cost summary/raw exact')
      gain=data[f'rank{rank}-gain-case{caseindex}.json.gz'];need(value['work']['source']==gain['work']['source']and value['work']['rows']==gain['work']['rows']and value['work']['selected']==gain['work']['selected']and value['commit']['committed_tokens']==gain['commit']['committed_tokens']and value['next_work']['rows']==gain['next_work']['rows'],'actualfreshcost wholeprotocol/ownnext reproduces qualifiedcase')
      if rank:need(value['heads']==gain['heads'],'all80cost heads matchsame-qualifiedcase')
      actual.append(value['diagnostic_whole_protocol_and_next_HOST_seconds'])
     values.append(max(actual))
    arms[policy]=stats(values)
   proxies.append(dict(category=category,max_across_four_role_samples_only=True,role_times_not_added=True,arms=arms,median_joint_over_old=arms[jointpolicy]['median']/arms[oldpolicy]['median']))
  naturalfork=sum(x['natural_joint_forks']for x in summaries);sides=sum(x['natural_joint_side_accepts']for x in summaries)
  need((naturalfork,sides)==(9,0),'actual direct subsetreason');original=r['actual_bounded_audit']['new_B4_joint'];need(original['actual_joint_measured_fork_rows']==naturalfork and original['actual_joint_measured_side_accept_rows']==sides and original['status']=='passed_subset','actualretained audit subsetequation')
  need(sum(x['available_oldhead_side_matches']for x in summaries)==10 and not any(x['matching_side_selected_by_actual_joint']for x in opportunities),'tenavailable oldheadside opportunities allmissed byobservedselection')
  need(all(sha(p)==h for p,h in pins.items()),'raw snapshot exact before/after')
  result.update(status='completed_bounded_analysis',actual_subset_reason=dict(natural_joint_measured_forks=naturalfork,natural_joint_measured_side_accepts=sides,source_rule='passed_bounded_diagnostic requires both naturalfork>0 andnaturalside>0; otherwisepassed_subset',controlled_geometry_side_accepts_per_case=dict(geometry_sides),controlled_is_not_natural_gain=True),
   independent_original_integer_verifier_calls=integer_count,per_class=summaries,measured_pairs=paired,selected_side_parent_outcomes=dict(side_reasons),observed_shared_root_and_primary_heads=dict(shared),
   conditional_oldhead_opportunity=dict(actual_available_top2_matches=10,matching_side_opportunities=opportunities,oracle_global_mean=statistics.mean(x['conditional_oldhead_oracle_max']for x in paired),old_actual_global_mean=statistics.mean(x['old_g']for x in paired),joint_actual_global_mean=statistics.mean(x['joint_g']for x in paired),
    oracle_is_not_actual_newg=True,uses_only_queried_old_primary_heads=True,no_side_node_bonus_token_value_invented=True,side_accept_terminal_bonus_count_only=True,
    caveat='Target-awareconditional upperproxy over8legal4subsets. Full9/FIA arithmetic maydiffer, andunqueriedside-head value/EOSbehavior isunknown. Notanunrunmodelresult,notafutureselector,notproductiong. k3oracleuses0127, impossibleforcurrentgreedyjointtop4 becausetop1p3>=p7 sameparent.'),
   all_rank_costs=cost,max_role_cohort_proxies=proxies,raw_pins_file=dict(path=str(EVIDENCE/'RAW_PINS.json'),sha256=sha(EVIDENCE/'RAW_PINS.json'),files=len(pins)),
   timing_scope=dict(source=str(P/'actor206.py'),initial_prefill_outside=True,realrepair_refine_target_verdict_commit_compaction_and_ownnext_inside=True,full_independent_observer_oracle_outside=True,
    target_head_clone_shape_key_flag_checks_inside=True,all4_stage_and_drains_inside=True,order='codingold3thenjoint3,chatold3thenjoint3,summaryold3thenjoint3; notABBA',role_seconds_never_added=True,pureproduction_or_device_time=False,GP_inferred=False),
   limitations=['Only36measuredprefixes perpolicy:4sourceidentities/class x3correlated offsets; fillers excluded. Noheldoutpopulation orallAR claim.',
    'Sourceprefixes areteacherforced retainedcontexts; initialroot isnotindependently regenerated canonicalAR. Allpolicybranches executeowncurrentmodel/cache/protocol, butthisisboundeddiagnostic,notfull463production.',
    'Nochangedgtable/posteriorinstalled. Natural9fork/no-side ledsubset; geometrycontrolledsideaccepts orcounterfactualold-headoracles cannotupgradeit.',
    'Smallordereddiagnosticcost differences includeclones/checks/stagefences andnextproducer; theycannotpredictend-to-endGP,1.5x,orcausalbenefit.'],
   source_snapshot={str(p):sha(p)for p in EVIDENCE.iterdir()if p.is_file()})
 except BaseException:result['errors'].append(traceback.format_exc())
 need(not any(x.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','triton','acl')for x in sys.modules),'no backend imported')
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),status=result['status'],errors=result['errors'])));return int(bool(result['errors']))
if __name__=='__main__':raise SystemExit(main())
