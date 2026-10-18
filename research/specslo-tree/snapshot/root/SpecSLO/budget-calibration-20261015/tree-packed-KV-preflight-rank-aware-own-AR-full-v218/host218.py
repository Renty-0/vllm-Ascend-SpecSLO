"""HOST checks for exact cached bodies, spawn proxy and terminal supervision."""
import ast,contextlib,hashlib,io,json,pickle,sys,tempfile,types,unittest,copy,math,gzip
from pathlib import Path
import runtime218,extension218,guard218
from check218 import P,DONOR,B
import base209  # Establish the same canonical148/160/173 import paths as actual worker.


class Tests(unittest.TestCase):
    def test_real_mapper_bodies_equivalent_without_IO(self):
        import mapper209
        original=dict(vars(mapper209));calls=[];sentinel=object()
        class Parent:
            def __init__(self,*args,**kwargs):self.engine=None
        parent=types.SimpleNamespace(__file__='HOST-original-parent',controls_class=lambda:Parent,
            dispatch=lambda control,n,q,f:(sentinel,n,q,f))
        Controls,state,dispatch=extension218.cached_controls(mapper209,parent)
        control=Controls();self.assertIs(control.mapper._v209_control,control)
        self.assertEqual(dispatch.__code__.co_code,mapper209.dispatch.__code__.co_code)
        reference=types.FunctionType(mapper209.dispatch.__code__,dict(vars(mapper209),parent_module=lambda name:parent))
        for n in range(1,33):
            for q in (2*n,4*n,5*n):
                for family in('causal','full9'):
                    self.assertEqual(control.mapper(n,q,family),reference(control,n,q,family))
        control.engine=types.SimpleNamespace(_v209_usage_geometry=mapper209._ISSUER)
        for n in range(9,16):self.assertEqual(control.mapper(n,5*n,'causal'),(80,16,[5]*(16-n)))
        with self.assertRaises(RuntimeError):dispatch.__globals__['parent_module']('unqualified-foreign-module')
        self.assertGreater(state['cached_parent_returns'],100)
        self.assertEqual(state['initial_parent_source_checks'],1)
        self.assertTrue(all(vars(mapper209)[k]is value for k,value in original.items()))
        self.assertNotIn('torch',sys.modules)

    def test_spawn_and_exact_runtime_projection(self):
        saved=runtime218.ready
        try:
            runtime218.ready=lambda:None;r=runtime218.build_runtime()
            self.assertIs(pickle.loads(pickle.dumps(runtime218.worker)),runtime218.worker)
            self.assertIs(r['main'].__globals__['worker'],runtime218.worker)
            self.assertEqual(r['worker'].__kwdefaults__['extension_module'],'extension218')
            source=(DONOR/'runtime209.py').read_text();changed=source
            for old,new,count in r['source_projection']:
                self.assertEqual(changed.count(old),count);changed=changed.replace(old,new)
            for old,new,count in reversed(r['source_projection']):
                self.assertEqual(changed.count(new),count);changed=changed.replace(new,old)
            self.assertEqual(changed,source)
            self.assertNotIn('torch',sys.modules)
        finally:runtime218.ready=saved;runtime218._RUNTIME=None

    def test_original_supervision_all_terminal_paths(self):
        for mode in('success','child_fail','pre_fail','post_source_fail'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);source=root/'source';source.write_text('original')
                events=[];h=types.SimpleNamespace(device_state=lambda:dict(all_idle=True))
                ns=guard218.execution_namespace()
                ns.update(P=root,MAX_SECONDS=10,MAX_TAG_BYTES=1024**2,SCOPE=dict(HOST_stub_only=True),
                    prepared=lambda:({}, {str(source):hashlib.sha256(b'original').hexdigest()},{}),
                    storage=lambda tag:dict(bytes=0),verifier_report=lambda r:r,
                    launch_argv=lambda tag:['/usr/bin/python3','-c','raise SystemExit('+('1'if mode=='child_fail'else'0')+')'])
                def verify(tag,label):
                    events.append(label)
                    if mode=='pre_fail'and label=='pre':raise RuntimeError('HOST pre-verifier failure')
                    return dict(exit_code=0,actual_result=dict(status='pass'))
                def save(path,row):path.write_text(json.dumps(row))
                def cleanup(child,helper,row):
                    events.append('cleanup');row['owned_remaining']=[]
                    if child is not None:child.wait(timeout=5)
                def audit(actual):
                    if mode=='post_source_fail':source.write_text('changed')
                    return {},'HOST_only',{}
                ns.update(original_helpers=lambda:dict(save=save,helper=lambda:h,verify=verify,owned_cleanup=cleanup),actual_audit=audit)
                with contextlib.redirect_stdout(io.StringIO()):
                    try:ns['execute_locked']('HOST-'+mode)
                    except SystemExit as e:self.assertEqual(e.code,1)
                row=json.loads((root/('receipt-HOST-'+mode+'.json')).read_text())
                self.assertEqual(row['status'],'complete'if mode=='success'else'failed')
                self.assertEqual(events,['pre','cleanup','post']);self.assertTrue(row['devices_after']['all_idle'])
                self.assertEqual(row['owned_remaining'],[])

    def test_terminal_audit_loads_actual_new_fields(self):
        old=guard218.donor_guard;old_AR=guard218.reference_audit
        guard218.reference_audit=lambda directory:dict(HOST_stub_only=True)
        try:
            donor=types.SimpleNamespace(actual_audit=lambda p:({},'passed_full',{}))
            guard218.donor_guard=lambda:donor
            with tempfile.TemporaryDirectory()as directory:
                p=Path(directory)
                for rank in range(4):
                    r=dict(overlay=dict(ROOT218_overlay='cached-parent-SLO-admission1-cost-aware-N9to16',SLO_AWARE_ADMISSION=True,cost_aware_N9to16_budget=True,packed_target_KV_preflight=True),
                        ROOT218_cached_parent_dispatch=dict(initial_parent_source_checks=1,cached_parent_returns=20,canonical_mapper_or_base_module_mutated=False,cost_aware_N9to16_budget=dict(private_shape_bindings=1,by_generation={'5':dict(shape_calls=1)},original_cost_table=[dict(N=n,Q=q,M=m,S=ss)for(n,q),(m,ss)in sorted(__import__('cost218').original_cost_table().items())])),
                        actual_helper_origins_before_constructor=dict(ROOT218_explicit_overlay_sources={'explicit':1}),
                        actual_helper_origins_after_constructor=dict(ROOT218_explicit_overlay_sources={'explicit':1}),postfull_independent_FIA_AR=dict(completed=True),ROOT218_slot_preflight=dict(shadow=False,qualified=True,shadow_pairs=1,shadow_draft_pairs=int(rank==0),shadow_target_pairs=int(rank!=0),by_generation={'5':dict(calls=1,direct_packed_target_calls=int(rank!=0),unchanged_other_calls=int(rank==0))}))
                    (p/f'rank{rank}-local-diagnostic.json').write_text(json.dumps(r))
                    summary_folder=p/'postfull-independent-FIA-AR';summary_folder.mkdir(exist_ok=True)
                    (summary_folder/f'rank{rank}-summary.json').write_text(json.dumps(r['postfull_independent_FIA_AR']))
                    folder=p/'windows/window32/full463';folder.mkdir(parents=True,exist_ok=True)
                    trace=dict(source_generation=5,normal_target_calls=[{}],decode_counters=dict(spec_rhythm_slo_aware_admission_enabled=1,spec_rhythm_target_slo_first_enabled=0))
                    (folder/f'rank{rank}-full463-actual-trace.json.gz').write_bytes(gzip.compress(json.dumps(trace).encode()))
                self.assertEqual(guard218.actual_audit(p)[1],'passed_full')
                r['ROOT218_cached_parent_dispatch']['initial_parent_source_checks']=0
                (p/'rank3-local-diagnostic.json').write_text(json.dumps(r))
                with self.assertRaises(RuntimeError):guard218.actual_audit(p)
                r['ROOT218_cached_parent_dispatch']['initial_parent_source_checks']=1
                r['overlay']['SLO_AWARE_ADMISSION']=False
                (p/'rank3-local-diagnostic.json').write_text(json.dumps(r))
                with self.assertRaises(RuntimeError):guard218.actual_audit(p)
                r['overlay']['SLO_AWARE_ADMISSION']=True
                (p/'rank3-local-diagnostic.json').write_text(json.dumps(r))
                trace['decode_counters']['spec_rhythm_slo_aware_admission_enabled']=0
                (folder/'rank3-full463-actual-trace.json.gz').write_bytes(gzip.compress(json.dumps(trace).encode()))
                with self.assertRaises(RuntimeError):guard218.actual_audit(p)
        finally:guard218.donor_guard=old;guard218.reference_audit=old_AR

    def test_original_admission_policy_and_exact_enabled_flag(self):
        native=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
        names={'order_spec_rhythm_admission_candidates','order_spec_rhythm_admission_params'}
        nodes=[copy.deepcopy(n)for n in ast.parse(native.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
        self.assertEqual({n.name for n in nodes},names)
        for node in nodes:
            node.returns=None
            for arg in (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs):arg.annotation=None
        namespace=dict(math=math)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<unaltered-original-admission>','exec'),namespace)
        params=[types.SimpleNamespace(slo_tpot_ms=s,arrival_ts=a)for s,a in[(50,2.),(150,0.),(40,3.),(40,1.)]]
        before=[(p.slo_tpot_ms,p.arrival_ts)for p in params];order=namespace['order_spec_rhythm_admission_params']
        self.assertEqual(order(range(4),request_params=params,enabled=False),(0,1,2,3))
        self.assertEqual(order(range(4),request_params=params,enabled=True),(3,2,0,1))
        self.assertEqual(before,[(p.slo_tpot_ms,p.arrival_ts)for p in params])
        startup=(P/'startup218.sh').read_text();self.assertEqual(startup.count('VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1'),1)
        self.assertNotIn('VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=0',startup)




class CostTests(unittest.TestCase):
    def plan(self, budgets=None, *, caps=None, available=192, normal_limit=96, eager=()):
        from collections import Counter
        from split_budget_v132 import DraftWorkPlan,DraftWorkLimit,TargetAdmissionLimit
        budgets=dict(enumerate([3,4]*8)) if budgets is None else dict(budgets)
        ids=tuple(budgets);all_ids=ids+tuple(eager)
        policy=types.SimpleNamespace(counts=Counter({f'grant_B{b}':list(budgets.values()).count(b) for b in range(1,6)}))
        eb={i:3 for i in eager}
        plan=DraftWorkPlan(plan_id=7,target_limit=TargetAdmissionLimit(),
            work_limit=DraftWorkLimit(normal_candidates=normal_limit,normal_per_request=4),
            available_selected_candidates=available,requested_normal=ids,admitted_eager=tuple(eager),
            normal_budgets=budgets,eager_budgets=eb,progress_gaps=dict.fromkeys(all_ids,1.),
            eager_priorities=dict.fromkeys(eager,1.),caps=dict.fromkeys(all_ids,4) if caps is None else caps,
            needs=dict.fromkeys(all_ids,1),reserve_requested=sum(eb.values()),reserve_used=sum(eb.values()),
            reserved_by_owner=eb,raw_reserved_by_owner=eb,raw_eager_budgets=eb)
        return policy,plan

    def test_original_typed_plan_and_budget_accounting(self):
        from collections import Counter
        from cost218 import repair_plan
        policy,plan=self.plan(eager=(16,17));counts=Counter()
        new=repair_plan(policy,plan,context_len=4000,counts=counts)
        self.assertEqual(list(plan.normal_budgets.values()),[3,4]*8)
        self.assertEqual(list(new.normal_budgets.values()),[4]*16)
        self.assertEqual(new.allocated_selected_candidates,70)
        self.assertEqual(dict(new.eager_budgets),dict(plan.eager_budgets))
        self.assertEqual(new.reserve_used,plan.reserve_used)
        self.assertEqual(dict(new.raw_reserved_by_owner),dict(plan.raw_reserved_by_owner))
        self.assertEqual(policy.counts['grant_B3'],0);self.assertEqual(policy.counts['grant_B4'],16)
        self.assertEqual(counts['added_selected_candidates'],8)
        for field in plan.__dataclass_fields__:
            if field!='normal_budgets':self.assertEqual(getattr(new,field),getattr(plan,field))
        self.assertTrue(new.require_work(list(range(18)),[4]*16+[3,3],[False]*16+[True,True],
            [types.SimpleNamespace(candidate_budget=b)for b in [4]*16+[3,3]]))

    def test_cap_context_role_and_global_limit_block(self):
        from collections import Counter
        from cost218 import repair_plan
        cases=[dict(caps={i:(3 if i==0 else 4)for i in range(16)}),dict(available=56),dict(normal_limit=56)]
        for kw in cases:
            with self.subTest(kw=kw):
                policy,plan=self.plan(**kw);counts=Counter();before=dict(policy.counts)
                self.assertIs(repair_plan(policy,plan,context_len=4092,counts=counts),plan)
                self.assertEqual(dict(policy.counts),before);self.assertEqual(counts['blocked_by_original_limits'],1)
        policy,plan=self.plan();counts=Counter()
        self.assertIs(repair_plan(policy,plan,context_len=4093,counts=counts),plan)

    def test_scope_excludes_shared3_homogeneous4_other_sizes_and_B2(self):
        from collections import Counter
        from cost218 import repair_plan
        for values in ([3]*16,[4]*16,[3,4]*4,[3,4]*8+[3], [2]+[4]*15):
            policy,plan=self.plan(dict(enumerate(values)));counts=Counter()
            self.assertIs(repair_plan(policy,plan,context_len=100,counts=counts),plan)
            self.assertEqual(counts['eligible_mixed_N9to16'],0)

    def test_exact_private_function_and_original_closure_retained(self):
        from collections import Counter
        from cost218 import bind_candidate_shape
        import split_shape_v132
        original=dict(vars(split_shape_v132));policy,plan=self.plan()
        def solver(p,s,**kw):self.assertIs(p,policy);return plan
        shape=types.FunctionType(split_shape_v132.shape.__code__,dict(vars(split_shape_v132),
            shape_draft_work=solver,select_normal=lambda *a,**k:types.SimpleNamespace(admitted=tuple(range(16))),
            finish_normal=lambda *a:None))
        ns={'shape':shape}
        exec('def factory(policy):\n def candidate_shape(instance,**kwargs):return shape(policy,instance,**kwargs)\n return candidate_shape',ns)
        old=ns['factory'](policy);control=types.SimpleNamespace(candidate_shape=old,engine=types.SimpleNamespace(_dynamic_tree_generation=5))
        state=dict(private_shape_bindings=0,by_generation={});new=bind_candidate_shape(control,state)
        self.assertIs(new.__closure__,old.__closure__);self.assertEqual(new.__code__.co_code,old.__code__.co_code)
        self.assertIs(new.__globals__['shape'].__code__,shape.__code__)
        self.assertIs(new.__globals__['shape'].__globals__['refine_draft_groups'],shape.__globals__['refine_draft_groups'])
        states={i:types.SimpleNamespace(max_gamma=5) for i in range(16)}
        value=new(types.SimpleNamespace(max_gamma=5),normal_request_indices=tuple(range(16)),eager_request_indices=(),states=states,context_len=4092)
        self.assertEqual(list(value.normal_budgets.values()),[4]*16)
        self.assertEqual(state['by_generation']['5']['repaired_calls'],1)
        self.assertTrue(all(vars(split_shape_v132)[k]is v for k,v in original.items()))

    def test_all_current_qualified_sizes_and_both_real_cost_cliffs(self):
        from collections import Counter
        from cost218 import repair_plan,original_cost_table
        for n in range(9,17):
            for fours in range(1,n):
                budgets=dict(enumerate([4]*fours+[3]*(n-fours)))
                policy,plan=self.plan(budgets);counts=Counter()
                new=repair_plan(policy,plan,context_len=4092,counts=counts)
                self.assertEqual(list(new.normal_budgets.values()),[4]*n)
                old_m,_=original_cost_table()[n,4*n+fours]
                self.assertEqual(counts['proposed_target_physical_queries_saved'],old_m-80)
                self.assertEqual(policy.counts['grant_B3'],0);self.assertEqual(policy.counts['grant_B4'],n)
        self.assertEqual(original_cost_table()[9,37][0],96)
        self.assertEqual(original_cost_table()[9,44][0],128)

    def test_actual_mapper_cost_discontinuity(self):
        import mapper173
        self.assertEqual(mapper173.expected_geometry(16,80,'causal')[:2],(80,16))
        for q in range(65,80):self.assertEqual(mapper173.expected_geometry(16,q,'causal')[:2],(128,38))

    def test_actual_generation_and_invalid_accounting_rejected(self):
        from cost218 import audit_snapshot,original_cost_table
        cache=dict(cost_aware_N9to16_budget=dict(private_shape_bindings=1,by_generation={'5':dict(shape_calls=12,eligible_mixed_N9to16=4,repaired_calls=3,repaired_N16_calls=3,repaired_old_M128_calls=3,blocked_by_original_limits=1,upgraded_owners=8,added_selected_candidates=8,proposed_target_physical_queries_saved=144)}))
        cache['cost_aware_N9to16_budget']['original_cost_table']=[dict(N=n,Q=q,M=m,S=ss)for(n,q),(m,ss)in sorted(original_cost_table().items())]
        self.assertTrue(audit_snapshot(cache,5))
        with self.assertRaises(RuntimeError):audit_snapshot(cache,6)
        cache['cost_aware_N9to16_budget']['by_generation']['5']['added_selected_candidates']=9
        with self.assertRaises(RuntimeError):audit_snapshot(cache,5)




class ReferenceTests(unittest.TestCase):
    def test_actual_original_artifact_descriptor_and_bad_hash_rejection(self):
        from reference218 import retained_trace
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'rank1-full463-actual-trace.json.gz'
            p.write_bytes(gzip.compress(json.dumps(dict(rank=1,scope_count=463)).encode()))
            info=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            windows=dict(traces=[dict(original_trace=dict(role='full463',scope_count=463,artifact=info))])
            artifact,value=retained_trace(windows);self.assertEqual(artifact,p);self.assertEqual(value['rank'],1)
            info['sha256']='0'*64
            with self.assertRaises(RuntimeError):retained_trace(windows)


    def test_reference_frame_only_own_frontier_and_causal_layout(self):
        from reference218 import frame
        prompts=[[10,11,12],[20]*130];outputs=[[40,41],[50]]
        x=frame(prompts,outputs,[1,0]);self.assertEqual(x['tokens'][:10],[50,0,0,0,0,41,0,0,0,0])
        self.assertEqual(x['positions'][:10],list(range(130,135))+list(range(4,9)))
        self.assertEqual(x['ids'][:10],[1]*5+[0]*5)
        self.assertEqual(x['sequence_lens'][:2],[135,9]);self.assertEqual(x['real_head_rows'],[0,5])
        self.assertEqual(x['tokens'][10:],[0]*70);self.assertEqual(x['ids'][10:15],[463]*5)
        alt=frame(prompts,outputs,[1,0],future=151643)
        self.assertEqual(alt['tokens'][::5],x['tokens'][::5])
        self.assertEqual(prompts,[[10,11,12],[20]*130]);self.assertEqual(outputs,[[40,41],[50]])
        with self.assertRaises(RuntimeError):frame(prompts,outputs,[])
        with self.assertRaises(RuntimeError):frame(prompts,outputs,[0,0])
        with self.assertRaises(RuntimeError):frame([[0]*4092],[[1,2]],[0])

    def test_complete_exact_comparison_never_turns_mismatch_to_success(self):
        from reference218 import compare
        a={'a':[1]*256,'b':[2]*256};b={'a':[1]*256,'b':[2]*255+[3]}
        r=compare(a,b);self.assertFalse(r['exact_all_tokens_equal']);self.assertEqual(r['equal_requests'],1)
        self.assertEqual(r['rows'][1]['first_mismatch_zero_based'],255)
        with self.assertRaises(RuntimeError):compare(a,{'a':[1]*256})
        with self.assertRaises(RuntimeError):compare({'a':[1]*255},{'a':[1]*255})


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    print(json.dumps(dict(status='passed'if result.wasSuccessful()else'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),Torch_imported='torch'in sys.modules,NPU_used=False)))
    raise SystemExit(0 if result.wasSuccessful()else 1)
