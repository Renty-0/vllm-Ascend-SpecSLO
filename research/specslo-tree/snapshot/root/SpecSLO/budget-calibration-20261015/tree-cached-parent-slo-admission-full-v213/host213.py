"""HOST checks for exact cached bodies, spawn proxy and terminal supervision."""
import ast,contextlib,hashlib,io,json,pickle,sys,tempfile,types,unittest,copy,math,gzip
from pathlib import Path
import runtime213,extension213,guard213
from check213 import P,DONOR,B


class Tests(unittest.TestCase):
    def test_real_mapper_bodies_equivalent_without_IO(self):
        import mapper209
        original=dict(vars(mapper209));calls=[];sentinel=object()
        class Parent:
            def __init__(self,*args,**kwargs):self.engine=None
        parent=types.SimpleNamespace(__file__='HOST-original-parent',controls_class=lambda:Parent,
            dispatch=lambda control,n,q,f:(sentinel,n,q,f))
        Controls,state,dispatch=extension213.cached_controls(mapper209,parent)
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
        saved=runtime213.ready
        try:
            runtime213.ready=lambda:None;r=runtime213.build_runtime()
            self.assertIs(pickle.loads(pickle.dumps(runtime213.worker)),runtime213.worker)
            self.assertIs(r['main'].__globals__['worker'],runtime213.worker)
            self.assertEqual(r['worker'].__kwdefaults__['extension_module'],'extension213')
            source=(DONOR/'runtime209.py').read_text();changed=source
            for old,new,count in r['source_projection']:
                self.assertEqual(changed.count(old),count);changed=changed.replace(old,new)
            for old,new,count in reversed(r['source_projection']):
                self.assertEqual(changed.count(new),count);changed=changed.replace(new,old)
            self.assertEqual(changed,source)
            self.assertNotIn('torch',sys.modules)
        finally:runtime213.ready=saved;runtime213._RUNTIME=None

    def test_original_supervision_all_terminal_paths(self):
        for mode in('success','child_fail','pre_fail','post_source_fail'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);source=root/'source';source.write_text('original')
                events=[];h=types.SimpleNamespace(device_state=lambda:dict(all_idle=True))
                ns=guard213.execution_namespace()
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
        old=guard213.donor_guard
        try:
            donor=types.SimpleNamespace(actual_audit=lambda p:({},'passed_full',{}))
            guard213.donor_guard=lambda:donor
            with tempfile.TemporaryDirectory()as directory:
                p=Path(directory)
                for rank in range(4):
                    r=dict(overlay=dict(ROOT213_overlay='cached-parent-and-SLO-admission1',SLO_AWARE_ADMISSION=True),
                        ROOT213_cached_parent_dispatch=dict(initial_parent_source_checks=1,cached_parent_returns=20,canonical_mapper_or_base_module_mutated=False),
                        actual_helper_origins_before_constructor=dict(ROOT213_explicit_overlay_sources={'explicit':1}),
                        actual_helper_origins_after_constructor=dict(ROOT213_explicit_overlay_sources={'explicit':1}))
                    (p/f'rank{rank}-local-diagnostic.json').write_text(json.dumps(r))
                    folder=p/'windows/window32/full463';folder.mkdir(parents=True,exist_ok=True)
                    trace=dict(decode_counters=dict(spec_rhythm_slo_aware_admission_enabled=1,spec_rhythm_target_slo_first_enabled=0))
                    (folder/f'rank{rank}-full463-actual-trace.json.gz').write_bytes(gzip.compress(json.dumps(trace).encode()))
                self.assertEqual(guard213.actual_audit(p)[1],'passed_full')
                r['ROOT213_cached_parent_dispatch']['initial_parent_source_checks']=0
                (p/'rank3-local-diagnostic.json').write_text(json.dumps(r))
                with self.assertRaises(RuntimeError):guard213.actual_audit(p)
                r['ROOT213_cached_parent_dispatch']['initial_parent_source_checks']=1
                r['overlay']['SLO_AWARE_ADMISSION']=False
                (p/'rank3-local-diagnostic.json').write_text(json.dumps(r))
                with self.assertRaises(RuntimeError):guard213.actual_audit(p)
                r['overlay']['SLO_AWARE_ADMISSION']=True
                (p/'rank3-local-diagnostic.json').write_text(json.dumps(r))
                trace['decode_counters']['spec_rhythm_slo_aware_admission_enabled']=0
                (folder/'rank3-full463-actual-trace.json.gz').write_bytes(gzip.compress(json.dumps(trace).encode()))
                with self.assertRaises(RuntimeError):guard213.actual_audit(p)
        finally:guard213.donor_guard=old

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
        startup=(P/'startup213.sh').read_text();self.assertEqual(startup.count('VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1'),1)
        self.assertNotIn('VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=0',startup)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps(dict(status='passed'if result.wasSuccessful()else'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),Torch_imported='torch'in sys.modules,NPU_used=False)))
    raise SystemExit(0 if result.wasSuccessful()else 1)
