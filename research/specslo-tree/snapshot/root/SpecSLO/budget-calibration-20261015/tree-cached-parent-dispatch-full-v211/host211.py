"""HOST checks for exact cached bodies, spawn proxy and terminal supervision."""
import ast,contextlib,hashlib,io,json,pickle,sys,tempfile,types,unittest
from pathlib import Path
import runtime211,extension211,guard211
from check211 import P,DONOR,B


class Tests(unittest.TestCase):
    def test_real_mapper_bodies_equivalent_without_IO(self):
        import mapper209
        original=dict(vars(mapper209));calls=[];sentinel=object()
        class Parent:
            def __init__(self,*args,**kwargs):self.engine=None
        parent=types.SimpleNamespace(__file__='HOST-original-parent',controls_class=lambda:Parent,
            dispatch=lambda control,n,q,f:(sentinel,n,q,f))
        Controls,state,dispatch=extension211.cached_controls(mapper209,parent)
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
        saved=runtime211.ready
        try:
            runtime211.ready=lambda:None;r=runtime211.build_runtime()
            self.assertIs(pickle.loads(pickle.dumps(runtime211.worker)),runtime211.worker)
            self.assertIs(r['main'].__globals__['worker'],runtime211.worker)
            self.assertEqual(r['worker'].__kwdefaults__['extension_module'],'extension211')
            source=(DONOR/'runtime209.py').read_text();changed=source
            for old,new,count in r['source_projection']:
                self.assertEqual(changed.count(old),count);changed=changed.replace(old,new)
            for old,new,count in reversed(r['source_projection']):
                self.assertEqual(changed.count(new),count);changed=changed.replace(new,old)
            self.assertEqual(changed,source)
            self.assertNotIn('torch',sys.modules)
        finally:runtime211.ready=saved;runtime211._RUNTIME=None

    def test_original_supervision_all_terminal_paths(self):
        for mode in('success','child_fail','pre_fail','post_source_fail'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);source=root/'source';source.write_text('original')
                events=[];h=types.SimpleNamespace(device_state=lambda:dict(all_idle=True))
                ns=guard211.execution_namespace()
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
        old=guard211.donor_guard
        try:
            donor=types.SimpleNamespace(actual_audit=lambda p:({},'passed_full',{}))
            guard211.donor_guard=lambda:donor
            with tempfile.TemporaryDirectory()as directory:
                p=Path(directory)
                for rank in range(4):
                    r=dict(overlay=dict(ROOT211_overlay='cached-parent-mapper-only'),
                        ROOT211_cached_parent_dispatch=dict(initial_parent_source_checks=1,cached_parent_returns=20,canonical_mapper_or_base_module_mutated=False),
                        actual_helper_origins_before_constructor=dict(ROOT211_explicit_overlay_sources={'explicit':1}),
                        actual_helper_origins_after_constructor=dict(ROOT211_explicit_overlay_sources={'explicit':1}))
                    (p/f'rank{rank}-local-diagnostic.json').write_text(json.dumps(r))
                self.assertEqual(guard211.actual_audit(p)[1],'passed_full')
                r['ROOT211_cached_parent_dispatch']['initial_parent_source_checks']=0
                (p/'rank3-local-diagnostic.json').write_text(json.dumps(r))
                with self.assertRaises(RuntimeError):guard211.actual_audit(p)
        finally:guard211.donor_guard=old


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps(dict(status='passed'if result.wasSuccessful()else'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),Torch_imported='torch'in sys.modules,NPU_used=False)))
    raise SystemExit(0 if result.wasSuccessful()else 1)
