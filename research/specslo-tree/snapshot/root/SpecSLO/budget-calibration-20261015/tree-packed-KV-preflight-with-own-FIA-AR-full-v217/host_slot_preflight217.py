"""Execute the actual original/new preflight AST against integer tensor fixtures."""
import ast,copy,importlib.util,json,sys,types,typing,unittest
from collections import Counter
from pathlib import Path
import slot_preflight217 as subject


class Tensor:
    def __init__(self,values,*,dtype='i32',device='cpu',ops=None,storage=None,offset=0,count=None,ndim=1):
        self.values=list(values)if storage is None else storage
        self.offset=offset;self.count=len(self.values)-offset if count is None else count
        self.dtype=dtype;self.device=device;self.ndim=ndim;self.ops=ops if ops is not None else Counter()
    @property
    def shape(self):return(self.count,)
    def numel(self):return self.count
    def __getitem__(self,key):
        start,end,step=key.indices(self.count)
        assert step==1
        return Tensor([],dtype=self.dtype,device=self.device,ops=self.ops,storage=self.values,
                      offset=self.offset+start,count=end-start)
    def to(self,*,dtype):
        self.ops['to_calls']+=1
        if dtype==self.dtype:return self
        self.ops['conversions']+=1
        return Tensor(self.raw(),dtype=dtype,device=self.device,ops=self.ops)
    def detach(self):return self
    def cpu(self):self.ops['cpu_copies']+=1;return self
    def tolist(self):return self.raw()
    def raw(self):return self.values[self.offset:self.offset+self.count]
    def data_ptr(self):return id(self.values)+self.offset
    def storage_offset(self):return self.offset


def namespace(ops):
    def cat(values):
        ops['cat_calls']+=1
        return Tensor([v for row in values for v in row.raw()],dtype=values[0].dtype,ops=ops)
    def deferred(*args):ops['original_deferred_validator']+=1
    return dict(torch=types.SimpleNamespace(Tensor=Tensor,int32='i32',int64='i64',long='i64',cat=cat),
        SpecRhythmPipelineController=types.SimpleNamespace,TreeSpeculationPlan=types.SimpleNamespace,
        Mapping=typing.Mapping,Sequence=typing.Sequence,Any=typing.Any,
        _dt_validate_deferred_eager=deferred)


def fixture(budgets,ops,*,dtype='i32',draft=False):
    plans=[types.SimpleNamespace(candidate_budget=n,parent_indices=types.SimpleNamespace(numel=lambda n=n:n))for n in budgets]
    raw=Tensor(list(range(7,7+sum(n+1 for n in budgets))),dtype=dtype,ops=ops)
    controller=types.SimpleNamespace(validate_verification=lambda i:types.SimpleNamespace(proposal_id=100+i),staged_eager={})
    engine=types.SimpleNamespace(is_draft=draft,device='cpu',model=object(),_tree_cache_capacity=69632,
        _spec_rhythm_tree_cache_device=lambda device:device)
    payloads={100+i:dict(draft_kv_materialized=True)for i in range(len(budgets))}
    mappings={};cursor=0
    for i,n in enumerate(budgets):mappings[100+i]=raw[cursor:cursor+n+1];cursor+=n+1
    return(engine,controller,payloads,mappings,list(range(len(budgets))),plans,None if draft else dict(cache_slot_mapping=raw))


class Tests(unittest.TestCase):
    def test_reconstruction_matches_actual_original_canonical_installer(self):
        ops=Counter();ns=namespace(ops);tree=ast.parse(subject.NATIVE.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine')
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==subject.NAME))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(subject.NATIVE),'exec'),ns)
        native=types.ModuleType('_HOST217_native');native.__dict__.update(ns)
        native.NativePearlEngine=type('NativePearlEngine',(),{subject.NAME:ns[subject.NAME]})
        spec=importlib.util.spec_from_file_location('_HOST217_original_canonical',subject.CANONICAL)
        canonical=importlib.util.module_from_spec(spec);spec.loader.exec_module(canonical)
        canonical.install(native)
        actual=getattr(native.NativePearlEngine,subject.NAME)
        expected,_=subject.functions(namespace(Counter()))
        self.assertEqual(subject.fingerprint(actual.__code__),subject.fingerprint(expected.__code__))

    def test_all_target_sizes_dtype_formats_and_eliminated_operations(self):
        for n in range(1,33):
            for dtype in('i32','i64'):
                for host in(False,True):
                    with self.subTest(n=n,dtype=dtype,host=host):
                        ops=Counter();before,after=subject.functions(namespace(ops))
                        args=fixture([min(5,96//n)]*n,ops,dtype=dtype)
                        old=before(*args,return_host_mappings=True)
                        original_counts=dict(ops);ops.clear()
                        new=after(*args,return_host_mappings=True)
                        self.assertTrue(subject.compare_results(old,new))
                        self.assertEqual(original_counts['to_calls'],n)
                        self.assertEqual(original_counts['cat_calls'],1)
                        self.assertEqual(ops['to_calls'],ops['cat_calls'])
                        self.assertEqual(ops['to_calls'],0);self.assertEqual(ops['cpu_copies'],1)
                        returned=after(*args,return_host_mappings=host)
                        self.assertEqual(returned[1]if host else set(returned),new[1]if host else set(new[0]))

    def test_all_original_bad_mapping_guards_still_reject(self):
        def mutate(args,kind):
            raw=args[-1]['cache_slot_mapping']
            if kind=='ndim':raw.ndim=2
            elif kind=='short':raw.count-=1
            elif kind=='float':raw.dtype='float'
            elif kind=='device':raw.device='wrong'
            elif kind=='negative':raw.values[0]=-1
            elif kind=='aliased':raw.values[1]=raw.values[0]
            elif kind=='capacity':raw.values[0]=69632
            elif kind=='large64':raw.values[0]=2**40
            elif kind=='packed_plan':args[-2][0].candidate_budget+=1
        for kind in('ndim','short','float','device','negative','aliased','capacity','large64','packed_plan'):
            errors=[]
            for index in range(2):
                ops=Counter();functions=subject.functions(namespace(ops));args=fixture([4,3,4],ops,dtype='i64')
                mutate(args,kind)
                try:functions[index](*args,return_host_mappings=True)
                except RuntimeError as e:errors.append(str(e))
            self.assertEqual(len(errors),2,kind);self.assertEqual(errors[0],errors[1],kind)

    def test_draft_materialized_and_deferred_eager_keep_original_paths(self):
        for eager in(False,True):
            ops=Counter();before,after=subject.functions(namespace(ops));args=fixture([4,3],ops,draft=True)
            if eager:
                args[1].staged_eager[0]=types.SimpleNamespace(proposal_id=777)
                args[2][777]=dict(draft_kv_deferred_canonical=True)
            old=before(*args,return_host_mappings=True);counts=dict(ops);ops.clear()
            new=after(*args,return_host_mappings=True)
            self.assertTrue(subject.compare_results(old,new));self.assertEqual(dict(ops),counts)
            self.assertEqual(ops.get('original_deferred_validator',0),int(eager))

    def test_one_expression_exact_AST_inverse(self):
        original=subject.canonical_tree();changed=subject.optimized_tree(original);found=0
        for n in ast.walk(changed):
            if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='slot_values':
                self.assertIsInstance(n.value,ast.IfExp);n.value=n.value.orelse;found+=1
        self.assertEqual(found,1)
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(changed,include_attributes=False))

    def test_terminal_actual_accounting_and_negative(self):
        trace=dict(source_generation=5,normal_target_calls=[{},{}])
        state=dict(shadow=False,qualified=True,shadow_pairs=120,shadow_draft_pairs=0,shadow_target_pairs=120,
            by_generation={'5':dict(calls=3,direct_packed_target_calls=2,unchanged_other_calls=1)})
        self.assertTrue(subject.audit_snapshot(state,1,trace))
        state['by_generation']['5']['direct_packed_target_calls']=1
        with self.assertRaises(RuntimeError):subject.audit_snapshot(state,1,trace)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps(dict(status='passed'if result.wasSuccessful()else'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),Torch_imported='torch'in sys.modules,NPU_used=False)))
    raise SystemExit(0 if result.wasSuccessful()else 1)
