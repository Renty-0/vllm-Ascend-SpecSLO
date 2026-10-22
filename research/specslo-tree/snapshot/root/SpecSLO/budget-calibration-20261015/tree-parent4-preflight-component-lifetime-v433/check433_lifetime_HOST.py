"""Execute the unchanged218 installer and preflight wrapper using HOST fixtures."""
import ast,copy,importlib.util,json,types
from collections import Counter
from pathlib import Path


def make_subject(label):
    P=Path(__file__).resolve().parent.parent/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
    spec=importlib.util.spec_from_file_location('_HOST433_slot_'+label,P/'slot_preflight218.py')
    subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
    import host_slot_preflight218 as fixtures
    ops=Counter();ns=fixtures.namespace(ops)
    cls=next(n for n in ast.parse(subject.NATIVE.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==subject.NAME))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(subject.NATIVE),'exec'),ns)
    native=types.ModuleType('_HOST433_native_'+label);native.__dict__.update(ns)
    native.NativePearlEngine=type('NativePearlEngine',(),{subject.NAME:ns[subject.NAME]})
    spec=importlib.util.spec_from_file_location('_HOST433_canonical_'+label,subject.CANONICAL)
    canonical=importlib.util.module_from_spec(spec);spec.loader.exec_module(canonical)
    canonical.install(native)
    subject.install(native)
    return subject,getattr(native.NativePearlEngine,subject.NAME),fixtures,ops


def main():
    old,preflight,fixtures,ops=make_subject('old_premature_close')
    args=fixtures.fixture([4]*24,ops)
    preflight(*args,return_host_mappings=True)
    old.end_shadow(args[0])
    try:preflight(*args,return_host_mappings=True)
    except RuntimeError as error:
        assert 'actual current generation for execution accounting'in str(error)
        old_error=str(error)
    else:raise AssertionError('the actual old430 failure must reproduce')
    records=[]
    for draft in(False,True):
        subject,preflight,fixtures,ops=make_subject('draft'if draft else'target')
        pairs=0
        for n in(1,2,8,16,24):
            for dtype in('i32','i64'):
                args=fixtures.fixture([4]*n,ops,dtype=dtype,draft=draft)
                assert not hasattr(args[0],'_dynamic_tree_generation')
                preflight(*args,return_host_mappings=True);pairs+=1
        state=subject.snapshot()
        assert state['shadow']is True and state['qualified']is False and state['shadow_pairs']==pairs
        bad=fixtures.fixture([4]*24,ops,dtype='i64',draft=False)
        bad[-1]['cache_slot_mapping'].values[0]=-1
        try:preflight(*bad,return_host_mappings=True)
        except RuntimeError:pass
        else:raise AssertionError('the unchanged original negative slot guard must still reject')
        # Call exactly the original close API once, after all component calls.
        closed=subject.end_shadow(args[0])
        assert closed['shadow']is False and closed['qualified']is True
        try:preflight(*args,return_host_mappings=True)
        except RuntimeError as error:
            assert 'actual current generation for execution accounting'in str(error)
        else:raise AssertionError('production without a real generation must still reject')
        # Explicit HOST fixture field exercises the unchanged accounting path;
        # production433 never assigns an engine generation.
        args[0]._dynamic_tree_generation=7
        preflight(*args,return_host_mappings=True)
        final=subject.snapshot()
        assert subject.audit_snapshot(final,0 if draft else 1,
            dict(source_generation=7,normal_target_calls=[{}]))is True
        records.append(dict(role='draft'if draft else'target',actual_component_shadow_pairs=pairs,
            original_bad_mapping_rejected=True,production_missing_generation_rejected=True,
            original_generation7_accounting_audit=True,original_close_called_once=True))
    result=dict(status='PASS_HOST433_UNCHANGED_ORIGINAL218_COMPONENT_LIFETIME_AND_PRODUCTION_GUARDS',
        old430_missing_generation_reproduced=old_error,cases=records,
        original_canonical_installer_and_original218_install_end_shadow_and_audit_used=True,
        explicit_original_HOST_integer_tensor_fixtures=True,NPU_used=False,
        actual_Native_models_constructed=0,NN_qualification=False)
    (Path(__file__).resolve().parent/'HOST433_LIFETIME_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
