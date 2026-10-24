"""Actual qualifier routing and deepest terminal ABI; no model imports."""
from pathlib import Path
import ast, copy, datetime, hashlib, importlib.abc, inspect, json, subprocess, sys
from types import FunctionType, ModuleType, SimpleNamespace
P=Path(__file__).resolve().parent
B=P.parent
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','acl'):
            raise RuntimeError('HOST tests cannot import model/device packages')

def reject(fn):
    try:fn()
    except (RuntimeError,KeyError,ValueError,TypeError):return
    raise AssertionError('Invalid HOST evidence accepted')


def terminal_checks():
    import guard470
    # The complete deepest function, including its real slot/cache readers, is
    # exercised on retained466 data. Only the current audit module and its
    # completed comparison payload are HOST fixtures, declared explicitly.
    directory=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/credit-cost-empty48-current0127-root-v466-full463-ownAR-v1'
    completed=json.loads((directory/'completed.json').read_text())
    current=copy.deepcopy(completed['audit'])
    current.update(ROOT470_original466_all_checks_retained=True,ROOT470_actual_target_span_copy=[{'HOST_SYNTHETIC':True}]*3)
    completed['audit']=current
    original=guard470.constructed_audit();reads=original.__globals__['read'];calls=[]
    fake=ModuleType('audit470')
    def audit(where,windows):calls.append((Path(where),windows));return current
    fake.audit=audit;real=sys.modules['audit470'];sys.modules['audit470']=fake
    try:
        def read(path):return completed if Path(path)==directory/'completed.json' else reads(path)
        fn=FunctionType(original.__code__,dict(original.__globals__,read=read),original.__name__)
        packet=fn(directory)
        assert calls==[(directory,(64,))] and packet[0]is current
        assert guard470.check_packet(packet,completed)is packet
        negatives=0
        for mode in('dict','kind','AR','HOST_KV','wait','completed_mismatch','missing_current','missing_counts'):
            doc=copy.deepcopy(completed);bad=copy.deepcopy(packet)
            if mode=='dict':bad=bad[0]
            elif mode=='kind':bad=(bad[0],'incorrect',bad[2])
            elif mode=='AR':bad[2]['whole_workload_AR_qualified']=False
            elif mode=='HOST_KV':bad[2]['hidden_KV_recomputed_by_HOST']=True
            elif mode=='wait':doc['qualification_rpc_wait']['qualification_receive_timeout_seconds']=6000
            elif mode=='completed_mismatch':doc['audit']={}
            elif mode=='missing_current':bad[0].pop('ROOT470_original466_all_checks_retained');doc['audit']=bad[0]
            else:bad[0]['ROOT470_actual_target_span_copy']=[];doc['audit']=bad[0]
            reject(lambda:guard470.check_packet(bad,doc));negatives+=1
        wrong=copy.deepcopy(completed);wrong['audit']={}
        fail=FunctionType(original.__code__,dict(original.__globals__,read=lambda path:wrong if Path(path)==directory/'completed.json' else reads(path)),original.__name__)
        reject(lambda:fail(directory));negatives+=1
    finally:sys.modules['audit470']=real
    return dict(actual_deepest_current_import_called=True,original_three_tuple_retained=True,
                real466_slot_cache_AR_reports_read=True,current470_payload_HOST_fixture_only=True,
                negative_cases=negatives)


def main():
    sys.meta_path.insert(0, NoModel())
    shell = (P/'startup470.sh').read_text()
    line = next(s for s in shell.splitlines() if s.startswith('export PYTHONPATH='))
    sys.path[:0] = line[len('export PYTHONPATH='):-len('${PYTHONPATH:-}')].split(':')
    import runtime470, extension470, extension460, extension465, extension466, extension432, guard470, install470, audit470
    # Construct the genuine final nested qualifier, not a standalone mock of
    # the outer wrapper. Inspect source to prove installation precedes227 and
    # leaves the original681/693/lifecycle/own-AR path intact.
    leaf = extension470.projected_qualifier()
    assert 'install470' in leaf.__code__.co_names
    assert 'bucket460' not in leaf.__code__.co_names
    assert 'oracle257' in leaf.__code__.co_names
    assert 'base_component465' in leaf.__code__.co_names
    assert 'base_component433' not in leaf.__code__.co_names
    outer = extension470.qualifier()
    assert 'qualify460' in outer.__code__.co_names
    assert outer.__globals__['projected_qualifier'] is extension470.projected_qualifier
    before = extension465.projected_qualifier()
    assert leaf.__code__.co_code == before.__code__.co_code
    assert tuple('bucket460' if x == 'install470' else x for x in leaf.__code__.co_names) == before.__code__.co_names
    install = install470.prepared()
    assert 'FunctionType' in install.__code__.co_names and 'target_geometry' not in install.__code__.co_names
    assert 'layout' in install.__code__.co_names
    for name in ('prepare_install','install_entry','after_install','cache_snapshot','bucket_snapshot',
                 'metadata_snapshot','refill_snapshot','preflight_snapshot','postfull_AR'):
        assert getattr(extension470,name) is getattr(extension466,name)
    original = runtime470.original.source()
    inverse = runtime470.source()
    assert 'from trace466 import run_windows,warm_to_fixedpoint' in inverse
    for old, new in reversed(runtime470.ROUTES):
        inverse = inverse.replace(new, old)
    assert inverse == original
    current = guard470.execution_namespace()
    assert current['actual_audit'] is guard470.actual_audit
    assert current['execute_locked'].__globals__ is current
    assert current['SCOPE']['global_active_limit'] == 64 and current['SCOPE']['target_copy_programs'] == 64
    assert current['SCOPE']['original_native_tree_update_method_unchanged'] is True
    guard470.EXPECTED_FREEZE = '0'*64
    assert guard470.launch_argv('HOST_NOT_LAUNCHED').count(str(P/'launch470.sh')) == 1
    terminal = terminal_checks()
    from actor_check import check as check_actor
    actor = check_actor()
    for path in P.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    for path in P.glob('*.sh'):
        subprocess.run(['bash','-n',str(path)], check=True)
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST470_ORIGINAL_QUALIFIERS_AND_DEEPEST_TERMINAL_ABI',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        terminal_interfaces=terminal, actual_current_card_actor=actor, unchanged_original_227_681_693_and_AR=True,
        original_LIVE_and_clocks_unchanged=True, fresh_copy_install_before_node_math=True,
        NPU_used=False, NN_permission=False, global_active_limit=64, physical_cards=[0,1,2,7],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir()
                       if p.is_file() and p.suffix in ('.py','.sh')})
    (P/'HOST470_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__ == '__main__':
    main()
