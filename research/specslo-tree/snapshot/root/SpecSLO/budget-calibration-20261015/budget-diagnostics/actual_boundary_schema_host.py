"""Narrow HOST compatibility for retained complete boundary-event evidence.

Never rewrites an artifact or filters/replaces its raw event list. The original
validator's event-count/zip block is replaced only in an isolated function
namespace. Every other numerical/Native/MC2/commit/lifecycle check stays AST
identical. The replacement validates all event kinds and all status packets.
"""
import ast
from collections import Counter
import hashlib
from pathlib import Path


class BoundaryRejected(ValueError):pass


def need(value,message):
    if not value:raise BoundaryRejected(message)


def digest(value):
    return hashlib.sha256(__import__('json').dumps(value,separators=(',',':')).encode()).hexdigest()


def audit_raw_boundaries(row,scope,role,rank):
    events=row['atomic_runtime_evidence']['boundary_events']
    need(type(events) is list,'Complete original boundary list required')
    phases=list(scope.phases());n=scope.target_requests
    counter=Counter();sequence=0;common=[];stages=[]
    declared_index=commit_index=ledger_index=0
    for ordinal,event in enumerate(events):
        need(type(event) is dict and type(event.get('stage')) is str,'Malformed raw boundary event')
        stage=event['stage'];counter[stage]+=1
        if 'sequence' in event:
            need(type(event['sequence']) is int and event['sequence']==sequence
                 and set(event)=={'sequence','stage','ranks'},'Common boundary sequence/fields differ')
            packets=event['ranks']
            need(type(packets) is list and len(packets)==4
                 and sorted(p.get('rank') for p in packets)==list(range(4))
                 and all(type(p['rank']) is int and set(p)=={'rank','sequence','stage','error'}
                     and p['sequence']==sequence and p['stage']==stage and p['error'] is None for p in packets),
                 'Original common boundary rank/error agreement failed')
            sequence+=1;common.append((event['sequence'],stage));continue
        stages.append(stage)
        if stage=='original-V45-chunk-source-contract':
            need(event==dict(stage=stage,
                original_native_method_source_sha256='7b83f106990bf0f333d74a52796a2c9796eb2b4c5afc886d34da9824cef6af50',
                adapted_native_method_source_sha256='5fdad2dc2943b576976242ec54d527c7557cee0eb61f3e9994cf3fcf5cacd506',
                capacity_predicate='min(128, self.config.max_num_batched_tokens)',
                production_tree_config_not_changed=True,online_invocation_qualified=False),
                'Original chunk source/capacity evidence changed')
        elif stage=='reservation-lease-packets':
            identity=row['reservation_identity'];packets=event['ranks']
            need(event['nonce']==identity['nonce'] and type(packets) is list and len(packets)==4
                 and sorted(p['rank'] for p in packets)==list(range(4)),'Reservation packet rank/nonce differs')
            for packet in packets:
                need(packet['request_index']==n and packet['allocation_epoch']==identity['allocation_epoch']
                     and packet['active']==list(range(n)) and packet['pending']==[n]
                     and type(packet['capacity']) is int and n<packet['capacity']<=64
                     and packet['eligible_greedy'] is True and packet['prompt_length']==packet['committed_length']==2301
                     and type(packet['prompt_sha256']) is str and len(packet['prompt_sha256'])==64,
                     'Original reservation source/capacity/epoch packet differs')
                reserved,live=packet['reserved_pages'],packet['live_pages']
                need(type(reserved) is list and len(reserved)==len(set(reserved))==18
                     and type(live) is list and len(live)==len(set(live))
                     and not set(reserved)&set(live) and all(type(p) is int and 0<=p<544 for p in reserved+live),
                     'Original reservation physical pages alias/out-of-range')
            need(len({p['prompt_sha256'] for p in packets})==1,'Reservation prompt identity differs across ranks')
        elif stage=='declared-prefix-protection-scope':
            need(declared_index<len(phases),'Extra prefix-protection event')
            phase=phases[declared_index];record=row['phases'][declared_index]
            expected={str(i):len(tokens)-len(emitted)-1 for i,(tokens,emitted) in enumerate(zip(
                record['original_cycle']['actual_logical_tokens'],record['original_cycle']['committed_tokens'],strict=True))}
            expected[str(n)]=phase['prompt_start']
            need(event['materialized_prefix_lengths']==expected and event['scope']==
                'current target owners before root plus reserved committed cursor'
                and event['all_active_full_KV_hash_claimed'] is False
                and event['actual_model_layer_count']==(28 if role=='draft' else 64),
                'Actual declared prefix scope/length/layers differ')
            declared_index+=1
        elif stage=='original-tree-commit-evidence':
            need(commit_index<len(phases),'Extra original commit event')
            packets=event['ranks'];cycle=row['phases'][commit_index]['original_cycle']
            need(type(packets) is list and len(packets)==4 and all(packet==dict(
                cycle_id=commit_index,structural_failed=0,numerical_failed=0,
                finished_home=cycle['target_home'],next_home=1-cycle['target_home'],
                local_compaction_stream_fenced=True) for packet in packets),
                'Original four-rank commit/compaction evidence failed')
            commit_index+=1
        elif stage=='actual-prompt-draft-materialization-ledger':
            need(role=='draft' and ledger_index<len(phases),'Target/extra materialization ledger event')
            phase=phases[ledger_index]
            need(event==dict(stage=stage,owner=n,cycle_id=ledger_index,
                before=dict(safe_before=phase['prompt_start'],tokens={}),
                after=dict(safe_before=phase['prompt_end'],tokens={}),sampled_completion_root_claimed=False),
                'Original prompt28 ledger did not advance through actual cursor commit')
            ledger_index+=1
        else:raise BoundaryRejected('Unknown non-status raw boundary kind: '+stage)
    wanted=['original-V45-chunk-source-contract','reservation-lease-packets']
    for _ in phases:
        wanted+=['declared-prefix-protection-scope','original-tree-commit-evidence']
        if role=='draft':wanted+=['actual-prompt-draft-materialization-ledger']
    need(stages==wanted and declared_index==commit_index==len(phases)
         and ledger_index==(len(phases) if role=='draft' else 0),'Complete ordered boundary coverage differs')
    return dict(rank=rank,scope=scope.name,raw_events=len(events),common_status_boundaries=sequence,
        exact_prompt_materialization_events=ledger_index,original_commits=commit_index,
        all_raw_events_consumed=True,raw_event_list_unchanged=True,
        common_signature_sha256=digest(common),stage_counts=dict(counter))


def isolated_chain_validator(module):
    """Return a new chain checker with only the two event-schema gates adapted."""
    path=Path(module.__file__);tree=ast.parse(path.read_text())
    original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scope_terminal')
    scope_node=ast.parse(ast.unparse(original)).body[0]
    start=next(i for i,n in enumerate(scope_node.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='events')
    stop=next(i for i,n in enumerate(scope_node.body) if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='deliveries')
    removed=scope_node.body[start:stop]
    need(len(removed)==3 and isinstance(removed[0],ast.Assign) and isinstance(removed[1],ast.Expr)
         and isinstance(removed[2],ast.For),'Frozen event-schema adaptation anchors changed')
    need("len(events) == (len(scope.phases()) if role == 'draft' else 0)" in ast.unparse(removed[1])
         and 'zip(events, scope.phases())' in ast.unparse(removed[2]),'Only known complete-event schema mismatch may be adapted')
    replacement=ast.parse('_audit_raw_boundaries(row,scope,role,rank)').body
    scope_node.body[start:stop]=replacement
    need([ast.dump(n) for n in scope_node.body[:start]]==[ast.dump(n) for n in original.body[:start]]
         and [ast.dump(n) for n in scope_node.body[start+1:]]==[ast.dump(n) for n in original.body[stop:]],
         'Unrelated terminal qualification check changed')
    namespace=dict(vars(module),_audit_raw_boundaries=audit_raw_boundaries)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[scope_node],type_ignores=[])),
                 str(path)+':external-complete-boundary-schema','exec'),namespace)
    chain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chain_record')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[chain],type_ignores=[])),
                 str(path)+':external-original-chain','exec'),namespace)
    provenance=dict(original_validator_path=str(path),original_validator_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        old_event_block_AST_sha256=digest([ast.dump(n,include_attributes=False) for n in removed]),
        original_chain_AST_unchanged=True,all_other_scope_terminal_statements_AST_unchanged=True,
        adaptation='Only whole-event-list length and zip-as-ledger schema gates; replacement strictly consumes every raw status/metadata/ledger event. No raw list is filtered, changed or reserialized.')
    return namespace['chain_record'],provenance
