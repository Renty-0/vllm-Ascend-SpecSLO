"""Reconstruct full workload and ownAR, plus all fresh693 strict branch/prefix proofs."""
import hashlib
import json
from pathlib import Path
import audit257 as original
from projection383 import project_function
from geometry383 import valid_record


def original_raw_audit(directory,selected_windows):
    import ast,copy
    from window257 import windows
    old=original.old
    source=Path(old.__file__).read_text()
    changes=((' from mapper209 import expected_geometry',' from geometry383 import envelope as expected_geometry'),
             ('expected_geometry(n,q,f)',"expected_geometry(n,q,f,c['budgets'])"))
    projected=source
    for before,after in changes:
        if projected.count(before)!=1:raise RuntimeError('ROOT383 one original raw audit anchor '+before)
        projected=projected.replace(before,after)
    inverse=projected
    for before,after in reversed(changes):inverse=inverse.replace(after,before)
    if inverse!=source:raise RuntimeError('ROOT383 complete original raw audit source inverse')
    nodes=[copy.deepcopy(n)for n in ast.parse(projected).body if isinstance(n,ast.FunctionDef)and n.name=='original_audit']
    if len(nodes)!=1:raise RuntimeError('ROOT383 exact original raw trace auditor')
    namespace=dict(vars(old),windows=windows)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::original-raw-audit-with-real-budgets','exec'),namespace)
    return namespace['original_audit'](directory,selected_windows)


def need(value,message):
    if value is not True:raise RuntimeError('ROOT383 audit: '+message)


def projected_original_audit():
    return project_function(original.audit,(
        ('from projection255 import plan as physical_plan','from physical377 import plan as physical_plan'),
        ("physical_plan(ends,call['real_query_count'])","physical_plan(ends,call['real_query_count'],call['physical_M'])"),
        ("row['M'] == call['physical_M'] == 120","row['M'] == call['physical_M'] and row['M'] in (80,120)"),
        ("row['attention_segments']==24 and row['primary_side_groups']==row['N']+branches<=24",
         "row['attention_segments']==row['M']//5 and row['primary_side_groups']==row['N']+branches<=row['M']//5"),
        ("and row['fixed_KV_extent']==[2560]*24",'and _ROOT383_valid_record(row)'),
        ("row['actual_q_partition']==list(range(5,121,5))","row['actual_q_partition']==list(range(5,row['M']+1,5))"),
    ),{'_ROOT383_valid_record':valid_record,'original_with_new_geometry':original_raw_audit})


def audit(directory,selected_windows):
    result=projected_original_audit()(directory,selected_windows)
    from oracle377 import cases
    from dispatch377 import audit_calls
    all_heads=[]
    for rank in range(4):
        report=json.loads((Path(directory)/f'rank{rank}-local-diagnostic.json').read_text())
        extra=report['ROOT383_uniform_KV_math'];proof=extra['proof']
        need(proof['cases']==693 and proof['same_model_instance'] is True
            and proof['original227_FULL_math_preserved'] is True and proof['native_flags_assigned'] is False
            and len(proof['banks'])==3,'all current693 independent cases and original227')
        heads=[]
        for descriptor in proof['banks']:
            path=Path(descriptor['path']).resolve()
            need(path.is_relative_to(Path(directory).resolve())and descriptor['cases']==231
                and hashlib.sha256(path.read_bytes()).hexdigest()==descriptor['sha256'],'exact retained current bank proof')
            bank=json.loads(path.read_text());need(len(bank['math']['cases'])==231,'all231 per bank')
            if rank==0:
                need(bank['bank'] is None and bank['dispatch'] is None and all(x is None for x in bank['math']['cases']),'draft has no target math claim')
                continue
            b=bank['bank']
            need(b['independent_neural_prefills']==72 and b['all_triplet_roots_equal'] is True
                and b['all_triplet_prefix_KV_equal'] is True and bank['math']['all_independent_prefixes_unchanged'] is True,'all independent neural prefixes protected')
            audit_calls(cases(),bank['actual_graph_calls'],[[0]*n for n in bank['prefix_lengths']])
            for case,seen in zip(cases(),bank['math']['cases'],strict=True):
                need(seen['case']==case and seen['exact_greedy'] is True
                    and seen['candidate_tokens']==seen['reference_tokens']
                    and seen['hidden_bitwise_equal'] is True and seen['KV_bitwise_equal'] is True
                    and seen['all128_KV_original_matcher_passed'] is True and seen['graph_eager_KV_bitwise_equal'] is True
                    and seen['candidate_KV_copied'] is False,'unchanged Native matcher and strict all-value checks')
                heads.append(seen['candidate_tokens'])
        if rank:
            h=extra['hybrid'];need(h['applied'] is True and h['precision']=='BF16'and h['quantized'] is False
                and h['actual']['column_modules']==128 and h['actual']['head_modules']==1
                and h['Native_qualification_flags_assigned'] is False,'actual unchanged-precision hybrid route')
            all_heads.append(heads)
    need(all_heads[0]==all_heads[1]==all_heads[2],'all3 target ranks agree on693 cases')
    return dict(result,ROOT383_fresh_target_cases=693,ROOT383_original227_and_full463_AR_preserved=True,
        ROOT383_original_budget_allocator_and_optimized_draft_preserved=True)
