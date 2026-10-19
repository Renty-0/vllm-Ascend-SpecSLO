"""Exact original numerical audit predicates, explicit532/M160 names."""
def actual_audit(directory):
    directory = Path(directory)
    reports = [read(directory / f'rank{r}-numeric.json') for r in range(4)]
    installed = [read(directory / f'rank{r}-installed-dispatch-before-model.json') for r in range(4)]
    require([v['rank'] for v in installed] == list(range(4)) and all((v['actual_code_reconstructed_equal'] is True and v['models_loaded'] is False for v in installed)) and (len({v['source_SHA'] for v in installed}) == 1), 'all4 actual replay-first bodies verified BEFORE models loaded')
    require(read(directory / 'worker-exit.json') == dict(closed=True, all_exited=True), 'all diagnostic roles exited')
    require(read(directory / 'completed.json') == dict(replies=[['calibrated', r, dict(numeric_cases=532, performance=False)] for r in range(4)], performance=False, adopted=False), 'complete original diagnostic protocol;532 actual model cases')
    require([v['rank'] for v in reports] == list(range(4)), 'all four diagnostic reports')
    require(all((v['model_dtype'] == 'torch.bfloat16' for v in reports)), 'all four actual model BF16 dtype retained')
    require(all((v['performance'] is False and v['whole_workload_AR_qualified'] is False and (v['adopted'] is False) for v in reports)), 'no production or performance qualification')
    require(all((len(v['math']['cases']) == 532 and v['math']['all_independent_prefixes_unchanged'] is True and all((c['hidden_bitwise_equal'] and c['KV_bitwise_equal'] and c['exact_greedy'] and c['graph_eager_KV_bitwise_equal'] for c in v['math']['cases'])) for v in reports[1:])), 'all532 actual model cases hidden/greedy/all128 KV bitwise equal, all96 prefixes unchanged')
    require(all((len(v['fresh160_raw_operator']['checks']) == 144 and v['fresh160_raw_operator']['raw16_bit_coverage'] == {'256': 65536, '384': 65536} for v in reports[1:])), 'all3 target ranks freshly prove160 raw-cache programs, both widths, independent strided inputs and slot dtypes')
    expected = read(HERE / 'DECLARED326_CASES.json')['cases']
    require(len(expected) == 532, 'predeclared532 feasible M160 node cases')
    for report in reports[1:]:
        require([r['case'] for r in report['math']['cases']] == expected, 'all408 original and all124 new feasible cases executed')
        from audit_static_calls326 import audit_calls
        require(report['dispatch_audit'] == audit_calls(expected, report['actual_static_graph_calls']), 'actual original capture/change/restore and real call reconstruction')
        require(len(report['paired_target_cost']['rows']) == 16, 'all declared synchronous target cost configurations')
        require(report['all_prefixes_still_exact_after_costs'] is True, 'all96 actual prefixes protected after timing as well')
        for row in report['paired_target_cost']['rows']:
            require(row['rounds'] == 20 and len(row['pairs']) == 20 and row['original_greedy_output_stable'], 'all20 rotated real model cost pairs and stable greedy outputs')
    return (dict(status='bounded_M160_all532_node_cases_completed', reports=[dict(rank=v['rank'], numeric_cases=len(v['math']['cases']), path=str(directory / f"rank{v['rank']}-numeric.json"), sha256=sha(directory / f"rank{v['rank']}-numeric.json")) for v in reports], formal_Goodput=False, adopted=False), 'diagnostic_only', dict(original_startup_inventory_not_run=True, actual_model_cases532=True, six_grid_families=True))
