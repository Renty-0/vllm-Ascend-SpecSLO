"""Execute only the real receipt dict expression, never a model or device tool."""
import ast
import datetime
import json
from pathlib import Path

import guard400


def main():
    original = guard400.parent.execution_namespace()
    fixed = guard400.execution_namespace()
    source = (guard400.B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'execute_locked')
    row = next(n.value for n in function.body if isinstance(n, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'row' for target in n.targets))
    explicit = {keyword.arg for keyword in row.keywords if keyword.arg is not None}
    assert explicit & set(original['SCOPE']) == {'action', 'numeric_source_obligations_only'}
    assert not explicit & set(fixed['SCOPE'])
    namespace = dict(fixed, datetime=datetime, tag='HOST400-metadata', readiness={'HOST_only':True})
    expression = compile(ast.fix_missing_locations(ast.Expression(row)), __file__+'::actual-dict-only', 'eval')
    try:
        eval(expression, dict(namespace, SCOPE=original['SCOPE']))
    except TypeError as error:
        assert 'multiple values' in str(error)
    else:
        raise AssertionError('old duplicate metadata was not rejected')
    actual = eval(expression, namespace)
    assert actual['action'] == guard400.parent.ACTION399
    assert actual['numeric_source_obligations_only'] is True
    assert actual['supervision_metadata_repair400'] is True
    assert actual['active_request_limit'] == 64 and actual['effective_requests'] == 32
    assert actual['effective_candidates'] == 88
    assert actual['full463'] is actual['production_permission'] is False
    assert fixed['launch_argv'] is guard400.parent.launch_argv
    assert fixed['actual_audit'] is guard400.parent.actual_audit
    assert fixed['execute_locked'].__code__ == original['execute_locked'].__code__
    result = dict(status='PASS_HOST400_ACTUAL_RECEIPT_DUPLICATE_REPRODUCTION_AND_REPAIR',
                  NPU_used=False, models_constructed=0, device_tools_called=0,
                  exact_original_metadata_expression_executed=True,
                  duplicate_keywords_removed=['action','numeric_source_obligations_only'],
                  original399_supervision_body_and_numeric_runtime_retained=True,
                  original399_neural_sources_changed=False, original399_NPU_launched=False,
                  global_max_num_seqs=64, finite_N=32, finite_C=88,
                  Raw=None, Goodput=None, production_permission=False)
    (Path(__file__).resolve().parent/'HOST400_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','NPU_used','models_constructed')}))


if __name__ == '__main__':
    main()
