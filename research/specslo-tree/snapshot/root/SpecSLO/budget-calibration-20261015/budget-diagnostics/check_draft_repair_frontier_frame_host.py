"""Six stdlib checks for an uninstalled symbolic layout, never a model test."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

D = Path(__file__).resolve().parent
P = D.parent / 'tree-standalone-prefill-control-v143'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sys.version_info[:2] == (3, 11)
    assert not any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') for name in sys.modules)
    freeze = json.loads((P / 'PREPARED_SOURCE_HASHES_V132.json').read_text())
    assert freeze['external_files'][str(NATIVE)] == sha(NATIVE)
    text = NATIVE.read_text()
    wanted = ('_linear_draft_graph_buckets', '_next_linear_draft_graph_bucket')
    nodes = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(nodes) == 2 and tuple(node.name for node in nodes) == wanted
    native_bucket_source = '\n\n'.join(ast.get_source_segment(text, node) for node in nodes)
    namespace = {}
    exec(compile(native_bucket_source, str(NATIVE) + ':two-pure-HOST-bucket-functions', 'exec'), namespace)
    bucket = namespace['_next_linear_draft_graph_bucket']
    spec = importlib.util.spec_from_file_location('_fusion_layout_host', D / 'draft_repair_frontier_frame_host.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = lambda r, e: module.plan_single_query_frame(r, e, next_bucket=bucket)
    reports = []
    started = time.perf_counter()

    def check(name, callback):
        callback()
        reports.append(dict(name=name, passed=True))

    def head_and_dummy():
        value = plan([(400, 140, [13], 14)], [(i, 130 + i, 700 + i) for i in range(15)])
        assert value['original_frontier_physical_M'] == 16 and value['physical_M'] == 20
        assert value['frontier_original_head_rows'] == list(range(16))
        assert value['rows'][15] == dict(role='frontier_dummy', owner=None, position=0, token=0,
                                        context_length=1, slot_mapping=-1, block_table_anchor_owner=0)
        assert value['rows'][16]['owner'] == 400 and value['normal_repair_rows_in_union'] == [16]
        assert value['old_two_call_physical_queries'] == 17 < value['new_one_call_physical_queries']
    check('preserve_frontier_head_M_and_readonly_dummy_even_when_union_padding_increases', head_and_dummy)

    def endpoint():
        value = plan([(i, 128, [i], i + 1) for i in range(32)], [(i, 131, i + 100) for i in range(32, 64)])
        assert value['physical_M'] == 64 and len(value['rows']) == 64
        assert value['normal_repair_rows_in_union'] == list(range(32, 64))
        assert value['original_query_count'] == value['new_query_count'] == 64
    check('normal32_future32_union64_endpoint_without_new_concurrency_claim', endpoint)

    def owner_ids():
        value = plan([(462, 4095, [1], 2)], [(315, 100, 3)])
        assert [row['owner'] for row in value['rows']] == [315, 462]
        assert value['rows'][1]['context_length'] == 4096
    check('logical_owner_ID_is_not_a_physical_row_index', owner_ids)

    def fallbacks():
        assert plan([(3, 128, [1, 2], 3)], [(5, 135, 7)])['mode'] == 'original'
        assert plan([], [(5, 135, 7)])['mode'] == 'original'
        assert plan([(3, 128, [1], 2)], [])['mode'] == 'original'
    check('multiquery_and_empty_group_keep_original_execution_path', fallbacks)

    def rejects():
        bad = [([(3, 128, [1], 2)], [(3, 135, 7)]),
               ([(i, 128, [1], 2) for i in range(33)], [(40, 135, 7)]),
               ([(3, 4095, [1, 2], 2)], [(5, 135, 7)]),
               ([(3, 128, [True], 2)], [(5, 135, 7)])]
        for r, e in bad:
            try:
                plan(r, e)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid request accepted')
    check('reject_duplicate_owner_row_overflow_context_overflow_and_noninteger_token', rejects)

    def conservation():
        r = [(5, 130, [20], 21), (9, 131, [22], 23)]
        e = [(15, 133, 24), (19, 134, 25)]
        before = json.dumps([r, e])
        value = plan(r, e)
        actual = {(row['owner'], row['position'], row['token']) for row in value['rows'] if row['owner'] is not None}
        assert actual == {(5, 130, 20), (9, 131, 22), (15, 133, 24), (19, 134, 25)}
        assert json.dumps([r, e]) == before
        assert value['physical_slots_must_be_checked_on_actual_metadata']
        assert value['production_installed'] is False
    check('exact_real_token_position_conservation_without_mutating_inputs_or_ledger', conservation)

    report = dict(schema='HOST.draft-repair-frontier-symbolic-layout-prototype.v1', status='passed', tests_run=len(reports),
                  checks=reports, elapsed_seconds=time.perf_counter() - started, actual_HOST_argv=sys.orig_argv,
                  source_sha256={str(Path(__file__)): sha(Path(__file__)),
                                 str(D / 'draft_repair_frontier_frame_host.py'): sha(D / 'draft_repair_frontier_frame_host.py'),
                                 str(NATIVE): sha(NATIVE)},
                  actual_two_Native_bucket_functions_sha256=hashlib.sha256(native_bucket_source.encode()).hexdigest(),
                  Torch_imported=False, NPU_initialized=False, model_or_device_query=False,
                  production_installed=False, numerical_or_latency_qualified=False,
                  limitations=['Symbolic layout only; no actual KV tables, slot alias check, tensor, model, head, numerical gate or timing.'])
    output = D / 'DRAFT_REPAIR_FRONTIER_FRAME_HOST_PROTOTYPE_CHECK.json'
    with output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps(dict(status='passed', tests_run=len(reports), output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
