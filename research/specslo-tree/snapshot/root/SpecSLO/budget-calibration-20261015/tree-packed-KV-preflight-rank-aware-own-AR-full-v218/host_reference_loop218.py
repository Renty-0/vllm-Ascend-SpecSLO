"""Synthetic HOST control-flow test of all four actual-ABI reference roles.

Token values come from recorded candidates solely as a test fixture. There is
NO neural model, Torch library, device calculation, or numerical qualification.
The test exercises the real reference runner and independent history auditor.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import reference218 as runner
import audit_reference218 as auditor

P = Path(__file__).resolve().parent
B = P.parent
ORIGINAL = B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/cost-aware-N9to16-root-v216-full463-AR-v1/windows/window32/full463'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise AssertionError(message)


def main():
    need('torch' not in sys.modules and 'torch_npu' not in sys.modules, 'no real backend already imported')
    root = P/'HOST-loop-attempts'/str(time.time_ns())
    folder = root/'synthetic-fixture'
    traces = folder/'windows/window32/full463'
    traces.mkdir(parents=True, exist_ok=False)
    source_inputs = {}
    for rank in range(4):
        path = ORIGINAL/f'rank{rank}-full463-actual-trace.json.gz'
        source_inputs[str(path)] = sha(path)
        (traces/path.name).write_bytes(path.read_bytes())
    lead = auditor.read(traces/'rank1-full463-actual-trace.json.gz')
    groups = [r['sequence_ids'] for r in lead['standalone_prefill_calls']]
    candidate = {r['request_id']:r['completion_token_ids'] for r in lead['generate_results']}
    by_index = [candidate[r['request_id']] for r in lead['requests']]
    counters = tuple(lead['native']['before'])
    need(len(counters) == 21, 'original21 retained counter names')
    fake_torch = types.ModuleType('torch')
    fake_dist = types.ModuleType('torch.distributed')
    fake_torch.distributed = fake_dist
    fake_scope = types.ModuleType('scope_v146')
    fake_scope.NATIVE_COUNTERS = counters
    fake_scope.FAILURES = counters
    real_step = runner.target_step
    active_engine = None
    calls = []

    class Engine:
        def __init__(self, rank):
            self.rank = rank
            self.is_draft = rank == 0
            self.topology = types.SimpleNamespace(target_leader_rank=1, draft_leader_rank=0)
            self.graph_runner = types.SimpleNamespace(graph_cache_sealed=True)
            self.groups = types.SimpleNamespace(verification_coordination_group=object())
            self.cache_allocation = self.cache_block_tables = None
            self.ordinal = 0

        def graph_metrics(self):
            return dict.fromkeys(counters, 0)

        def _allocate_cache(self, source, **kwargs):
            self.owners = groups[self.ordinal]
            need(source == [lead['tokenized_inputs'][i] for i in self.owners], 'original own prompt/cohort source')
            self.prompts = source
            self.expected = [by_index[i] for i in self.owners]
            tables = [[-1]*32 for _ in range(527)]
            cursor = 0
            for owner, prompt in enumerate(source):
                count = (len(prompt)+256+9+127)//128
                tables[owner][:count] = list(range(cursor, cursor+count))
                cursor += count
            for owner in range(463, 479):
                tables[owner][0] = cursor
                cursor += 1
            need(cursor <= 544, 'synthetic page fixture within original envelope')
            self.cache_allocation = types.SimpleNamespace(block_tables=tables)
            self.cache_block_tables = tables

        def _prefill_and_sample_target_batch(self, source, states, ids):
            need(ids == list(range(len(source))), 'original local cohort owner order')
            return [v[0] for v in self.expected]

        def _vote_spec_rhythm_prefill_finiteness(self):
            return None

        def _release_cache(self):
            self.cache_allocation = self.cache_block_tables = None
            self.ordinal += 1

    def step(engine, data):
        heads = [0]*80
        for row, owner in enumerate(data['real_owners']):
            offset = data['positions'][5*row]-len(engine.prompts[owner])
            need(data['tokens'][5*row] == engine.expected[owner][offset], 'real runner consumes own preceding fixture token')
            heads[5*row] = engine.expected[owner][offset+1]
        ids = data['ids']
        positions = data['positions']
        tables = engine.cache_allocation.block_tables
        physical_owners = ids[::5]
        return dict(heads80=heads, real_tokens=[heads[i] for i in data['real_head_rows']],
            metadata=dict(positions=positions, slots=[tables[i][p//128]*128+p%128 for i,p in zip(ids,positions,strict=True)],
                contexts=[p+1 for p in positions], block_tables=[tables[i] for i in physical_owners],
                actual_seq_lengths_q=data['actual_seq_lengths_q'], sequence_lens=data['sequence_lens']))

    def gather(rows, packet, **kwargs):
        digest = runner.digest(active_engine.expected)
        rows[:] = [dict(rank=rank, reference_sha256=None if rank==0 else digest) for rank in range(4)]
        need(rows[active_engine.rank] == packet, 'real runner emits truthful own-role packet')

    def stage(label, callback):
        calls.append((active_engine.rank, label))
        return callback()

    fake_dist.all_gather_object = gather
    sys.modules.update({'torch':fake_torch, 'torch.distributed':fake_dist, 'scope_v146':fake_scope})
    runner.target_step = step
    try:
        for rank in range(4):
            active_engine = Engine(rank)
            path = traces/f'rank{rank}-full463-actual-trace.json.gz'
            windows = dict(traces=[dict(original_trace=dict(role='full463', scope_count=463,
                artifact=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))))])
            native = types.SimpleNamespace(PearlPipelineState=lambda *args, **kwargs:types.SimpleNamespace())
            result = runner.run(active_engine, native, stage, windows, folder)
            need(result['completed'] and result['reference_trajectories_generated'] == (0 if rank==0 else 463), 'complete actual runner control flow')
        positive = auditor.audit(folder)
        need(positive['passed_retained_history_metadata_audit'] and positive['same_backend_exact_output_qualified'], 'synthetic fixture auditor roundtrip only')
        # These are transcript/role checks. No device qualification is inferred.
        path = folder/'postfull-independent-FIA-AR/rank2-summary.json'
        saved = path.read_bytes()
        changed = json.loads(saved)
        changed['candidate_result_artifact']['sha256'] = '0'*64
        path.write_text(json.dumps(changed))
        try:
            auditor.audit(folder)
        except RuntimeError:
            negative = True
        else:
            raise AssertionError('wrong candidate artifact accepted')
        path.write_bytes(saved)
    finally:
        runner.target_step = real_step
        for name in ('torch.distributed', 'torch', 'scope_v146'):
            sys.modules.pop(name, None)
    need(all(sum(r==rank and label=='AR218/retained-output-leader-comparison-only' for r,label in calls)==1 for rank in range(4)), 'one actual synchronized candidate load per role')
    output = dict(schema='ROOT218.synthetic-HOST-reference-control-flow.v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='passed', actual_reference_runner_roles=4, simulated_requests_per_target=463,
        simulated_target_steps_per_rank=positive['by_target_rank'][0]['target_steps'],
        wrong_comparison_artifact_rejected=negative, original_role_None_traces_preserved=True,
        source_sha256={str(path):sha(path) for path in (Path(__file__),P/'reference218.py',P/'audit_reference218.py',P/'candidate_comparison218.py')},
        original_actual_input_sha256=source_inputs, synthetic_fixture_only=True, real_Torch_imported=False,
        NPU_used=False, model_constructed=False, neural_output_qualified=False, Goodput=None, adopted=False)
    (root/'RESULT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(result=str(root/'RESULT.json'), **{k:v for k,v in output.items() if k not in ('source_sha256','original_actual_input_sha256')})))


if __name__ == '__main__':
    main()
