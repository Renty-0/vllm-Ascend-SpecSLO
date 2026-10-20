"""Extra S24 resource bound in the LIVE whole-proposal packer only.

The original controller class is never modified. This projects its actual
build_plan body with one additional additive resource and preserves ordering,
wait ages, home selection, complete tickets, and reservation validation.
"""
import __future__
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

from vllm_ascend.spec_decode.pearl import spec_rhythm as original

SEGMENT_LIMIT = 24


def ready_costs(controller, payloads, active):
    costs = {}
    for owner in active:
        ticket = controller.ready.get(owner)
        if ticket is None:
            continue
        payload = payloads[ticket.proposal_id]
        plan = payload['plan']
        budget = plan.candidate_budget
        if type(budget) is not int or not 1 <= budget <= 5 or ticket.gamma != budget:
            raise ValueError('ROOT398 exact ready ticket and qualified B1..5 plan required')
        if plan.parent_indices.device.type != 'cpu':
            raise ValueError('ROOT398 segment accounting must not read a device tensor')
        parents = tuple(plan.parent_indices.tolist())
        if budget <= 4:
            if parents != (-1, 0, 1, 2)[:budget]:
                raise ValueError('ROOT398 qualified primary spine required')
            costs[owner] = 1
        else:
            if len(parents) != 5 or parents[:4] != (-1, 0, 1, 2) or parents[4] not in (-1, 0, 1, 2):
                raise ValueError('ROOT398 qualified one-side B5 requires exactly two segments')
            costs[owner] = 2
    return costs


def projected_source():
    source = textwrap.dedent(inspect.getsource(original.SpecRhythmPipelineController.build_plan))
    edits = (
        ('    reserved_target_home_batch_id: int | None = None,\n',
         '    reserved_target_home_batch_id: int | None = None,\n'
         '    max_target_segments: int | None = None,\n'
         '    ready_segment_counts: Mapping[int, int] | None = None,\n', 1),
        ('    if (reserved_target_request_indices is None) != (reserved_target_home_batch_id is None):\n',
         '    if (max_target_segments is None) != (ready_segment_counts is None):\n'
         '        raise ValueError("ROOT398 explicit segment limit and counts must be supplied together")\n'
         '    if max_target_segments is not None and (type(max_target_segments) is not int or max_target_segments <= 0):\n'
         '        raise ValueError("ROOT398 positive integer segment limit required")\n'
         '    if (reserved_target_request_indices is None) != (reserved_target_home_batch_id is None):\n', 1),
        ('    if any(value <= 0 for value in candidate_counts.values()):\n',
         '    segment_counts = ({index: ready_segment_counts[index] for index in target_candidates}\n'
         '                      if ready_segment_counts is not None else {})\n'
         '    if any(type(value) is not int or not 1 <= value <= 2 or value > max_target_segments\n'
         '           for value in segment_counts.values()):\n'
         '        raise ValueError("ROOT398 complete ready proposal must fit the segment limit")\n'
         '    if any(value <= 0 for value in candidate_counts.values()):\n', 1),
        ('        target = reserved_target\n',
         '        if max_target_segments is not None and sum(segment_counts[index] for index in reserved_target) > max_target_segments:\n'
         '            raise ValueError("ROOT398 target replay reservation exceeds the segment limit")\n'
         '        target = reserved_target\n', 1),
        ('        or (verification_budget is not None and sum(candidate_counts.values()) > verification_budget)\n',
         '        or (verification_budget is not None and sum(candidate_counts.values()) > verification_budget)\n'
         '        or (max_target_segments is not None and sum(segment_counts.values()) > max_target_segments)\n', 1),
        ('        selected_tokens = 0\n',
         '        selected_tokens = 0\n'
         '        selected_segments = 0\n', 1),
        ('            selected.append(index)\n',
         '            if max_target_segments is not None and selected_segments + segment_counts[index] > max_target_segments:\n'
         '                continue\n'
         '            selected.append(index)\n'
         '            if max_target_segments is not None:\n'
         '                selected_segments += segment_counts[index]\n', 1),
    )
    result = source
    for before, after, count in edits:
        if result.count(before) != count:
            raise RuntimeError('ROOT398 exact original packer anchor ' + before)
        result = result.replace(before, after)
    inverse = result
    for before, after, _ in reversed(edits):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT398 entire original build_plan inverse')
    return source, result


def construct():
    source, projected = projected_source()
    namespace = dict(vars(original))
    flags = original.SpecRhythmPipelineController.build_plan.__code__.co_flags & __future__.annotations.compiler_flag
    exec(compile(ast.parse(projected), str(Path(__file__).resolve()) + '::segment-bounded-original-build-plan',
                 'exec', flags=flags, dont_inherit=True), namespace)
    function = namespace['build_plan']
    function.__module__ = __name__
    return function


build_plan = construct()


def descriptor():
    before, after = projected_source()
    return dict(original_source_sha256=hashlib.sha256(before.encode()).hexdigest(),
                projected_source_sha256=hashlib.sha256(after.encode()).hexdigest(),
                original_controller_method_unmodified=True, additive_attention_segment_limit=SEGMENT_LIMIT,
                whole_proposals_only=True, original_priority_and_age_code_preserved=True,
                original_candidate_budgets_unchanged=True, neural_code_changed=False,
                global_active_limit=64, NPU_qualified=False, Raw=None, Goodput=None)
