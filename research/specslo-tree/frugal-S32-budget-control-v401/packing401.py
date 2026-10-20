"""HOST candidate: pack whole linear tickets with an independent B3 quota."""
import __future__
import ast
import hashlib
from pathlib import Path

import packing398


def projected_source():
    original, source398 = packing398.projected_source()
    edits = (
        ('not 1 <= value <= 2', 'not 0 <= value <= 1'),
        ('max_target_segments', 'max_large_groups401'),
        ('ready_segment_counts', 'ready_large_groups401'),
        ('segment_counts', 'large_counts401'),
        ('selected_segments', 'selected_large_groups401'),
        ('ROOT398', 'ROOT401'),
        ('segment limit', 'large-group limit'),
    )
    result = source398
    for before, after in edits:
        if before not in result:
            raise RuntimeError('exact qualified398 control projection anchor ' + before)
        result = result.replace(before, after)
    inverse = result
    for before, after in reversed(edits):
        inverse = inverse.replace(after, before)
    if inverse != source398:
        raise RuntimeError('entire398 controller projection inverse')
    return original, source398, result


def construct():
    _, _, source = projected_source()
    original = packing398.original
    namespace = dict(vars(original))
    flags = original.SpecRhythmPipelineController.build_plan.__code__.co_flags & __future__.annotations.compiler_flag
    exec(compile(ast.parse(source), str(Path(__file__).resolve()) + '::whole-linear-B3-quota',
                 'exec', flags=flags, dont_inherit=True), namespace)
    return namespace['build_plan']


build_plan = construct()


def descriptor():
    before, intermediate, after = projected_source()
    return dict(original_source_sha256=hashlib.sha256(before.encode()).hexdigest(),
                parent398_projected_sha256=hashlib.sha256(intermediate.encode()).hexdigest(),
                projected_source_sha256=hashlib.sha256(after.encode()).hexdigest(),
                resource_is_count_of_B3_owners=True, maximum_B3_owners=24,
                physical_attention_groups=32, C=88, M=120,
                original_controller_class_modified=False, tickets_trimmed=False,
                NPU_qualified=False, production_permission=False)
