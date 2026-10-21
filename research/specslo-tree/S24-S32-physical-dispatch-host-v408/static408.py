"""Reversible projection of the original state; no installation on import."""
import hashlib
import inspect

import static_state255 as original


def sources():
    source = inspect.getsource(original.install)
    edits = (
        ('from projection255 import transform', 'from projection408 import transform'),
        ('    activation=branch_tail255.install(engine,native)\n',
         '    activation=branch_tail255.install(engine,native)\n'
         '    from padding408 import pad\n'
         '    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad\n'),
        ('extent=2560,mask_capacity=11 if state.enabled else 10)',
         'extent=2560,mask_capacity=11 if state.enabled else 10,real_N=count,copy_enabled=state.enabled)'),
        ('attention_segments=24,actual_seq_lengths_q=',
         'attention_segments=len(meta.sequence_lens),actual_seq_lengths_q='),
        ('fixed_q5_groups=24,fixed_extent=2560,',
         'narrow_q5_groups=24,wide_q4_groups=24,wide_q3_groups=8,maximum_extent=2560,'),
    )
    after = source
    for before, replacement in edits:
        if after.count(before) != 1:
            raise RuntimeError('HOST408 exact old state anchor ' + before)
        after = after.replace(before, replacement)
    inverse = after
    for before, replacement in reversed(edits):
        inverse = inverse.replace(replacement, before)
    if inverse != source:
        raise RuntimeError('HOST408 complete original state source inverse')
    return source, after


def construct():
    _, source = sources()
    namespace = dict(vars(original))
    exec(compile(source, __file__ + '::explicit-real-N-dual-projection',
                 'exec', dont_inherit=True), namespace)
    return namespace['install']


install = construct()


def descriptor():
    before, after = sources()
    return dict(original_sha256=hashlib.sha256(before.encode()).hexdigest(),
                projected_sha256=hashlib.sha256(after.encode()).hexdigest(),
                original_static_install_unmodified=True, complete_source_inverse=True,
                M=120, narrow_S=24, wide_S=32, real_owner_count_explicit=True,
                original_forward_and_eager_model_calls_preserved=True,
                uniform_KV_bucketing_not_implemented=True,
                NPU_used=False, LIVE_installed=False, production_permission=False,
                Raw=None, Goodput=None)
