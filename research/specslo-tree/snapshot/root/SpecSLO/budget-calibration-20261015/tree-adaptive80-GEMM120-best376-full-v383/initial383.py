"""Install one copy state before original227 FULL120 cases; enable80 only later."""
def install(engine,native):
    from install377 import install as physical
    from bucket377 import bind
    from repair384 import install as parser_repair
    from hybrid382 import install as hybrid
    parser_repair()
    activation=physical(engine,native)
    bucket=bind(engine)
    engine._branch_tail_state.adaptive377_enabled=False
    if not engine.is_draft:engine._branch_tail_state.uniform_KV_bucket377_enabled=False
    math=hybrid(engine)
    return dict(physical=activation,KV=bucket,hybrid=math,
        original227_first_use_FULL120_KV2560=True,exactly_one_original_branch_copy_state=True,
        no_neural_qualification_inherited=True)
