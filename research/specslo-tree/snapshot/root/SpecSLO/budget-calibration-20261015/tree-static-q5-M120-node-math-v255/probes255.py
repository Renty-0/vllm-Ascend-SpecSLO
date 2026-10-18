"""Same-instance real M120 tree and independently prefilled path math."""
def run(engine,native,stage,output,fixture):
    from pathlib import Path
    import json
    from static_state255 import install
    from bank251 import build
    from oracle255 import run as qualify,CASE_COUNT
    if CASE_COUNT!=227:raise RuntimeError('explicit declared227 case set')
    activation=stage('ROOT255/activate-new-M120-backend',lambda:install(engine,native))
    bank=build(engine,native,stage,fixture['prefixes'])
    folder=Path(output)/'independent-node-math'
    stage('ROOT255/create-math-output',lambda:folder.mkdir(exist_ok=False)if engine.rank==0 else None)
    math=qualify(engine,native,stage,bank,output=folder)
    stage('ROOT255/release-independent-bank',lambda:engine._release_cache()if not engine.is_draft else None)
    return dict(rank=engine.rank,activation=activation,math=math,bank=bank['evidence'],
        physical_M=120,physical_query_sizes=[5]*24,
        actual_static_graph_calls=engine._branch_tail_state.static_graph_calls,
        actual_model_cases=227,whole_workload_AR_qualified=False,performance=False,adopted=False)
