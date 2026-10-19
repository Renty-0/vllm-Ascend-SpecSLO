"""Unchanged326 neural protocol, with actual capacity/accounting binding."""
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
B=P.parent
sys.path.insert(0,str(B/'tree-M160-S32-N32-target-math-cost-v326'))
import runtime326 as original
import startup_config_v132 as configuration

base_config=configuration.make_config
base_worker=original.worker

def make_config():
    from dataclasses import replace
    return replace(base_config(),spec_rhythm_verification_budget=128)

def worker329(*args,**kwargs):
    from check329 import ready
    ready()
    cfg,rank=args[:2]
    if cfg.spec_rhythm_verification_budget!=128:
        raise RuntimeError('fresh constructor configuration must carryC128')
    import install234
    install=install234.install_entry
    def install_with_capacity(entry,native):
        policy=install(entry,native)
        from capacity329 import bind
        proof=bind(policy,native)
        import json
        path=Path(kwargs['output'])/f'rank{rank}-capacity329-before-model.json'
        with path.open('x')as stream:json.dump(dict(proof,rank=rank,constructor_candidates=cfg.spec_rhythm_verification_budget),stream,indent=2)
        return policy
    install234.install_entry=install_with_capacity
    try:
        return base_worker(*args,**kwargs)
    finally:
        install234.install_entry=install

configuration.make_config=make_config
original.worker=worker329
if __name__=='__main__':original.main()
