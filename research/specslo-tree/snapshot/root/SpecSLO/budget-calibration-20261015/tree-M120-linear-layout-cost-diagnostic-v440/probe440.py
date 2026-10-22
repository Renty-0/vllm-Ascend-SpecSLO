"""Reversible385 capture: retain one additional genuine FFN gate-up input."""
import inspect
from projection425 import project_function
import probe385 as original

def construct():
    changes=(
        ('from original385 import run as original','from linear440 import measure as original'),
        ("kind.forward_mc2=observed\n        try:","kind.forward_mc2=observed\n        gate=engine.model.layers[0].mlp.gate_up_proj\n        old_gate=gate.forward\n        def observed_gate(x):\n            if 'gate_up' not in payloads:\n                if gate.bias is not None:raise RuntimeError('ROOT440 original bias-free gate-up')\n                payloads['gate_up']=dict(activation=x.detach().clone(),weight=gate.weight)\n            return old_gate(x)\n        gate.forward=observed_gate\n        try:"),
        ('finally:kind.forward_mc2=old','finally:\n            kind.forward_mc2=old\n            gate.forward=old_gate'),
        ("if set(payloads)!={'attention','down'}:","if set(payloads)!={'attention','down','gate_up'}:"),
        ('ROOT385/original128-epilogue-graph-qualifier-and-balanced-cost','ROOT440/actual-device-event-local-linear-layout-padding-cost'),
        ('current_profile_unchanged=True,new_profile_installed=False,','current_profile_unchanged=True,new_profile_installed=False,'),
    )
    # Do not treat the identical declaration as a source edit.
    return project_function(original.run,changes[:-1])
run=construct()
