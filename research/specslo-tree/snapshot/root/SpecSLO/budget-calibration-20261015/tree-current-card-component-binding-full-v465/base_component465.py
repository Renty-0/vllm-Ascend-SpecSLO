"""Original173/209 math, with actual current-card checks inside the actor."""
from pathlib import Path
from types import FunctionType, SimpleNamespace
import base_component425 as original

ACTOR_CHANGES = (
    ('from import_origins_v132 import require_loaded_origins as assert_origins',
     'from extension465 import loaded_origins as assert_origins', 1),
    ('physical_cards=[2,5,6,7],physical_rank_card=[2,5,6,7][engine.rank]',
     'physical_cards=[0,1,2,7],physical_rank_card=[0,1,2,7][engine.rank]', 1),
    ("os.environ.get('SPECSLO_VISIBLE_DEVICES')=='2,5,6,7'",
     "os.environ.get('SPECSLO_VISIBLE_DEVICES')=='0,1,2,7'", 1),
)


def parent_component():
    import component199
    from base199 import P173, BASE, functions
    previous = component199._PARENT

    def selected_functions(path, namespace, **kwargs):
        if Path(path) == BASE / 'actor_v146.py':
            kwargs['changes'] = tuple(kwargs.get('changes', ())) + ACTOR_CHANGES
        return functions(path, namespace, **kwargs)

    namespace = dict(vars(previous), functions=selected_functions)
    functions(P173 / 'component173.py', namespace)
    return SimpleNamespace(**namespace)


def construct209():
    fn = original.construct209
    return FunctionType(fn.__code__, dict(fn.__globals__, parent_component=parent_component),
                        fn.__name__, fn.__defaults__, fn.__closure__)()


def qualify(engine, native, stage, policy, output):
    # Retain433's lifetime: close the preflight shadow only after all75 cases.
    return construct209()(engine, native, stage, policy, output)
