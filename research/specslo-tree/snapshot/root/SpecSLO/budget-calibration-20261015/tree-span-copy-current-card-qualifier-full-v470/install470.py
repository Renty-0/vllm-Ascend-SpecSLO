"""Preserve original B1..5 geometry; install span copies before fresh checks."""


def prepared():
    import span_install
    from projection425 import project_function
    return project_function(span_install.install, (
        ('    import target_geometry\n', '    from types import FunctionType\n'),
        ('desc = target_geometry.layout(*args, **kwargs)', 'desc = install377.layout(*args, **kwargs)'),
        ("    adapted = project_function(install377.install, (\n"
         "        ('from geometry377 import pad_function', 'from target_geometry import pad_function'),\n"
         "    ), dict(layout=checked_layout if not engine.is_draft else target_geometry.layout))",
         "    original_install = install377.install\n"
         "    adapted = FunctionType(original_install.__code__,\n"
         "        dict(original_install.__globals__, layout=checked_layout if not engine.is_draft else install377.layout),\n"
         "        original_install.__name__, original_install.__defaults__, original_install.__closure__)"),
        ('state.adaptive377_enabled = True', 'state.adaptive377_enabled = False'),
        ('state.uniform_KV_bucket425_enabled = True', 'state.uniform_KV_bucket425_enabled = False'),
    ))


def install(*args, **kwargs):
    return prepared()(*args, **kwargs)
