"""Keep original narrow proposals; route only N25..32 to the frozen wide padder."""


def pad(engine, plans, roots, rows, ids, active, states):
    if not 1 <= len(plans) <= 32:
        raise ValueError('HOST408 complete positive target batch at most32')
    if len(plans) <= 24:
        from padding255 import pad as original
    else:
        from padding399 import pad as original
    return original(engine, plans, roots, rows, ids, active, states)
