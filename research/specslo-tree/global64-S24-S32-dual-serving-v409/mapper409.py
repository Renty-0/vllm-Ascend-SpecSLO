"""Declared raw padding geometry for complete observed target calls."""
def envelope(n, q, family):
    if n > 24:
        from mapper399 import envelope as original
    else:
        from mapper255 import envelope as original
    return original(n, q, family)
