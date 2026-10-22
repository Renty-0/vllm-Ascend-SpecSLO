"""Lossless fixed-width success messages for the four-rank HOST stage protocol.

This is an uninstalled HOST prototype. Errors and unsupported stage names
retain the original object-gather path. No Torch, device or model import.
"""
MAX_STAGE_BYTES = 64
WORDS = 4 + MAX_STAGE_BYTES // 8


def encode(rank, sequence, stage, error):
    if type(rank) is not int or type(sequence) is not int or sequence < 0:
        raise ValueError('Original stage requires integer rank and nonnegative sequence')
    try:
        raw = stage.encode('utf-8') if type(stage) is str else None
    except UnicodeEncodeError:
        raw = None
    supported = raw is not None and len(raw) <= MAX_STAGE_BYTES
    length = len(raw) if supported else -1
    data = raw.ljust(MAX_STAGE_BYTES, b'\0') if supported else bytes(MAX_STAGE_BYTES)
    return (rank, sequence, int(error is not None), length,
            *(int.from_bytes(data[i:i+8], 'little', signed=True)
              for i in range(0, MAX_STAGE_BYTES, 8)))


def decode_success(rows):
    """Return exact old success objects, or request the original object path.

    The caller MUST collectively take the object path when this returns None.
    That rule covers errors, unbounded/non-string stage names, malformed input
    and rank/sequence/name disagreement. Nothing is accepted from a digest.
    """
    if len(rows) != 4:
        return None
    decoded = []
    try:
        for row in rows:
            if len(row) != WORDS or any(type(x) is not int for x in row):
                return None
            rank, sequence, failed, length, *words = row
            if failed != 0 or not 0 <= length <= MAX_STAGE_BYTES or sequence < 0:
                return None
            data = b''.join(x.to_bytes(8, 'little', signed=True) for x in words)
            if any(data[length:]):
                return None
            stage = data[:length].decode('utf-8')
            decoded.append(dict(rank=rank, sequence=sequence, stage=stage, error=None))
    except (OverflowError, UnicodeDecodeError, ValueError):
        return None
    first = decoded[0]
    if (sorted(row['rank'] for row in decoded) != list(range(4))
            or any(row['sequence'] != first['sequence'] or row['stage'] != first['stage']
                   for row in decoded)):
        return None
    return decoded
