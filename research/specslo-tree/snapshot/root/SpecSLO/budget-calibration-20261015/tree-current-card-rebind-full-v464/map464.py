"""Query retained ACL to reject stale physical mapping before a model launch."""
import ctypes
import json
import sys
from pathlib import Path


def main():
    runtime = ctypes.CDLL('/root/data/specslo-cann-runtime/cann-9.0.0/lib64/libascendcl.so')
    runtime.aclInit.argtypes = [ctypes.c_char_p]
    assert runtime.aclInit(None) == 0
    rows = []
    try:
        count = ctypes.c_uint32()
        assert runtime.aclrtGetDeviceCount(ctypes.byref(count)) == 0
        for user in range(count.value):
            logical = ctypes.c_int32()
            physical = ctypes.c_int32()
            assert runtime.aclrtGetLogicDevIdByUserDevId(ctypes.c_int32(user), ctypes.byref(logical)) == 0
            assert runtime.aclrtGetPhyDevIdByLogicDevId(logical, ctypes.byref(physical)) == 0
            rows.append({'user': user, 'logical': logical.value, 'physical': physical.value})
    finally:
        assert runtime.aclFinalize() == 0
    assert rows == [{'user': i, 'logical': i, 'physical': x} for i, x in enumerate([0, 1, 2, 7])], rows
    requested = [int(x) for x in sys.argv[1].split(',')]
    mapping = {r['physical']: r['logical'] for r in rows}
    destination = Path(__file__).resolve().parent / 'actual-retained-device-mapping.json'
    data = {'devices': rows, 'ACL_library': str(runtime._name), 'acl_set_device_called': False}
    if not destination.exists():
        destination.write_text(json.dumps(data, indent=2) + '\n')
    else:
        assert json.loads(destination.read_text()) == data
    print(','.join(str(mapping[x]) for x in requested))


if __name__ == '__main__':
    main()
