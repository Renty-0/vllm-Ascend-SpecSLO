"""Resolve current requested physical cards without selecting or allocating a device."""
import ctypes
import json
from pathlib import Path
import sys


def main():
    requested = [int(value) for value in sys.argv[1].split(",")]
    if len(requested) != 4 or len(set(requested)) != 4:
        raise RuntimeError("Exactly four distinct physical cards are required for TP4")
    runtime = ctypes.CDLL("/root/data/specslo-cann-runtime/cann-9.0.0/lib64/libascendcl.so")
    runtime.aclInit.argtypes = [ctypes.c_char_p]
    if runtime.aclInit(None) != 0:
        raise RuntimeError("ACL initialization failed during mapping query")
    rows = []
    try:
        count = ctypes.c_uint32()
        assert runtime.aclrtGetDeviceCount(ctypes.byref(count)) == 0
        for user in range(count.value):
            logical = ctypes.c_int32()
            physical = ctypes.c_int32()
            assert runtime.aclrtGetLogicDevIdByUserDevId(ctypes.c_int32(user), ctypes.byref(logical)) == 0
            assert runtime.aclrtGetPhyDevIdByLogicDevId(logical, ctypes.byref(physical)) == 0
            rows.append({"user": user, "logical": logical.value, "physical": physical.value})
    finally:
        assert runtime.aclFinalize() == 0
    mapping = {row["physical"]: row["logical"] for row in rows}
    if len(mapping) != len(rows) or any(card not in mapping for card in requested):
        raise RuntimeError(f"Requested physical cards are not available: {requested}; actual {rows}")
    # Preserve the original mapper's established user/logical namespace contract.
    if any(row["user"] != row["logical"] for row in rows):
        raise RuntimeError(f"Unexpected user/logical namespace; refusing stale mapping: {rows}")
    data = {"devices": rows, "requested_physical_devices": requested,
            "ACL_library": str(runtime._name), "acl_set_device_called": False,
            "container_runtime_devices": [mapping[card] for card in requested]}
    destination = Path(__file__).resolve().parent / "actual-retained-device-mapping.json"
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(",".join(str(mapping[card]) for card in requested))


if __name__ == "__main__":
    main()
