"""Narrow, opt-in cleanup of Native's redundant default-CPU function mode.

This module does not install hooks, import torch, execute a model, or disable
TorchFunctionMode.  Call capture before the original Native constructor and
restore after it.  Any state outside the exact checked transition is preserved.
The caller must run this on the same thread, outside all measurement intervals.

State restoration is not global factory equivalence: tensor/as_tensor/asarray
can inherit a device from Tensor data when no default DeviceContext exists.
The caller must separately audit executed factories and qualify its component.
"""
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class ConstructorState:
    torch: object
    thread_id: int
    device_module: object
    context_holder: object
    original_global_context: object
    mode_stack: tuple
    current_device: object
    default_dtype: object
    default_device: object
    legacy_default_device: object
    api_functions: tuple
    eligible: bool
    reason: str


def _device_key(device):
    return None if device is None else (device.type, device.index)


def _same_stack(left, right):
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _summary(raw):
    return {
        "mode_stack": [{"class": type(mode).__module__ + "." + type(mode).__qualname__,
                        "object_id": id(mode),
                        "device": str(mode.device) if isinstance(mode, raw["device_class"]) else None}
                       for mode in raw["stack"]],
        "global_context_object_id": (None if raw["global_context"] is None
                                     else id(raw["global_context"])),
        "current_device": str(raw["current_device"]) if raw["current_device"] is not None else None,
        "default_device": str(raw["default_device"]) if raw["default_device"] is not None else None,
        "default_dtype": str(raw["dtype"]),
        "legacy_default_device": raw["legacy_default_device"],
        "default_device_read": raw["default_device"] is not None,
    }


def _raw(torch, device_module):
    # Do not call get_default_device for a non-CPU DeviceContext: PyTorch's
    # implementation can allocate a tensor to resolve an unspecified index.
    stack = tuple(torch.overrides._get_current_function_mode_stack())
    context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
    legacy_getter = getattr(torch._C, "_get_default_device", None)
    legacy_device = legacy_getter() if callable(legacy_getter) else None
    contexts = [mode for mode in stack if isinstance(mode, device_module.DeviceContext)]
    non_cpu = any(mode.device.type != "cpu" for mode in contexts)
    non_cpu = non_cpu or (context is not None and
                         (not isinstance(context, device_module.DeviceContext) or context.device.type != "cpu"))
    return {"stack": stack, "global_context": context,
            "current_device": device_module.CURRENT_DEVICE,
            "device_class": device_module.DeviceContext,
            "default_device": None if non_cpu or legacy_device != "cpu" else torch.get_default_device(),
            "legacy_default_device": legacy_device,
            "dtype": torch.get_default_dtype()}


def capture_before_native_constructor(torch):
    """Record object identities; never change device, dtype, modes, or APIs."""
    import importlib
    device_module = importlib.import_module("torch.utils._device")
    raw = _raw(torch, device_module)
    if any(isinstance(mode, device_module.DeviceContext) for mode in raw["stack"]):
        reason = "before_already_has_device_context"
    elif raw["global_context"] is not None:
        reason = "before_has_global_device_context"
    elif raw["legacy_default_device"] != "cpu":
        reason = "before_legacy_default_device_is_not_cpu"
    elif _device_key(raw["default_device"]) != ("cpu", None):
        reason = "before_default_device_is_not_plain_cpu"
    elif raw["current_device"] is not None:
        reason = "before_current_device_is_not_none"
    else:
        reason = "eligible_plain_cpu_baseline"
    return ConstructorState(torch, threading.get_ident(), device_module,
                            torch._GLOBAL_DEVICE_CONTEXT, raw["global_context"], raw["stack"], raw["current_device"],
                            raw["dtype"], raw["default_device"], raw["legacy_default_device"],
                            (torch.set_default_device, torch.get_default_device,
                             torch.get_default_dtype, torch.overrides._get_current_function_mode_stack,
                             getattr(torch._C, "_get_default_device", None)),
                            reason == "eligible_plain_cpu_baseline", reason)


def _baseline_summary(state):
    return _summary({"stack": state.mode_stack, "global_context": state.original_global_context,
                     "current_device": state.current_device, "device_class": state.device_module.DeviceContext,
                     "default_device": state.default_device, "dtype": state.default_dtype,
                     "legacy_default_device": state.legacy_default_device})


def _identity_reason(torch, state):
    if torch is not state.torch or threading.get_ident() != state.thread_id:
        return "different_torch_or_thread"
    if not state.eligible:
        return state.reason
    if torch._GLOBAL_DEVICE_CONTEXT is not state.context_holder:
        return "global_context_holder_changed"
    if not _same_stack((torch.set_default_device, torch.get_default_device, torch.get_default_dtype,
                        torch.overrides._get_current_function_mode_stack,
                        getattr(torch._C, "_get_default_device", None)), state.api_functions):
        return "torch_api_changed"
    return None


def _shared_reason(torch, state, raw):
    reason = _identity_reason(torch, state)
    if reason:
        return reason
    if raw["dtype"] is not state.default_dtype:
        return "default_dtype_changed"
    if raw["legacy_default_device"] != state.legacy_default_device:
        return "legacy_default_device_changed"
    if _device_key(raw["default_device"]) != _device_key(state.default_device):
        return "default_device_changed_or_not_read"
    return None


def _baseline_reason(torch, state, raw):
    reason = _shared_reason(torch, state, raw)
    if reason:
        return reason
    if not _same_stack(raw["stack"], state.mode_stack):
        return "mode_stack_is_not_baseline"
    if raw["global_context"] is not None:
        return "global_device_context_is_not_none"
    if raw["current_device"] is not state.current_device:
        return "current_device_is_not_baseline"
    return None


def _redundant_reason(torch, state, raw):
    reason = _shared_reason(torch, state, raw)
    if reason:
        return reason
    stack, cls = raw["stack"], state.device_module.DeviceContext
    if len(stack) != len(state.mode_stack) + 1 or not _same_stack(stack[1:], state.mode_stack):
        return "after_is_not_one_extra_bottom_mode"
    mode = stack[0]
    if type(mode) is not cls or _device_key(mode.device) != ("cpu", None):
        return "extra_bottom_mode_is_not_exact_plain_cpu_device_context"
    if any(isinstance(other, cls) for other in stack[1:]):
        return "another_device_context_exists"
    if raw["global_context"] is not mode:
        return "extra_mode_is_not_global_context_owned"
    if raw["current_device"] is not mode.device or mode.old_device is not state.current_device:
        return "device_context_restore_state_is_not_baseline"
    return None


def observe_cpu_device_context(torch, state, *, redundant_cpu):
    """Read and validate one control state, without changing it or its modes."""
    if type(redundant_cpu) is not bool:
        raise TypeError("redundant_cpu must be an explicit bool")
    reason = _identity_reason(torch, state)
    if reason:
        return {"passed": False, "reason": reason, "baseline": _baseline_summary(state),
                "mutation": None, "expected_redundant_cpu": redundant_cpu}
    raw = _raw(torch, state.device_module)
    reason = (_redundant_reason if redundant_cpu else _baseline_reason)(torch, state, raw)
    return {"passed": reason is None, "reason": reason,
            "baseline": _baseline_summary(state), "observed": _summary(raw),
            "mutation": None, "expected_redundant_cpu": redundant_cpu,
            "global_factory_equivalence_proved": False}


def restore_redundant_cpu_device_context(torch, state):
    """Remove exactly one global-owned bottom CPU context, otherwise no-op.

    A failure after the sole authorized mutation raises, so a caller cannot
    silently label an unverified state as cleaned.  No model or accelerator
    execution occurs.  PyTorch get_default_device may construct an empty CPU
    tensor when a CPU DeviceContext is present.
    """
    identity_reason = _identity_reason(torch, state)
    if identity_reason:
        return {"status": "preserved", "reason": identity_reason, "mutation": None,
                "baseline": _baseline_summary(state)}
    raw = _raw(torch, state.device_module)
    reason = _redundant_reason(torch, state, raw)
    result = {"status": "preserved", "reason": reason, "baseline": _baseline_summary(state),
              "observed": _summary(raw), "mutation": None,
              "global_factory_equivalence_proved": False}
    if reason:
        return result
    torch.set_default_device(None)
    final = _raw(torch, state.device_module)
    mismatch = _baseline_reason(torch, state, final)
    if mismatch:
        raise RuntimeError("CPU DeviceContext cleanup postcondition failed: " + mismatch)
    result.update(status="cleaned", reason="exact_redundant_cpu_context_removed",
                  final=_summary(final), mutation="torch.set_default_device(None)")
    return result


def reinstall_redundant_cpu_device_context(torch, state):
    """Reinstall the checked CPU mode for a diagnostic control, outside timing.

    This is an explicit control intervention, never constructor behavior or a
    performance optimization.  Only an exact baseline state may be changed.
    """
    identity_reason = _identity_reason(torch, state)
    if identity_reason:
        return {"status": "preserved", "reason": identity_reason, "mutation": None,
                "baseline": _baseline_summary(state)}
    raw = _raw(torch, state.device_module)
    reason = _baseline_reason(torch, state, raw)
    result = {"status": "preserved", "reason": reason, "baseline": _baseline_summary(state),
              "observed": _summary(raw), "mutation": None,
              "global_factory_equivalence_proved": False}
    if reason:
        return result
    torch.set_default_device("cpu")
    final = _raw(torch, state.device_module)
    mismatch = _redundant_reason(torch, state, final)
    if mismatch:
        raise RuntimeError("CPU DeviceContext reinstall postcondition failed: " + mismatch)
    result.update(status="reinstalled", reason="exact_redundant_cpu_context_installed",
                  final=_summary(final), mutation="torch.set_default_device('cpu')")
    return result
