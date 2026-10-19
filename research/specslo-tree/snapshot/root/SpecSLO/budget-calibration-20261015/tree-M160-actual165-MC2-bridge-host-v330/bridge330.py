"""Private model adapter: FIA160/S32, actual qualified MC2 projection165.

HOST prototype only until fresh independent model and cost gates pass.
No source/profile/class mutation, no invented row160 MC2 qualification.
"""
from types import FunctionType, MethodType


def clone_with_binder(function, binder):
    namespace = dict(function.__globals__)
    if "bind_mc2_static_route" not in namespace:
        raise RuntimeError("Original decoder binder missing")
    namespace["bind_mc2_static_route"] = binder
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    return result


class ProjectionBridge:
    def __init__(self, original_binder, *, pad):
        self.original_binder = original_binder
        self.pad = pad
        self.active_layer = None
        self.pending = {}
        self.witnesses = {}

    def bind(self, route, profile, x, weight, residual, gamma):
        if self.active_layer is None:
            return self.original_binder(route, profile, x, weight, residual, gamma)
        if (x.ndim != 2 or tuple(residual.shape) != (160, 5120)
                or x.shape[0] != 160 or x.shape[1] not in (3072, 8576)
                or tuple(weight.shape) != (5120, x.shape[1])
                or tuple(gamma.shape) != (5120,)):
            raise RuntimeError("Bridge requires exact physical160 projection")
        if self.pending:
            raise RuntimeError("Previous padded projection ticket not consumed")
        xp = self.pad(x, (0, 0, 0, 5))
        rp = self.pad(residual, (0, 0, 0, 5))
        ticket = self.original_binder(route, profile, xp, weight, rp, gamma)
        if ticket.qualification.qualified is not True:
            raise RuntimeError("Actual165 route was not qualified")
        self.pending[id(ticket)] = (ticket, x, residual, weight, gamma,
                                    route, profile, xp, rp, self.active_layer)
        key = (self.active_layer, int(x.shape[1]))
        self.witnesses[key] = self.witnesses.get(key, 0) + 1
        return ticket

    def forward(self, original, x, residual, weight, context, **kwargs):
        ticket = kwargs.get("qualification")
        packet = self.pending.get(id(ticket))
        if packet is None:
            if self.active_layer is not None:
                raise RuntimeError("Active160 projection has no actual165 ticket")
            return original(x, residual, weight, context, **kwargs)
        (same, ox, ores, ow, gamma, route, profile, xp, rp, layer) = packet
        if (same is not ticket or ox is not x or ores is not residual
                or ow is not weight or original.__self__.weight is not gamma
                or kwargs.get("static_route") is not route
                or kwargs.get("profile") is not profile
                or layer != self.active_layer
                or kwargs.get("mc2_chain") is not None):
            raise RuntimeError("Padded projection ticket ownership changed")
        output = original(xp, rp, weight, context, **kwargs)
        if len(output) != 2 or any(tuple(t.shape) != (165, 5120) for t in output):
            raise RuntimeError("Original nonchained MC2 ABI requires two165 outputs")
        self.pending.pop(id(ticket))
        return output[0][:160], output[1][:160]

    def scope(self, index, function, *args, **kwargs):
        metadata = kwargs.get("attention_metadata", args[3] if len(args) > 3 else None)
        hidden = args[1] if len(args) > 1 else kwargs.get("hidden_states")
        active = bool(metadata is not None and metadata.tree_attention
                      and hidden.ndim == 2 and hidden.shape[0] == 160
                      and tuple(metadata.actual_seq_lengths_q) == tuple(range(5, 161, 5)))
        if not active:
            return function(*args, **kwargs)
        if self.active_layer is not None or self.pending:
            raise RuntimeError("Bridge scope is not reentrant")
        if kwargs.get("mc2_chain") is not None:
            raise RuntimeError("Physical160 has no qualified deferred MC2 chain")
        self.active_layer = index
        try:
            result = function(*args, **kwargs)
            if self.pending:
                raise RuntimeError("Decoder left an unconsumed padded projection")
            return result
        finally:
            self.active_layer = None

    def summary(self):
        return dict(FIA_rows=160, FIA_groups=32, actual_projection_MC2_rows=165,
                    candidate_queries_added=0, projection_padding_rows=5,
                    profile160_claimed=False, deferred_chain_enabled=False,
                    counts_are_Python_graph_construction_or_eager_not_device_replays=True,
                    construction_witnesses=[dict(layer=i, K=k, calls=v)
                        for (i,k),v in sorted(self.witnesses.items())],
                    pending_tickets=len(self.pending), model_qualified=False,
                    Goodput=None)


def install(model, native_model):
    """Instance-only rebinding; all native function code objects are retained."""
    import torch.nn.functional as F
    if hasattr(model, "_specslo_bridge330"):
        raise RuntimeError("Bridge330 must be installed exactly once")
    bridge = ProjectionBridge(native_model.bind_mc2_static_route, pad=F.pad)
    for index, layer in enumerate(model.layers):
        if not layer.enable_mc2 or not layer.enable_down_mc2:
            raise RuntimeError("Bridge requires both original qualified TP3 routes")
        if not layer.mc2_routes_frozen:
            raise RuntimeError("Bridge requires frozen original route identities")
        for route in (layer.mc2_attention_route, layer.mc2_down_route):
            if route is None or not route.decisions[165].qualified:
                raise RuntimeError("Original165 profile must qualify every layer route")
        original = layer.forward.__func__
        cloned = clone_with_binder(original, bridge.bind)
        bound = MethodType(cloned, layer)
        def layer_forward(self, *args, _index=index, _bound=bound, **kwargs):
            return bridge.scope(_index, _bound, *args, **kwargs)
        layer.forward = MethodType(layer_forward, layer)
        down = layer.mlp.forward_down_mc2.__func__
        layer.mlp.forward_down_mc2 = MethodType(clone_with_binder(down, bridge.bind), layer.mlp)
    for module in model.modules():
        if isinstance(module, native_model.NativeRMSNorm):
            original = module.forward_mc2
            def norm_forward(self, x, residual, weight, context, _original=original, **kwargs):
                return bridge.forward(_original, x, residual, weight, context, **kwargs)
            module.forward_mc2 = MethodType(norm_forward, module)
    model._specslo_bridge330 = bridge
    return bridge
