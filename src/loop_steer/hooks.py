"""Residual-stream hooks for standard and looped (Ouro) HF decoder models.

Sites are ``(loop, layer)``. Standard models have one loop (loop=0). Ouro
applies the same ``model.model.layers`` stack ``total_ut_steps`` times and
passes the zero-based loop index as the ``current_ut`` kwarg, so every hook
here reads it to tell loops apart.

All hooks act on the *residual stream*:
- capture / add: input to decoder layer ``layer`` in loop ``loop`` (resid_pre).
- ablate: every write into the residual stream (see ``ablation_hooks``).
"""

from contextlib import contextmanager

import torch


def _loop(kwargs) -> int:
    return kwargs.get("current_ut", 0)


def _layer_input(args, kwargs) -> torch.Tensor:
    return args[0] if args else kwargs["hidden_states"]


def _replace_layer_input(args, kwargs, new):
    if args:
        return (new, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": new}


def decoder_layers(model):
    return model.model.layers


@contextmanager
def hooked(handles_fn):
    """Register hooks via ``handles_fn() -> list[RemovableHandle]`` and always remove them."""
    handles = handles_fn()
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------- capture
@contextmanager
def capture_window_means(model, weights: torch.Tensor, layers=None, loops=None):
    """Accumulate weighted token-means of resid_pre at each (loop, layer).

    ``weights``: [W, B, T] per-window token weights (rows sum to 1, zeros on
    padding / outside the window). Yields a dict ``{(loop, layer): [W, B, D]}``
    (float32, CPU) that is filled once the forward pass runs.
    """
    out = {}
    blocks = decoder_layers(model)
    layers = range(len(blocks)) if layers is None else layers

    def make(layer):
        def hook(module, args, kwargs):
            loop = _loop(kwargs)
            if loops is not None and loop not in loops:
                return
            h = _layer_input(args, kwargs)
            w = weights.to(device=h.device, dtype=torch.float32)
            out[(loop, layer)] = torch.einsum("wbt,btd->wbd", w, h.float()).cpu()
        return hook

    with hooked(lambda: [blocks[l].register_forward_pre_hook(make(l), with_kwargs=True) for l in layers]):
        yield out


# ---------------------------------------------------------------- interventions
def _project_out(x: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    u = unit.to(x)
    return x - (x @ u).unsqueeze(-1) * u


def actadd_hooks(model, vector: torch.Tensor, layer: int, coeff: float = 1.0, loops=None):
    """Add ``coeff * vector`` to resid_pre of ``layer`` (at all positions) in the given loops."""
    block = decoder_layers(model)[layer]

    def hook(module, args, kwargs):
        if loops is not None and _loop(kwargs) not in loops:
            return None
        h = _layer_input(args, kwargs)
        return _replace_layer_input(args, kwargs, h + coeff * vector.to(h))

    return lambda: [block.register_forward_pre_hook(hook, with_kwargs=True)]


def ablation_hooks(model, direction: torch.Tensor, loops=None, sites=("resid_pre", "attn", "mlp")):
    """Project ``direction`` out of every residual write (Arditi et al. directional ablation).

    For pre-norm models (Qwen/Llama) this is: layer input + self_attn output +
    mlp output at every layer, which is equivalent to weight orthogonalization.
    Ouro is dispatched to ``ouro_ablation_hooks`` (sandwich norms).
    """
    if hasattr(decoder_layers(model)[0], "input_layernorm_2"):
        return ouro_ablation_hooks(model, direction, loops=loops)
    unit = direction / direction.norm()
    blocks = decoder_layers(model)

    def pre(module, args, kwargs):
        if loops is not None and _loop(kwargs) not in loops:
            return None
        return _replace_layer_input(args, kwargs, _project_out(_layer_input(args, kwargs), unit))

    def post(module, args, output):
        if isinstance(output, tuple):
            return (_project_out(output[0], unit), *output[1:])
        return _project_out(output, unit)

    def register():
        handles = []
        for block in blocks:
            if "resid_pre" in sites:
                handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
            if "attn" in sites:
                handles.append(block.self_attn.register_forward_hook(post))
            if "mlp" in sites:
                handles.append(block.mlp.register_forward_hook(post))
        return handles

    return register


def ouro_ablation_hooks(model, direction: torch.Tensor, loops=None):
    """Directional ablation for Ouro's sandwich-norm, looped residual stream.

    Ouro writes ``RMSNorm_2(sublayer(x))`` into the residual, and the norm's
    learned gain can re-introduce a direction projected out of ``o_proj`` /
    ``down_proj`` -- so weight orthogonalization is not an ablation here. We
    instead project the direction out of (a) every layer input in every loop,
    which covers the embeddings and the inter-loop ``model.norm`` output, and
    (b) the outputs of ``input_layernorm_2`` and ``post_attention_layernorm_2``
    (the attention and MLP residual writes). ``loops`` restricts this to the
    given zero-based loop indices.
    """
    unit = direction / direction.norm()
    state = {"loop": 0}  # set by each layer's pre-hook; its norms run inside that call

    def active():
        return loops is None or state["loop"] in loops

    def pre(module, args, kwargs):
        state["loop"] = _loop(kwargs)
        if not active():
            return None
        return _replace_layer_input(args, kwargs, _project_out(_layer_input(args, kwargs), unit))

    def post(module, args, output):
        return _project_out(output, unit) if active() else None

    def register():
        handles = []
        for block in decoder_layers(model):
            handles.append(block.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(block.input_layernorm_2.register_forward_hook(post))
            handles.append(block.post_attention_layernorm_2.register_forward_hook(post))
        return handles

    return register
