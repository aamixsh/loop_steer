"""Residual-stream interventions inside a vLLM engine (requires ``enforce_eager=True``).

Shipped to the worker with ``LLM.apply_model(functools.partial(install, specs=...))``.
vLLM decoder layers use the fused-residual convention: layer ``l`` is called as
``layer(positions, hidden_states, [current_ut,] residual)`` and the residual
stream entering it is ``hidden_states + residual`` (``residual`` is None for
the first layer of each pass, where ``hidden_states`` is the full stream).

Spec dicts (tensors may be on CPU):
  {"kind": "actadd", "vec": [D], "layer": l, "coeff": c, "loops": None | [t, ...]}
      resid_pre(l) += coeff * vec in the given loops (all positions).
  {"kind": "ablate", "dir": [D], "loops": None | [t, ...]}
      project dir out of every layer input and every residual write
      (attn/mlp outputs; for Ouro, the outputs of the post-sublayer norms).
"""

import torch


def _is_ouro(model) -> bool:
    return hasattr(model.model.layers[0], "input_layernorm_2")


def clear(model):
    for h in getattr(model, "_ls_handles", []):
        h.remove()
    model._ls_handles = []
    return True


def install(model, specs):
    """Remove previously installed hooks, then install ``specs``."""
    clear(model)
    layers = model.model.layers
    ouro = _is_ouro(model)
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    state = {"loop": 0}

    def unpack(args):  # -> (hidden, loop, residual, rebuild)
        if ouro:
            pos, h, loop, res = args[0], args[1], args[2], args[3] if len(args) > 3 else None
            return h, loop, res, lambda h2, r2: (pos, h2, loop, r2, *args[4:])
        pos, h, res = args[0], args[1], args[2] if len(args) > 2 else None
        return h, 0, res, lambda h2, r2: (pos, h2, r2, *args[3:])

    def track(module, args):
        state["loop"] = args[2] if ouro else 0

    handles = [layer.register_forward_pre_hook(track) for layer in layers]
    for spec in specs:
        loops = spec.get("loops")
        active = (lambda: True) if loops is None else (lambda loops=loops: state["loop"] in loops)

        if spec["kind"] == "actadd":
            vec = (spec["coeff"] * spec["vec"].float()).to(device=device, dtype=dtype)

            def add(module, args, vec=vec, active=active):
                if not active():
                    return None
                h, _, res, rebuild = unpack(args)
                return rebuild(h + vec, res)

            handles.append(layers[spec["layer"]].register_forward_pre_hook(add))

        elif spec["kind"] == "ablate":
            u = spec["dir"].float()
            u = (u / u.norm()).to(device=device, dtype=dtype)

            def proj(x, u=u):
                return x - (x @ u).unsqueeze(-1) * u

            def pre(module, args, proj=proj, active=active):
                if not active():
                    return None
                h, _, res, rebuild = unpack(args)
                return rebuild(proj(h), None if res is None else proj(res))

            def post(module, args, output, proj=proj, active=active):
                if not active():
                    return None
                if isinstance(output, tuple):
                    return (proj(output[0]), *output[1:])
                return proj(output)

            for layer in layers:
                handles.append(layer.register_forward_pre_hook(pre))
                if ouro:
                    handles.append(layer.input_layernorm_2.register_forward_hook(post))
                    handles.append(layer.post_attention_layernorm_2.register_forward_hook(post))
                else:
                    handles.append(layer.self_attn.register_forward_hook(post))
                    handles.append(layer.mlp.register_forward_hook(post))
        else:
            raise ValueError(f"unknown intervention kind {spec['kind']!r}")
    model._ls_handles = handles
    return True
