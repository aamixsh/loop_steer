"""Residual-stream interventions inside a vLLM engine (requires ``enforce_eager=True``).

Shipped to the worker with ``LLM.apply_model(functools.partial(install, specs=...))``.
vLLM decoder layers use the fused-residual convention: layer ``l`` is called as
``layer(positions, hidden_states, [loop_idx,] residual)`` (gpt-oss: ``layer(hidden_states, positions, residual)``) and
the residual stream entering it is ``hidden_states + residual`` (``residual`` is None for the first layer of each
pass, where ``hidden_states`` is the full stream). Residual writes are the outputs of ``self_attn`` and ``mlp``
(gpt-oss: ``attn`` and ``mlp``, the MoE block: router-weighted sum of the selected experts' outputs).

Spec dicts (tensors may be on CPU):
  {"kind": "actadd", "vec": [D], "layer": l, "coeff": c, "loops": None | [t, ...], "positions": None | [cls, ...]}
      resid_pre(l) += coeff * vec in the given loops (all positions).
  {"kind": "ablate", "dir": [D], "loops": None | [t, ...], "positions": None | [cls, ...]}
      project dir out of every layer input and every residual write
      (attn/mlp outputs; for Ouro, the outputs of the post-sublayer norms).
  {"kind": "record"}
      count the tokens of each position class per request (``fetch_records``); for validation.

Position classes (``positions``): ``prompt`` (the formatted user turn, up to the generation prompt), ``cot`` (thinking
tokens up to and including ``</think>``), ``answer`` (after it). A token's class depends only on its position in its
sequence, so it is the same in stage 1 and stage 2 and prefix caching stays consistent. The class boundaries come from
each request's ``SamplingParams.extra_args["ls_bounds"] = (P, C)``: positions ``< P`` are prompt, ``< C`` (or all, if
``C < 0``) CoT, the rest answer; ``loop_steer.sampling.vllm_generator(gated=True)`` sets them. Tokens of requests
without bounds get no class and are never intervened on.
"""

import torch

CLASSES = {"prompt": 0, "cot": 1, "answer": 2}


def _is_ouro(model) -> bool:
    return hasattr(model.model.layers[0], "input_layernorm_2")


def _is_gpt_oss(model) -> bool:
    return hasattr(model.model.layers[0], "attn") and not hasattr(model.model.layers[0], "self_attn")


def _is_looped(model) -> bool:
    """Layers take (positions, hidden, loop_idx, residual): Ouro, or the Nanbeige port."""
    return _is_ouro(model) or hasattr(model.model, "num_loops")


def clear(model):
    for h in getattr(model, "_ls_handles", []):
        h.remove()
    model._ls_handles = []
    return True


def _model_runner():
    """The worker's GPU model runner (holds the request states).

    ``install`` runs inside ``Worker.apply_model(fn)``, so the worker is a ``self`` up the call stack. (vLLM freezes
    the garbage collector after start-up, so ``gc.get_objects()`` cannot find the runner.)
    """
    import sys

    frame = sys._getframe(1)
    while frame is not None:
        runner = getattr(frame.f_locals.get("self"), "model_runner", None)
        if runner is not None and hasattr(runner, "requests") and hasattr(runner, "input_batch"):
            return runner
        frame = frame.f_back
    raise RuntimeError("no vLLM model runner up the call stack: call install() through LLM.apply_model")


def _token_classes(runner, positions):
    """Per-token position class (-1: none) of the current forward pass, from the attention metadata."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    n = positions.shape[0]
    cls = torch.full((n,), -1, dtype=torch.int8, device=positions.device)
    if not is_forward_context_available():
        return cls, None
    md = get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0] if md else None
    if not md:
        return cls, None
    qsl = next(iter(md.values())).query_start_loc
    n_reqs = qsl.numel() - 1
    req_ids = runner.input_batch.req_ids[:n_reqs]
    bounds = []
    for r in req_ids:
        state = runner.requests.get(r)
        extra = (state.sampling_params.extra_args or {}) if state is not None and state.sampling_params else {}
        bounds.append(extra.get("ls_bounds", (-1, -1)))
    if not bounds:
        return cls, req_ids
    b = torch.tensor(bounds, dtype=torch.long, device=positions.device)
    tok = torch.arange(n, device=positions.device)
    req = torch.searchsorted(qsl[1:].long(), tok, right=True)  # request of each token (n_reqs: padding)
    valid = req < n_reqs
    req = req.clamp(max=n_reqs - 1)
    P, C = b[req, 0], b[req, 1]
    pos = positions.long()
    c = torch.where(pos < P, 0, torch.where((C < 0) | (pos < C), 1, 2))
    cls = torch.where(valid & (P >= 0), c, -1).to(torch.int8)
    return cls, (req_ids, req, valid)


def install(model, specs):
    """Remove previously installed hooks, then install ``specs``."""
    clear(model)
    layers = model.model.layers
    ouro = _is_ouro(model)
    looped = _is_looped(model)
    gpt_oss = _is_gpt_oss(model)
    pos_arg = 1 if gpt_oss else 0  # index of the positions argument of a decoder layer
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    state = {"loop": 0, "cls": None, "masks": {}}
    gated = any(s.get("positions") is not None or s["kind"] == "record" for s in specs)
    handles = []

    if gated:
        runner = _model_runner()
        model._ls_records = getattr(model, "_ls_records", {})
        recording = any(s["kind"] == "record" for s in specs)

        # once per forward pass: layer 0 in loop 0 (a hook on model.model can be bypassed by vLLM's compile wrapper)
        def classify(module, args):
            if looped and args[2] != 0:
                return
            state["cls"], info = _token_classes(runner, args[pos_arg])
            state["masks"] = {}
            if recording and info is not None:
                req_ids, req, valid = info
                for i, r in enumerate(req_ids):
                    counts = torch.bincount(state["cls"][valid & (req == i)].long() + 1, minlength=4).tolist()
                    rec = model._ls_records.setdefault(r, [0, 0, 0, 0])  # no class, prompt, cot, answer
                    for k in range(4):
                        rec[k] += counts[k]

        handles.append(layers[0].register_forward_pre_hook(classify))

    def mask(positions):
        """Float [N, 1] mask of the tokens in the given classes (None: every token)."""
        if positions is None:
            return None
        key = tuple(sorted(positions))
        if key not in state["masks"]:
            allowed = torch.tensor([CLASSES[p] for p in positions], dtype=torch.int8, device=device)
            state["masks"][key] = torch.isin(state["cls"], allowed).to(dtype).unsqueeze(-1)
        return state["masks"][key]

    def fit(m, x):  # mask rows to the hidden-state rows (padding rows get no intervention)
        if m is None or m.shape[0] == x.shape[0]:
            return m
        out = torch.zeros(x.shape[0], 1, dtype=m.dtype, device=m.device)
        k = min(x.shape[0], m.shape[0])
        out[:k] = m[:k]
        return out

    def unpack(args):  # -> (hidden, loop, residual, rebuild)
        if gpt_oss:
            h, pos, res = args[0], args[1], args[2] if len(args) > 2 else None
            return h, 0, res, lambda h2, r2: (h2, pos, r2, *args[3:])
        if looped:
            pos, h, loop, res = args[0], args[1], args[2], args[3] if len(args) > 3 else None
            return h, loop, res, lambda h2, r2: (pos, h2, loop, r2, *args[4:])
        pos, h, res = args[0], args[1], args[2] if len(args) > 2 else None
        return h, 0, res, lambda h2, r2: (pos, h2, r2, *args[3:])

    def track(module, args):
        state["loop"] = args[2] if looped else 0

    handles += [layer.register_forward_pre_hook(track) for layer in layers]
    for spec in specs:
        if spec["kind"] == "record":
            continue
        loops = spec.get("loops")
        pos_cls = spec.get("positions")
        active = (lambda: True) if loops is None else (lambda loops=loops: state["loop"] in loops)

        if spec["kind"] == "actadd":
            vec = (spec["coeff"] * spec["vec"].float()).to(device=device, dtype=dtype)

            def add(module, args, vec=vec, active=active, pos_cls=pos_cls):
                if not active():
                    return None
                h, _, res, rebuild = unpack(args)
                m = fit(mask(pos_cls), h)
                return rebuild(h + (vec if m is None else m * vec), res)

            handles.append(layers[spec["layer"]].register_forward_pre_hook(add))

        elif spec["kind"] == "ablate":
            u = spec["dir"].float()
            u = (u / u.norm()).to(device=device, dtype=dtype)

            def proj(x, u=u, pos_cls=pos_cls):
                coef = (x @ u).unsqueeze(-1)
                m = fit(mask(pos_cls), x)
                return x - (coef if m is None else m * coef) * u

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
                    handles.append((layer.attn if gpt_oss else layer.self_attn).register_forward_hook(post))
                    handles.append(layer.mlp.register_forward_hook(post))
        else:
            raise ValueError(f"unknown intervention kind {spec['kind']!r}")
    model._ls_handles = handles
    return True


def fetch_records(model, reset=True):
    """Token counts per request id recorded by a ``record`` spec: [no class, prompt, cot, answer]."""
    out = dict(getattr(model, "_ls_records", {}))
    if reset:
        model._ls_records = {}
    return out
