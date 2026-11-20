"""Interventions and activation capture inside vLLM's Inkling 2 Mini (``Inkling2LoopModel``, vLLM PR 69028).

Inkling has no single residual stream. Its 24 layers are tied together by block attention residuals: the outputs of
each block of 6 layers are summed into a block representation (``stack[b]``; ``stack[0]`` is the embedding), and every
sublayer input is a learned softmax mix of the RMS-normalized completed blocks and the running partial block, then an
RMSNorm. Layers 0-5 (prelude) run once, the core (layers 6-17, two blocks) runs ``num_passes`` times with shared
weights, and layers 18-23 (coda) run once from the last pass's state. Every pass restarts from the prelude's state;
from the second pass on, each core sublayer input is blended with a message read from the previous pass's two blocks
through a gated rank-64 controller.

Sites (``(loop, layer)``, as ``loop_steer.models.model_sites``): the prelude and coda layers are loop 0 (they run
once), the core layers are loop = pass. The activation at a site is its **resid_pre**: the running partial block as the
layer visit starts (for the first layer of a block, the block just completed; for layer 6 in every pass, the prelude's
block). It is the only additive stream of the model, and what activation addition and kicks change: a vector added to it
stays in that block, so every later sublayer that reads the block sees it (in a core pass: the rest of that pass, the
next pass through its messages, and for the last pass, the coda).

Unit selection (``loops``), as Dory's: core passes are 0, 1, 2; the prelude and coda count as loop -1 (``[-1, 0, 1, 2]``
is everything); -2 selects the prelude alone (whose state every pass restarts from), -3 the coda alone.

Spec dicts (tensors may be on CPU), installed with ``LLM.apply_model(functools.partial(install, specs=...))``:
  {"kind": "ablate", "dir": [D], "loops": None | [...], "positions": None | [cls, ...]}
      directional ablation in the selected units: ``dir`` is projected out of every sublayer input (the fp32 mixes
      before their RMSNorm, and the cross-pass messages), every write (the attention output after its short conv, and
      the block after the layer's update), and, with the prelude selected, the embedding; with the coda selected, the
      final mix before the last norm.
  {"kind": "actadd", "vec": [D], "layer": l, "coeff": c, "loops": None | [t, ...], "positions": None | [cls, ...]}
      resid_pre(l) += coeff * vec in the selected passes (prelude / coda layers: any selection containing -1 or 0).
  {"kind": "kick", "vec": [D], "layer": l, "eps": e, "loops": ..., "sink_ratio": r}
      resid_pre(l) += eps * |resid_pre| * unit(vec) per token except attention sinks (position 0 and tokens with
      |resid_pre| > r * the median of the first forward pass with >= 64 tokens at that site), as ``vllm_hooks``.
  {"kind": "capture", "n_req": R, "n_windows": W}
      accumulate, per request with ``extra_args["ls_cap"] = (k, [(lo, hi)] * W)`` (k < R), the sum of resid_pre over
      the tokens at positions ``lo <= pos < hi`` of each window at every site, and the token counts. Run without prefix
      caching (cached tokens are not computed) and fetch with ``fetch_capture``.

Position classes (``positions``): as ``loop_steer.vllm_hooks`` (``extra_args["ls_bounds"]``).

A layer visit no spec touches runs vLLM's fused path unchanged; touched visits run the same ops unfused (short conv,
then the AttnRes mix), which differs from the fused kernels only in rounding. Needs ``enforce_eager=True``.
"""

import torch

from loop_steer.vllm_hooks import CLASSES


# ---------------------------------------------------------------------- vLLM model runner (v2, ``v1/worker/gpu``)
def _model_runner():
    """The worker's GPU model runner, found up the call stack of ``Worker.apply_model`` (as ``vllm_hooks``)."""
    import sys

    frame = sys._getframe(1)
    while frame is not None:
        runner = getattr(frame.f_locals.get("self"), "model_runner", None)
        if runner is not None and hasattr(runner, "req_states") and hasattr(runner, "prepare_inputs"):
            _patch_runner(runner)
            return runner
        frame = frame.f_back
    raise RuntimeError("no vLLM model runner up the call stack: call install() through LLM.apply_model")


def _patch_runner(runner):
    """Keep each request's ``SamplingParams.extra_args`` (``runner._ls_extra``) and the batch being run
    (``runner._ls_batch``): the v2 runner keeps neither in a form the forward pass can read."""
    cls = type(runner)
    if hasattr(cls, "_ls_orig_add_requests"):
        return
    cls._ls_orig_add_requests, cls._ls_orig_prepare_inputs = cls.add_requests, cls.prepare_inputs

    def add_requests(self, scheduler_output):
        extra = self.__dict__.setdefault("_ls_extra", {})
        for r in scheduler_output.scheduled_new_reqs:
            sp = r.sampling_params
            extra[r.req_id] = (sp.extra_args or {}) if sp is not None else {}
        return cls._ls_orig_add_requests(self, scheduler_output)

    def prepare_inputs(self, *args, **kwargs):
        batch = cls._ls_orig_prepare_inputs(self, *args, **kwargs)
        self._ls_batch = batch
        return batch

    cls.add_requests, cls.prepare_inputs = add_requests, prepare_inputs


def _batch(runner, positions):
    """(req_ids, extra_args per request, request index per token, valid mask) of the current forward pass."""
    b = getattr(runner, "_ls_batch", None)
    if b is None:
        return None
    n_reqs = b.num_reqs
    qsl = b.query_start_loc[: n_reqs + 1]
    req_ids = list(b.req_ids[:n_reqs])
    extras = [getattr(runner, "_ls_extra", {}).get(r, {}) for r in req_ids]
    tok = torch.arange(positions.shape[0], device=positions.device)
    req = torch.searchsorted(qsl[1:].to(positions.device).long(), tok, right=True)
    valid = req < n_reqs
    return req_ids, extras, req.clamp(max=max(n_reqs - 1, 0)), valid


def _token_classes(runner, positions):
    """Per-token position class (-1: none) and position relative to the end of the prompt (as ``vllm_hooks``)."""
    n = positions.shape[0]
    cls = torch.full((n,), -1, dtype=torch.int8, device=positions.device)
    rel = torch.zeros(n, dtype=torch.long, device=positions.device)
    info = _batch(runner, positions)
    if info is None or not info[0]:
        return cls, rel, info
    _, extras, req, valid = info
    b = torch.tensor([e.get("ls_bounds", (-1, -1)) for e in extras], dtype=torch.long, device=positions.device)
    P, C = b[req, 0], b[req, 1]
    pos = positions.long()
    c = torch.where(pos < P, 0, torch.where((C < 0) | (pos < C), 1, 2))
    cls = torch.where(valid & (P >= 0), c, -1).to(torch.int8)
    return cls, pos - P, info


def is_inkling(model) -> bool:
    return type(model.model).__name__ == "Inkling2LoopModel"


def _unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm()


class _State:
    def __init__(self, lm, runner, specs):
        self.lm, self.runner = lm, runner
        dev = lm.embed_tokens.weight.device
        cfg = lm.config
        self.core = lm.loop_core
        self.passes = cfg.num_loop_passes
        self.n_layers = len(lm.layers)
        self.ablate, self.actadd, self.kick, self.capture = [], [], [], None
        for s in specs:
            k = s["kind"]
            if k == "ablate":
                self.ablate.append({**s, "dir": _unit(s["dir"]).to(dev)})
            elif k == "actadd":
                self.actadd.append({**s, "vec": torch.as_tensor(s["vec"], dtype=torch.float32).to(dev)})
            elif k == "kick":
                self.kick.append({**s, "vec": _unit(s["vec"]).to(dev), "median": None})
            elif k == "capture":
                n_sites = self.n_layers + (self.passes - 1) * len(self.core)
                self.capture = {"sum": torch.zeros(s["n_req"], s["n_windows"], n_sites, cfg.hidden_size,
                                                   dtype=torch.float32, device=dev),
                                "count": torch.zeros(s["n_req"], s["n_windows"], dtype=torch.float64, device=dev),
                                "n_windows": s["n_windows"]}
            else:
                raise ValueError(f"unsupported spec kind {k!r} for Inkling")
        self.fwd = None  # per-forward token info

    # ------------------------------------------------------------------ selection
    def unit_selected(self, loops, i, pass_idx):
        if loops is None:
            return True
        if i < self.core.start:
            return -1 in loops or -2 in loops
        if i >= self.core.stop:
            return -1 in loops or -3 in loops
        return pass_idx in loops

    def site_selected(self, loops, i, pass_idx):  # actadd / kick at a layer
        if loops is None:
            return True
        if i in self.core:
            return pass_idx in loops
        return -1 in loops or 0 in loops

    def visit_index(self, i, pass_idx):
        c = self.core
        if i < c.start:
            return i
        if i < c.stop:
            return c.start + pass_idx * len(c) + (i - c.start)
        return i + (self.passes - 1) * len(c)

    def touched(self, i, pass_idx):
        if self.capture is not None:
            return True
        if any(self.unit_selected(s.get("loops"), i, pass_idx) for s in self.ablate):
            return True
        return any(s["layer"] == i and self.site_selected(s.get("loops"), i, pass_idx)
                   for s in self.actadd + self.kick)

    # ------------------------------------------------------------------ token info
    def tokens(self, positions):
        """Position classes (per token) of the current forward, computed once per forward pass."""
        if self.fwd is not None and self.fwd[0] is positions:
            return self.fwd[1]
        cls, rel, reqinfo = _token_classes(self.runner, positions)
        info = {"cls": cls, "rel": rel, "reqinfo": reqinfo}
        self.fwd = (positions, info)
        return info

    def mask(self, positions, spec):
        info = self.tokens(positions)
        cls = info["cls"]
        pos = spec.get("positions")
        if pos is None:
            m = cls >= 0
        else:
            m = torch.zeros_like(cls, dtype=torch.bool)
            for c in pos:
                m |= cls == CLASSES[c]
        return m

    # ------------------------------------------------------------------ edits
    def project(self, x, positions, i, pass_idx):
        """Ablate every selected direction from ``x`` [..., T, D] in place (masked tokens only); returns x."""
        for s in self.ablate:
            if not self.unit_selected(s.get("loops"), i, pass_idx):
                continue
            m = self.mask(positions, s).to(torch.float32)  # [T]
            u = s["dir"]
            coef = (x.float() @ u) * m  # [..., T]
            x.sub_((coef.unsqueeze(-1) * u).to(x.dtype))
        return x

    def add_site(self, partial, positions, i, pass_idx):
        for s in self.actadd:
            if s["layer"] == i and self.site_selected(s.get("loops"), i, pass_idx):
                m = self.mask(positions, s).to(torch.float32).unsqueeze(-1)
                partial.add_((m * (s["coeff"] * s["vec"])).to(partial.dtype))
        for s in self.kick:
            if s["layer"] == i and self.site_selected(s.get("loops"), i, pass_idx):
                norms = partial.float().norm(dim=-1)
                if s["median"] is None and partial.shape[0] >= 64:
                    s["median"] = float(norms.median())
                m = self.mask(positions, s) & (positions != 0)
                if s["median"] is not None:
                    m &= norms <= s.get("sink_ratio", 10.0) * s["median"]
                delta = (s["eps"] * norms * m.float()).unsqueeze(-1) * s["vec"]
                partial.add_(delta.to(partial.dtype))

    def record(self, partial, positions, i, pass_idx):
        cap = self.capture
        if cap is None:
            return
        info = self.tokens(positions)
        if "cap" not in info:
            info["cap"] = self._capture_index(positions, info)
        idx = info["cap"]
        if idx is None:
            return
        k, w_mask = idx  # [T] request slot (-1: none), [W, T] window membership
        v = self.visit_index(i, pass_idx)
        h = partial.float()
        valid = k >= 0
        for w in range(cap["n_windows"]):
            sel = valid & w_mask[w]
            if sel.any():
                cap["sum"][:, w, v].index_add_(0, k[sel], h[sel])
                if v == 0:
                    cap["count"][:, w].index_add_(0, k[sel], torch.ones_like(k[sel], dtype=torch.float64))

    def _capture_index(self, positions, info):
        reqinfo = info["reqinfo"]
        if reqinfo is None:
            return None
        _, extras, req, valid = reqinfo
        W = self.capture["n_windows"]
        slots, bounds = [], []
        for extra in extras:
            cap = extra.get("ls_cap")
            if cap is None:
                slots.append(-1)
                bounds.append([(0, 0)] * W)
            else:
                slots.append(int(cap[0]))
                bounds.append([tuple(b) for b in cap[1]])
        dev = positions.device
        slot = torch.tensor(slots, dtype=torch.long, device=dev)[req]
        b = torch.tensor(bounds, dtype=torch.long, device=dev)[req]  # [T, W, 2]
        pos = positions.long().unsqueeze(-1)
        w_mask = ((pos >= b[..., 0]) & (pos < b[..., 1])).T  # [W, T]
        k = torch.where(valid, slot, torch.full_like(slot, -1))
        return k, w_mask


def _visit(self, st, i, pass_idx, stack, num_blocks, inputs, bank, next_query, next_norm, positions, tau):
    """Unfused ``Inkling2LoopModel._run_layer`` (TP 1) with the interventions of ``st``."""
    from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
    from vllm.models.inkling2.nvidia.ops.norm import rms_norm

    layer = self.layers[i]
    owner = layer.conv_states[pass_idx]
    core = self.loop_core
    partial = stack[num_blocks]
    restart = i == core.start and pass_idx > 0

    if num_blocks == 0 and i == 0:
        st.project(stack[0], positions, i, pass_idx)  # the embedding (prelude selected)
    st.add_site(partial, positions, i, pass_idx)
    st.record(partial, positions, i, pass_idx)

    # Attention inputs. Without ablation here, the mixes are normalized in the same fused kernel as vLLM's path
    # (bit-identical inputs: the NVFP4 activation quantization of the experts turns any rounding difference into a
    # visible one); with ablation, the fp32 mixes are edited before their norm.
    ablate_here = any(st.unit_selected(s.get("loops"), i, pass_idx) for s in st.ablate)
    n_in = layer.num_attn_inputs
    norm = layer.attn_norm
    messages = None
    if bank is not None:
        messages = attn_res(bank[:1], bank[1], None, self.message_queries[i - core.start], None, 0.0, True)
        st.project(messages, positions, i, pass_idx)
    if num_blocks == 0:
        if ablate_here:
            dests = partial.float().unsqueeze(0).expand(n_in, -1, -1).contiguous()
            st.project(dests, positions, i, pass_idx)
            attn_inputs = [rms_norm(dests[j], norm.weight[j], norm.eps, torch.bfloat16) for j in range(n_in)]
        else:
            attn_inputs = [norm(partial, j) for j in range(n_in)]
    else:
        query = self.attn_res.attn_queries[i - 1]
        if messages is not None or ablate_here:
            dests = attn_res(stack[:num_blocks], partial, None, query, None, 0.0, True)
            st.project(dests, positions, i, pass_idx)
            if messages is not None:
                d3, w3 = dests, norm.weight
                if layer.is_local:
                    d3, w3 = dests[:1].expand(3, -1, -1), norm.weight.expand(3, -1)
                attn_inputs = layer.blend(slice(0, 3), d3, messages[:3].contiguous(), w3, norm.eps)
            else:
                attn_inputs = [rms_norm(dests[j], norm.weight[j], norm.eps, torch.bfloat16)
                               for j in range(dests.shape[0])]
        else:
            attn_inputs = attn_res(stack[:num_blocks], partial, None, query, norm.weight, norm.eps, True)
    attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)
    attn_out = layer.attn_sconv(attn_partial, positions, owner=owner).contiguous()
    st.project(attn_out, positions, i, pass_idx)

    # MLP input: the blocks with (partial +) attn_out
    first = i % self.block_size == 0
    mq = self.attn_res.mlp_queries[i]
    mnorm = layer.mlp_norm
    src = (stack[: num_blocks + 1], attn_out, None) if first else (stack[:num_blocks], partial, attn_out)
    if messages is not None or ablate_here:
        mlp_dests = attn_res(*src, mq, None, 0.0, True)
        st.project(mlp_dests, positions, i, pass_idx)
        if messages is not None:
            mlp_in = layer.blend(slice(3, 4), mlp_dests, messages[3:].contiguous(), mnorm.weight, mnorm.eps)
        else:
            mlp_in = [rms_norm(mlp_dests[0], mnorm.weight[0], mnorm.eps, torch.bfloat16)]
    else:
        mlp_in = attn_res(*src, mq, mnorm.weight, mnorm.eps, True)

    residual = None if first else partial
    if residual is None:
        num_blocks += 1
    layer.mlp_sconv(layer.mlp(mlp_in[0], pass_idx=pass_idx), positions, addend=attn_out, residual=residual,
                    out=stack[num_blocks], owner=owner)
    st.project(stack[num_blocks], positions, i, pass_idx)

    last = i == self.config.num_hidden_layers - 1
    if last and st.ablate:
        mix = attn_res(stack[:num_blocks], stack[num_blocks], None, next_query, None, 0.0, True)
        st.project(mix, positions, i, pass_idx)
        outputs = rms_norm(mix, next_norm.weight[0], next_norm.eps, torch.bfloat16) if next_norm is not None else mix
        return num_blocks, outputs
    outputs = attn_res(stack[:num_blocks], stack[num_blocks], None, next_query,
                       None if next_norm is None else next_norm.weight, 0.0 if next_norm is None else next_norm.eps,
                       True)
    return num_blocks, outputs


def _patched_run_layer(self, i, pass_idx, stack, num_blocks, inputs, bank, next_query, next_norm, positions, tau,
                       plan):
    st = getattr(self, "_ls_state", None)
    if st is None or not st.touched(i, pass_idx):
        return self._ls_orig_run_layer(i, pass_idx, stack, num_blocks, inputs, bank, next_query, next_norm,
                                       positions, tau, plan)
    return _visit(self, st, i, pass_idx, stack, num_blocks, inputs, bank, next_query, next_norm, positions, tau)


def _patch(lm):
    cls = type(lm)
    if not hasattr(cls, "_ls_orig_run_layer"):
        cls._ls_orig_run_layer = cls._run_layer
        cls._run_layer = _patched_run_layer


def clear(model):
    model.model._ls_state = None
    return True


def install(model, specs):
    """Remove previously installed specs, then install ``specs`` (call through ``LLM.apply_model``)."""
    lm = model.model
    _patch(lm)
    lm._ls_state = None
    if specs:
        lm._ls_state = _State(lm, _model_runner(), specs)
    return True


def fetch_capture(model, reset=True):
    """(sums [R, W, S, D] fp32, counts [R, W]) on CPU; with ``reset``, zero them."""
    st = model.model._ls_state
    cap = st.capture
    out = (cap["sum"].cpu(), cap["count"].cpu())
    if reset:
        cap["sum"].zero_()
        cap["count"].zero_()
    return out
