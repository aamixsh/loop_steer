"""Interventions and activation capture inside vLLM's Inkling 2 Mini (``Inkling2LoopModel``, vLLM PR 69028), compatible
with vLLM's (breakable) CUDA graphs.

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
  {"kind": "kick", "vec": [D] | "vecs": [K, D], "layer": l, "eps": e, "loops": ..., "sink_ratio": r}
      resid_pre(l) += eps * |resid_pre| * unit(vec) per token except attention sinks: position 0, and tokens with
      |resid_pre| > r * the site's median stream norm (``directions/stream_norms.pt``; the eager hooks of
      ``vllm_hooks`` use the median of the first forward pass instead). With ``vecs``, a request with
      ``extra_args["ls_vec"] = k`` gets row k; requests without an index are not kicked (run without prefix caching).
  {"kind": "capture", "n_req": R, "n_windows": W[, "norms": True]}   (eager engines only: ``enforce_eager=True``)
      accumulate, per request with ``extra_args["ls_cap"] = (k, [(lo, hi)] * W)`` (k < R), the sum of resid_pre over
      the tokens at positions ``lo <= pos < hi`` of each window at every site, and the token counts. Run without prefix
      caching (cached tokens are not computed) and fetch with ``fetch_capture``. With ``"norms": True`` it also keeps
      the per-token norms of the window-0 tokens at every site (``fetch_capture_norms``: median per site).
  {"kind": "final"}   (eager engines only)
      keep the final normalized hidden state (the LM head's input) of every request with
      ``extra_args["ls_final"] = (k, lo, hi)`` at positions ``lo <= pos < hi``; ``fetch_final`` returns
      ``{k: [hi - lo, D]}`` (``scripts/kick_kl.py``: logits through the model's own ``compute_logits``).
  {"kind": "coda_from", "pass": p}   (eager engines only)
      run the coda on the blocks of pass ``p`` (0-based) instead of the last pass: the readout "after pass p + 1"
      (later passes still run, but nothing reads them). Running passes 1..p+1 and then the coda is the same
      computation as a (p+1)-pass run (``scripts/loop_readout.py``).
  {"kind": "msg", "mode": "ablate" | "zero", "passes": [p, ...], "dir": [D]}   (eager engines only)
      change the cross-pass messages read by the core layers of the given passes (0-based; messages exist from
      pass 1 on): ``ablate`` projects ``dir`` out of them, ``zero`` replaces them by 0 (the pass then gets nothing
      from the previous one beyond what its gate makes of a zero message) (``scripts/inkling_rebuild.py``).
  {"kind": "norms"}   (eager engines only)
      keep the per-token norm of resid_pre at every site (every token but position 0); ``fetch_norms`` returns the
      median per site (``scripts/stream_norms.py``, for norm-matched activation addition).

Position classes (``positions``): as ``loop_steer.vllm_hooks`` (``extra_args["ls_bounds"]``).

**CUDA graphs.** vLLM captures the forward pass once per batch size and replays it, so nothing an intervention does may
depend on Python at replay time. Every layer visit therefore runs the same instrumented code whatever is installed: the
edits are tensor ops on persistent buffers (directions, per-visit on/off weights and coefficients, per-token class and
kick-slot indices), which ``install`` overwrites in place; an edit that is off multiplies by 0, which leaves the stream
bit-identical. Inputs are normalized by vLLM's fused mix-and-norm kernel unless an ablation is on at the visit (then the
fp32 mix is edited before an unfused RMSNorm, ~0.13 nat/token of rounding noise through the NVFP4 experts); the
selection between the two is a ``torch.where`` on the visit's ablation weight. The per-token buffers are filled from the
scheduled batch before each forward (``GPUModelRunner.prepare_inputs``). ``register()`` must run in the engine's
process before the engine is built (``loop_steer.models.prepare_vllm`` does it, with V1 multiprocessing off).
"""

import torch

from loop_steer.vllm_hooks import CLASSES

N_ABL, N_ADD, N_KICK = 2, 4, 4  # spec slots per kind
MAX_KICK_ROWS = 1024  # per-request kick directions (MATH-500: 500); one more zero row for requests without an index


# ---------------------------------------------------------------------- vLLM model runner (v2, ``v1/worker/gpu``)
def _model_runner():
    """The worker's GPU model runner, found up the call stack of ``Worker.apply_model``."""
    import sys

    frame = sys._getframe(1)
    while frame is not None:
        runner = getattr(frame.f_locals.get("self"), "model_runner", None)
        if runner is not None and hasattr(runner, "req_states") and hasattr(runner, "prepare_inputs"):
            return runner
        frame = frame.f_back
    raise RuntimeError("no vLLM model runner up the call stack: call install() through LLM.apply_model")


def _token_info(runner, positions, n_tokens):
    """(request index per token, valid mask, extra_args per request) of the batch being prepared."""
    b = runner._ls_batch
    n_reqs = b.num_reqs
    qsl = b.query_start_loc[: n_reqs + 1]
    extras = [runner._ls_extra.get(r, {}) for r in b.req_ids[:n_reqs]]
    tok = torch.arange(n_tokens, device=positions.device)
    req = torch.searchsorted(qsl[1:].long(), tok, right=True)
    valid = req < n_reqs
    return req.clamp(max=max(n_reqs - 1, 0)), valid, extras


def _fill_token_buffers(runner):
    """Per-token class (cls + 1; 0 = none) and kick slot of the batch just prepared, into the model's buffers."""
    lm = runner._ls_lm
    g = getattr(lm, "_lsg", None)
    if g is None or not g.need_tokens:
        return
    b = runner._ls_batch
    n = b.num_tokens_after_padding if hasattr(b, "num_tokens_after_padding") else b.num_tokens
    positions = b.positions[:n]
    req, valid, extras = _token_info(runner, positions, n)
    dev = positions.device
    bounds = torch.tensor([e.get("ls_bounds", (-1, -1)) for e in extras] or [(-1, -1)], dtype=torch.long, device=dev)
    P, C = bounds[req, 0], bounds[req, 1]
    pos = positions.long()
    c = torch.where(pos < P, 0, torch.where((C < 0) | (pos < C), 1, 2))
    g.cls[:n] = torch.where(valid & (P >= 0), c + 1, 0)
    slots = torch.tensor([int(e["ls_vec"]) if "ls_vec" in e else MAX_KICK_ROWS for e in extras] or [MAX_KICK_ROWS],
                         dtype=torch.long, device=dev).clamp(max=MAX_KICK_ROWS)
    g.slot[:n] = torch.where(valid, slots[req], MAX_KICK_ROWS)


def _patch_runner(cls):
    """Keep each request's ``extra_args`` and the batch being run (the v2 runner keeps neither in reach of the
    forward pass), and fill the per-token buffers before every forward."""
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
        if getattr(self, "_ls_lm", None) is not None:
            _fill_token_buffers(self)
        return batch

    cls.add_requests, cls.prepare_inputs = add_requests, prepare_inputs


# ---------------------------------------------------------------------- buffers
class _Buffers:
    """Persistent intervention state (fixed addresses: CUDA graphs read these)."""

    def __init__(self, lm, max_tokens):
        cfg = lm.config
        dev = lm.embed_tokens.weight.device
        D = cfg.hidden_size
        core = lm.loop_core
        self.passes = cfg.num_loop_passes
        self.core = core
        self.n_layers = len(lm.layers)
        V = self.n_layers + (self.passes - 1) * len(core)
        f32 = dict(dtype=torch.float32, device=dev)
        self.abl_dir = torch.zeros(N_ABL, D, **f32)
        self.abl_on = torch.zeros(N_ABL, V, **f32)  # 1 where the spec ablates at this visit
        self.abl_any = torch.zeros(V, dtype=torch.bool, device=dev)  # some ablation on at this visit
        self.abl_cls = torch.ones(N_ABL, 4, **f32)  # token weight by class + 1 (0: no class)
        self.add_vec = torch.zeros(N_ADD, D, **f32)
        self.add_coef = torch.zeros(N_ADD, V, **f32)
        self.add_cls = torch.ones(N_ADD, 4, **f32)
        self.kick_tab = torch.zeros(N_KICK, MAX_KICK_ROWS + 1, D, **f32)  # unit rows; last row 0
        self.kick_eps = torch.zeros(N_KICK, V, **f32)
        self.kick_cls = torch.ones(N_KICK, 4, **f32)
        self.kick_thr = torch.full((V,), float("inf"), **f32)  # sink threshold on |resid_pre|
        self.cls = torch.zeros(max_tokens, dtype=torch.long, device=dev)
        self.slot = torch.zeros(max_tokens, dtype=torch.long, device=dev)
        self.need_tokens = False
        self.capture = None  # eager-only analysis state
        self.norms = None
        self.final = None
        self.coda_from, self.saved_blocks = None, None
        self.msg = []

    def visit_index(self, i, pass_idx):
        c = self.core
        if i < c.start:
            return i
        if i < c.stop:
            return c.start + pass_idx * len(c) + (i - c.start)
        return i + (self.passes - 1) * len(c)

    def unit_of(self, i, pass_idx):
        """Selection test of a visit for ``loops`` (see module doc)."""
        def sel(loops):
            if loops is None:
                return True
            if i < self.core.start:
                return -1 in loops or -2 in loops
            if i >= self.core.stop:
                return -1 in loops or -3 in loops
            return pass_idx in loops
        return sel

    def visits(self):
        c = self.core
        out = [(i, 0) for i in range(c.start)] + [(i, p) for p in range(self.passes) for i in c]
        return out + [(i, 0) for i in range(c.stop, self.n_layers)]

    def reset(self):
        for t in (self.abl_dir, self.abl_on, self.add_vec, self.add_coef, self.kick_tab, self.kick_eps):
            t.zero_()
        self.abl_any.zero_()
        for t in (self.abl_cls, self.add_cls, self.kick_cls):
            t.fill_(1.0)
        self.kick_thr.fill_(float("inf"))
        self.cls.zero_()
        self.slot.fill_(0)
        self.need_tokens = False
        self.capture = self.norms = self.final = None
        self.coda_from, self.saved_blocks = None, None
        self.msg = []

    # ------------------------------------------------------------------ token weights (graph-safe)
    def w(self, table, k, T):
        """[T] per-token weight of spec slot k (class gating)."""
        return table[k][self.cls[:T]]

    def project(self, x, T, v):
        """Ablate every slot's direction from ``x`` [..., T, D] in place, weighted by its on-weight at visit v and
        the tokens' class weights (exactly 0 where off)."""
        for k in range(N_ABL):
            u = self.abl_dir[k]
            wt = self.abl_on[k, v] * self.w(self.abl_cls, k, T)  # [T]
            coef = (x.float() @ u) * wt
            x.sub_((coef.unsqueeze(-1) * u).to(x.dtype))
        return x

    def add_site(self, partial, positions, v):
        T = partial.shape[0]
        for k in range(N_ADD):
            wt = (self.add_coef[k, v] * self.w(self.add_cls, k, T)).unsqueeze(-1)
            partial.add_((wt * self.add_vec[k]).to(partial.dtype))
        norms = partial.float().norm(dim=-1)
        ok = (positions != 0) & (norms <= self.kick_thr[v])
        for k in range(N_KICK):
            wt = (self.kick_eps[k, v] * self.w(self.kick_cls, k, T) * ok * norms).unsqueeze(-1)
            partial.add_((wt * self.kick_tab[k][self.slot[:T]]).to(partial.dtype))

    def record(self, partial, positions, v, runner):  # eager only
        if self.norms is not None:
            self.norms.setdefault(v, []).append(partial[positions != 0].float().norm(dim=-1).cpu())
        cap = self.capture
        if cap is None:
            return
        T = partial.shape[0]
        if cap.get("fwd") is not positions:
            req, valid, extras = _token_info(runner, positions, T)
            W = cap["n_windows"]
            slots = [int(e["ls_cap"][0]) if "ls_cap" in e else -1 for e in extras]
            bounds = [[tuple(b) for b in e["ls_cap"][1]] if "ls_cap" in e else [(0, 0)] * W for e in extras]
            dev = positions.device
            slot = torch.tensor(slots or [-1], dtype=torch.long, device=dev)[req]
            b = torch.tensor(bounds or [[(0, 0)] * W], dtype=torch.long, device=dev)[req]  # [T, W, 2]
            pos = positions.long().unsqueeze(-1)
            cap["idx"] = (torch.where(valid, slot, -1), ((pos >= b[..., 0]) & (pos < b[..., 1])).T)
            cap["fwd"] = positions
        k, w_mask = cap["idx"]
        h = partial.float()
        for w in range(cap["n_windows"]):
            sel = (k >= 0) & w_mask[w]
            if w == 0 and cap.get("norm_list") is not None and sel.any():
                cap["norm_list"].setdefault(v, []).append(h[sel].norm(dim=-1).cpu())
            if sel.any():
                cap["sum"][:, w, v].index_add_(0, k[sel], h[sel])
                if v == 0:
                    cap["count"][:, w].index_add_(0, k[sel], torch.ones_like(k[sel], dtype=torch.float64))


def _buffers(lm):
    g = getattr(lm, "_lsg", None)
    if g is None:
        from vllm.config import get_current_vllm_config

        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        g = lm._lsg = _Buffers(lm, max_tokens + 64)
    return g


# ---------------------------------------------------------------------- the layer visit
def _visit(self, i, pass_idx, stack, num_blocks, bank, next_query, next_norm, positions, tau):
    """``Inkling2LoopModel._run_layer`` (TP 1) with the interventions of ``self._lsg``: short conv and AttnRes mix
    unfused (the mix-and-norm fused, as vLLM's ``conv_mix``, unless an ablation is on). Returns the number of
    completed blocks and, at the last visit only, the final normalized output (the other visits' outputs would be
    the next visit's inputs, which every visit recomputes from the stack)."""
    from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
    from vllm.models.inkling2.nvidia.ops.norm import rms_norm

    g = _buffers(self)
    layer = self.layers[i]
    owner = layer.conv_states[pass_idx]
    core = self.loop_core
    partial = stack[num_blocks]
    T = partial.shape[0]
    v = g.visit_index(i, pass_idx)
    on = g.abl_any[v]  # 0-dim bool: some ablation at this visit

    if g.coda_from is not None and i == core.stop and g.saved_blocks is not None:  # eager only (Python branch)
        nb_core = g.saved_blocks.shape[0]
        stack[num_blocks - nb_core + 1: num_blocks + 1].copy_(g.saved_blocks)
    if num_blocks == 0 and i == 0:
        g.project(stack[0], T, v)  # the embedding (prelude selected)
    g.add_site(partial, positions, v)
    g.record(partial, positions, v, getattr(self, "_ls_runner", None))

    norm = layer.attn_norm
    n_in = layer.num_attn_inputs
    messages = None
    if bank is not None:
        messages = attn_res(bank[:1], bank[1], None, self.message_queries[i - core.start], None, 0.0, True)
        g.project(messages, T, v)
        for m in g.msg:  # eager only (Python branch)
            if pass_idx in m["passes"]:
                if m["mode"] == "zero":
                    messages.zero_()
                else:
                    u = m["dir"]
                    messages.sub_((messages @ u).unsqueeze(-1) * u)
    if num_blocks == 0:
        fused = [norm(partial, j) for j in range(n_in)]
        dests = partial.float().unsqueeze(0).expand(n_in, -1, -1).contiguous()
        g.project(dests, T, v)
        attn_inputs = [torch.where(on, rms_norm(dests[j], norm.weight[j], norm.eps, torch.bfloat16), fused[j])
                       for j in range(n_in)]
    else:
        query = self.attn_res.attn_queries[i - 1]
        dests = attn_res(stack[:num_blocks], partial, None, query, None, 0.0, True)
        g.project(dests, T, v)
        if messages is not None:
            d3, w3 = dests, norm.weight
            if layer.is_local:
                d3, w3 = dests[:1].expand(3, -1, -1), norm.weight.expand(3, -1)
            attn_inputs = layer.blend(slice(0, 3), d3, messages[:3].contiguous(), w3, norm.eps)
        else:
            fused = attn_res(stack[:num_blocks], partial, None, query, norm.weight, norm.eps, True)
            unfused = torch.stack([rms_norm(dests[j], norm.weight[j], norm.eps, torch.bfloat16)
                                   for j in range(dests.shape[0])])
            attn_inputs = torch.where(on, unfused, fused)
    attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)
    attn_out = layer.attn_sconv(attn_partial, positions, owner=owner).contiguous()
    g.project(attn_out, T, v)

    first = i % self.block_size == 0
    mq = self.attn_res.mlp_queries[i]
    mnorm = layer.mlp_norm
    src = (stack[: num_blocks + 1], attn_out, None) if first else (stack[:num_blocks], partial, attn_out)
    mlp_dests = attn_res(*src, mq, None, 0.0, True)
    g.project(mlp_dests, T, v)
    if messages is not None:
        mlp_in = layer.blend(slice(3, 4), mlp_dests, messages[3:].contiguous(), mnorm.weight, mnorm.eps)
    else:
        fused = attn_res(*src, mq, mnorm.weight, mnorm.eps, True)
        mlp_in = torch.where(on, rms_norm(mlp_dests[0], mnorm.weight[0], mnorm.eps, torch.bfloat16)[None], fused)

    residual = None if first else partial
    if residual is None:
        num_blocks += 1
    layer.mlp_sconv(layer.mlp(mlp_in[0], pass_idx=pass_idx), positions, addend=attn_out, residual=residual,
                    out=stack[num_blocks], owner=owner)
    g.project(stack[num_blocks], T, v)
    if g.coda_from is not None and i == core.stop - 1 and pass_idx == g.coda_from:  # this pass's blocks for the coda
        nb_core = -(-len(core) // self.block_size)
        g.saved_blocks = stack[num_blocks - nb_core + 1: num_blocks + 1].clone()

    if i != self.config.num_hidden_layers - 1:
        return num_blocks, None
    mix = attn_res(stack[:num_blocks], stack[num_blocks], None, next_query, None, 0.0, True)
    g.project(mix, T, v)
    fused = attn_res(stack[:num_blocks], stack[num_blocks], None, next_query, next_norm.weight, next_norm.eps, True)
    out = torch.where(on, rms_norm(mix, next_norm.weight[0], next_norm.eps, torch.bfloat16), fused)
    if g.final is not None:  # eager only
        _record_final(g, out[0], positions, getattr(self, "_ls_runner", None))
    return num_blocks, out


def _record_final(g, h, positions, runner):
    req, valid, extras = _token_info(runner, positions, h.shape[0])
    for r, e in enumerate(extras):
        if "ls_final" not in e:
            continue
        k, lo, hi = e["ls_final"]
        sel = valid & (req == r) & (positions >= lo) & (positions < hi)
        if sel.any():
            g.final.setdefault(int(k), []).append((positions[sel].clone(), h[sel].clone()))


def _run_layer(self, i, pass_idx, stack, num_blocks, inputs, bank, next_query, next_norm, positions, tau, plan):
    return _visit(self, i, pass_idx, stack, num_blocks, bank, next_query, next_norm, positions, tau)


def register():
    """Patch vLLM's Inkling loop model and GPU model runner (in the engine's process, before the engine is built)."""
    from vllm.models.inkling2.nvidia.loop import Inkling2LoopModel
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if not getattr(Inkling2LoopModel, "_ls_patched", False):
        Inkling2LoopModel._ls_orig_run_layer = Inkling2LoopModel._run_layer
        Inkling2LoopModel._run_layer = _run_layer
        Inkling2LoopModel._ls_patched = True
    _patch_runner(GPUModelRunner)


# ---------------------------------------------------------------------- install
def is_inkling(model) -> bool:
    return type(model.model).__name__ == "Inkling2LoopModel"


def _cls_row(positions):
    row = torch.ones(4)
    if positions is not None:
        row.zero_()
        for c in positions:
            row[CLASSES[c] + 1] = 1.0
    return row


def _sink_thresholds(g, ratio, model_name):
    """ratio * the median stream norm of every site (directions/stream_norms[_loops<N>].pt of the served model at
    the engine's pass count), as [V]; None without that file."""
    from loop_steer.models import INKLING, INKLING_ID
    from loop_steer.paths import run_dir

    name = next((m for m in INKLING if m == model_name or str(model_name).endswith(m.split("/")[-1])), INKLING_ID)
    rd = run_dir(name) / "directions"
    trained = 3
    path = rd / ("stream_norms.pt" if g.passes == trained else f"stream_norms_loops{g.passes}.pt")
    if not path.exists():
        return None
    d = torch.load(path, weights_only=False)
    med = {tuple(s): float(x) for s, x in zip(d["sites"], d["norms"])}
    out = torch.full((len(g.visits()),), float("inf"))
    for v, (i, p) in enumerate(g.visits()):
        site = (p, i) if i in g.core else (0, i)
        if site in med:
            out[v] = ratio * med[site]
    return out


def clear(model):
    g = getattr(model.model, "_lsg", None)
    if g is not None:
        with torch.inference_mode():  # the buffers were made during vLLM's (inference-mode) profile run
            g.reset()
    return True


def install(model, specs):
    """Overwrite the intervention buffers with ``specs`` (call through ``LLM.apply_model``)."""
    with torch.inference_mode():  # the buffers were made during vLLM's (inference-mode) profile run
        return _install(model, specs)


def _actadd_slot(g, s, visits):
    """(unit direction, per-visit coefficient) of an actadd spec: coeff * |vec| at the spec's sites, else 0."""
    sel = s.get("loops")

    def at_site(i, p):
        if i != s["layer"]:
            return False
        if sel is None:
            return True
        return p in sel if i in g.core else (-1 in sel or 0 in sel)

    vec = torch.as_tensor(s["vec"], dtype=torch.float32)
    scale = float(vec.norm())
    unit = vec / scale if scale > 0 else vec
    coef = torch.tensor([float(s["coeff"]) * scale if at_site(i, p) else 0.0 for i, p in visits])
    return unit.to(g.add_vec.device), coef.to(g.add_coef.device)


def _merge_actadd(g, s, visits, used):
    """Fold an actadd spec into a filled slot with the same direction and token-class row (summing coefficients).
    intervene.py emits one spec per pass for relative strengths, so an all-pass actadd on a 5-pass run needs this."""
    unit, coef = _actadd_slot(g, s, visits)
    row = _cls_row(s.get("positions")).to(g.add_cls.device)
    for j in range(used):
        if torch.allclose(g.add_vec[j], unit, atol=1e-6) and torch.equal(g.add_cls[j], row):
            g.add_coef[j] += coef
            if s.get("positions") is not None:
                g.need_tokens = True
            return True
    return False


def _install(model, specs):
    lm = model.model
    if not getattr(type(lm), "_ls_patched", False):
        raise RuntimeError("vllm_inkling.register() was not called before the engine was built")
    runner = _model_runner()
    runner._ls_lm = lm
    lm._ls_runner = runner
    g = _buffers(lm)
    g.reset()
    visits = g.visits()
    n = {"ablate": 0, "actadd": 0, "kick": 0}
    sink = None
    for s in specs or []:
        k = s["kind"]
        for opt in ("ref", "decay", "layers"):
            if s.get(opt) is not None:
                raise ValueError(f"{k} spec option {opt!r} is not implemented for Inkling")
        if k == "capture":
            g.capture = {"sum": torch.zeros(s["n_req"], s["n_windows"], len(visits), g.abl_dir.shape[1],
                                            dtype=torch.float32, device=g.abl_dir.device),
                         "count": torch.zeros(s["n_req"], s["n_windows"], dtype=torch.float64, device=g.abl_dir.device),
                         "n_windows": s["n_windows"], "norm_list": {} if s.get("norms") else None}
            continue
        if k == "norms":
            g.norms = {}
            continue
        if k == "final":
            g.final = {}
            continue
        if k == "coda_from":
            g.coda_from = int(s["pass"])
            continue
        if k == "msg":
            u = torch.as_tensor(s.get("dir", torch.ones(1)), dtype=torch.float32)
            g.msg.append({"mode": s["mode"], "passes": list(s["passes"]),
                          "dir": (u / u.norm()).to(g.abl_dir.device) if s["mode"] == "ablate" else None})
            continue
        if k not in n:
            raise ValueError(f"unsupported spec kind {k!r} for Inkling")
        if k == "actadd" and _merge_actadd(g, s, visits, n["actadd"]):
            continue  # same direction and token classes as a filled slot (e.g. one spec per pass): coefs summed
        j = n[k]
        n[k] += 1
        if j >= {"ablate": N_ABL, "actadd": N_ADD, "kick": N_KICK}[k]:
            raise ValueError(f"too many {k} specs")
        sel = s.get("loops")
        if s.get("positions") is not None:
            g.need_tokens = True
        row = _cls_row(s.get("positions"))
        if k == "ablate":
            u = torch.as_tensor(s["dir"], dtype=torch.float32)
            g.abl_dir[j] = (u / u.norm()).to(g.abl_dir.device)
            on = torch.tensor([1.0 if g.unit_of(i, p)(sel) else 0.0 for i, p in visits])
            g.abl_on[j] = on.to(g.abl_on.device)
            g.abl_cls[j] = row.to(g.abl_cls.device)
        else:
            def at_site(i, p):
                if i != s["layer"]:
                    return False
                if sel is None:
                    return True
                return p in sel if i in g.core else (-1 in sel or 0 in sel)
            if k == "actadd":
                unit, coef = _actadd_slot(g, s, visits)
                g.add_vec[j], g.add_coef[j] = unit, coef
                g.add_cls[j] = row.to(g.add_cls.device)
            else:
                if "vecs" in s and s["vecs"] is not None:
                    vecs = torch.as_tensor(s["vecs"], dtype=torch.float32)
                    if len(vecs) > MAX_KICK_ROWS:
                        raise ValueError(f"at most {MAX_KICK_ROWS} per-request kick directions")
                    g.kick_tab[j, : len(vecs)] = (vecs / vecs.norm(dim=-1, keepdim=True)).to(g.kick_tab.device)
                    g.need_tokens = True
                else:
                    u = torch.as_tensor(s["vec"], dtype=torch.float32)
                    g.kick_tab[j, :MAX_KICK_ROWS] = (u / u.norm()).to(g.kick_tab.device)
                g.kick_eps[j] = torch.tensor([float(s["eps"]) if at_site(i, p) else 0.0 for i, p in visits]
                                             ).to(g.kick_eps.device)
                g.kick_cls[j] = row.to(g.kick_cls.device)
                ratio = s.get("sink_ratio", 10.0)
                if sink is not None and sink != ratio:
                    raise ValueError("kick specs with different sink ratios")
                sink = ratio
    eager_only = (g.capture, g.norms, g.final, g.coda_from, g.msg or None)
    if any(x is not None for x in eager_only) and not runner.model_config.enforce_eager:
        raise RuntimeError("capture, norms, final and coda_from run Python at every forward: build the engine with enforce_eager=True")
    g.abl_any.copy_((g.abl_on > 0).any(0))
    if sink is not None:
        thr = _sink_thresholds(g, sink, runner.model_config.model)
        if thr is not None:
            g.kick_thr.copy_(thr.to(g.kick_thr.device))
    if not g.need_tokens:  # every token in every class slot weighs 1, and fixed kick rows are all equal
        g.cls.zero_()
        g.slot.zero_()
    return True


def fetch_norms(model):
    """{visit index: median resid_pre norm} over every token seen since ``install``."""
    g = model.model._lsg
    return {v: float(torch.cat(x).median()) for v, x in g.norms.items()}


def fetch_final(model, reset=True):
    """{k: final hidden states [n, D] (GPU) in position order} since ``install``; with ``reset``, forget them."""
    g = model.model._lsg
    out = {}
    for k, parts in g.final.items():
        pos = torch.cat([p for p, _ in parts])
        out[k] = torch.cat([h for _, h in parts])[pos.argsort()]
    if reset:
        g.final = {}
    return out


def fetch_capture_norms(model):
    """{visit index: median norm of the window-0 tokens} of a capture with ``"norms": True``."""
    nl = model.model._lsg.capture["norm_list"]
    return {v: float(torch.cat(x).median()) for v, x in nl.items()}


def fetch_capture(model, reset=True):
    """(sums [R, W, S, D] fp32, counts [R, W]) on CPU; with ``reset``, zero them."""
    cap = model.model._lsg.capture
    out = (cap["sum"].cpu(), cap["count"].cpu())
    if reset:
        with torch.inference_mode():  # the buffers are inference tensors (made in install)
            cap["sum"].zero_()
            cap["count"].zero_()
    return out
