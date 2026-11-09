"""Shared pieces of the self-correction experiments (HF backend).

A model is described by *units*: the loops of a looped model, or ``blocks`` equal blocks of layers of a standard
(dense / MoE) model, so dense models can be compared with looped ones at the same position "loop k". Sites are keyed
``(loop, layer)`` as everywhere else (dense models: loop 0). ``Structure`` maps every site to its unit, its index in
execution order and its relative depth, and names the unit boundaries (the input of the first layer of each unit;
after the last unit, the final stream that enters the unembedding).

Used by ``scripts/perturb_response.py``, ``scripts/loop_patch.py``, ``scripts/caa_loops.py``,
``scripts/persona_vectors.py`` and ``scripts/neuron_push.py``.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from loop_steer.hooks import _layer_input, _loop, _replace_layer_input, decoder_layers, hooked
from loop_steer.models import model_sites, n_loops

FINAL = ("final", -1)  # key of the final stream (input of the unembedding)


@dataclass
class Structure:
    sites: list  # (loop, layer) in execution order
    T: int  # number of units (loops, or virtual blocks of a dense model)
    looped: bool
    unit: dict = field(default_factory=dict)  # site -> unit (Dory input stack -1, output stack T)
    order: dict = field(default_factory=dict)  # site -> execution index
    boundaries: list = field(default_factory=list)  # first site of each unit, then FINAL
    unit_layers: list = field(default_factory=list)  # per unit: layer indices (same for every loop)

    def depth(self, site):
        return 1.0 if site == FINAL else self.order[site] / len(self.sites)

    def site_at(self, unit, frac):
        """Site at fraction ``frac`` of the layers of ``unit``."""
        layers = self.unit_layers[unit]
        layer = layers[min(int(round(frac * len(layers))), len(layers) - 1)]
        return (unit if self.looped else 0, layer)


def structure(model, blocks=4) -> Structure:
    cfg = model.config
    T = n_loops(cfg)
    sites = model_sites(cfg)
    L = len(decoder_layers(model))
    if T > 1:
        if getattr(cfg, "model_type", None) == "dory":
            n_in, n_rec = cfg.n_input_layers, cfg.n_recurrent_layers
            rec = list(range(n_in, n_in + n_rec))
            unit = {s: (s[0] if n_in <= s[1] < n_in + n_rec else (-1 if s[1] < n_in else T)) for s in sites}
        else:
            rec = list(range(L))
            unit = {s: s[0] for s in sites}
        st = Structure(sites, T, True, unit, {s: i for i, s in enumerate(sites)},
                       [(t, rec[0]) for t in range(T)] + [FINAL], [rec] * T)
    else:
        size = L // blocks
        layers = [list(range(k * size, L if k == blocks - 1 else (k + 1) * size)) for k in range(blocks)]
        unit = {s: min(s[1] // size, blocks - 1) for s in sites}
        st = Structure(sites, blocks, False, unit, {s: i for i, s in enumerate(sites)},
                       [(0, ls[0]) for ls in layers] + [FINAL], layers)
    return st


def site_key(model, layer, kwargs):
    return (_loop(kwargs), layer)


@contextmanager
def stream_hooks(model, on_site=None, on_final=None):
    """Pre-hooks on every decoder layer: ``on_site((loop, layer), h) -> new h | None`` (called with the layer input,
    in execution order), and on the unembedding input: ``on_final(h) -> new h | None``. Hooks registered by an earlier
    ``stream_hooks`` run first (register injections before recorders)."""
    blocks = decoder_layers(model)

    def make(layer):
        def hook(module, args, kwargs):
            h = _layer_input(args, kwargs)
            new = on_site((_loop(kwargs), layer), h)
            return None if new is None else _replace_layer_input(args, kwargs, new)
        return hook

    def fin(module, args):
        new = on_final(args[0])
        return None if new is None else (new, *args[1:])

    def register():
        hs = []
        if on_site is not None:
            hs += [b.register_forward_pre_hook(make(l), with_kwargs=True) for l, b in enumerate(blocks)]
        if on_final is not None:
            hs.append(model.lm_head.register_forward_pre_hook(fin))
        return hs

    with hooked(register):
        yield


OURO_THOUGHT = "<think>\nOkay, I'll answer the user's question directly.\n</think>\n\n"


def no_think_prompt(tok, messages, ouro_thought=OURO_THOUGHT):
    """Chat prompt that makes the model answer without thinking: the template's own ``enable_thinking=False`` (a closed
    empty thinking block: Qwen3, Nanbeige, Dory). Ouro-Thinking has no such switch and ends its turn at once after an
    empty block; it gets a one-line closed thought (``ouro_thought``; 39 of 40 persona questions answered) or, with
    ``ouro_thought=None``, the plain template (it answers "Who are you?" directly, but thinks on harder questions)."""
    try:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if ouro_thought and "</think>" not in text.rsplit("assistant", 1)[-1]:
        text += ouro_thought
    return text


def pad_batch(seqs, pad_id):
    """Right-pad token-id lists -> (ids [B, S], valid mask [B, S]). Right padding keeps causal positions exact."""
    S = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), S), pad_id, dtype=torch.long)
    valid = torch.zeros(len(seqs), S, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        valid[i, : len(s)] = True
    return ids, valid


def load_dirfile(rd, spec):
    """``file:method:site`` (site ``t3.l16`` or ``l21``) -> raw vector from ``directions/<file>.pt``."""
    import re

    fname, method, site = spec.split(":")
    d = torch.load(rd / "directions" / f"{fname}.pt", weights_only=False)
    m = re.fullmatch(r"(?:t(\d+)\.)?l(\d+)", site)
    key = (int(m.group(1) or 0), int(m.group(2)))
    return d[method]["dirs"][d["sites"].index(key)].float()


DEFAULT_U = {  # the refusal direction used for each model's headline hook ablation
    "Ouro-1.4B-Thinking": "train:v4_baseline:t3.l16",
    "Nanbeige4.2-3B": "train:v4_baseline:t1.l15",
    "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16": "train:hh_eoi:t0.l26",
    "Qwen3-8B": "train:v4_baseline:l21",
    "gpt-oss-20b": "train:v4_baseline:l15",
}
