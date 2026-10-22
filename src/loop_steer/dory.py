"""Nemotron-Labs Dory 1.0 (4B): config and a plain PyTorch implementation for transformers 4.57.

Transformers has no Dory model (none in 5.20 either) and the checkpoint ships no modeling code, so this module
provides both, following NVIDIA's vLLM implementation (vllm-project/vllm#60149, commit ce9dd0e) line by line. It is
used for teacher-forced passes (activations, mechanism checks); generation runs in vLLM (``loop_steer.vllm_dory``).
No KV cache: ``generate`` is not supported.

Architecture (a looped normalized GPT). The 72 physical "layers" are sublayers, each attention-only or MLP-only:
16 input sublayers, 40 recurrent sublayers, 16 output sublayers::

    h = embed(ids); h = input sublayers(h); x = h
    state = x
    for t in range(n_recurrent_loops):            # 2 as trained; 1-4 supported ("elastic depth")
        h = justnorm(state + x)                   # the input is re-injected every loop
        h = recurrent sublayers(h, loop t)
        state = justnorm(h)
    h = output sublayers(state); logits = lm_head(h) * logit_scale / initializer_range

A sublayer computes ``y = mixer(h)`` (GQA attention with qk-norm and a sigmoid output gate, or SwiGLU; both apply a
gainless ``justnorm`` right before their output projection) and updates the stream by a normalized interpolation,
``h <- justnorm(justnorm(h) + lr * (justnorm(y) - justnorm(h)))`` with a learned per-dimension ``lr``. The stream stays
on the unit sphere; there is no RMSNorm gain anywhere, but ``lr`` is per-dimension.

Hook conventions (``loop_steer.hooks``): ``model.model.layers`` are the 72 sublayers, each called as
``layer(hidden_states, positions=..., loop_idx=t)``; the mixer (``layer.mixer``) output is the residual write.
Input and output sublayers run once and report ``loop_idx=0``.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

EPS = 1e-6


class DoryConfig(PretrainedConfig):
    model_type = "dory"

    def __init__(self, vocab_size=131072, hidden_size=2560, num_hidden_layers=72, intermediate_size=7168,
                 num_attention_heads=16, num_key_value_heads=4, head_dim=256, max_position_embeddings=8192,
                 tie_word_embeddings=False, hidden_act="silu", initializer_range=0.02, sliding_window=1024,
                 rope_parameters=None, layer_types=None, ngpt_alpha_init=0.05,
                 attention_output_gate_logit_scale_init=None, mlp_bias=False, attention_bias=False, n_input_layers=0,
                 n_recurrent_layers=None, n_output_layers=0, n_recurrent_loops=1, recurrent_kv_cache_mode="per_loop",
                 **kwargs):
        self.vocab_size, self.hidden_size, self.num_hidden_layers = vocab_size, hidden_size, num_hidden_layers
        self.intermediate_size, self.head_dim = intermediate_size, head_dim
        self.num_attention_heads, self.num_key_value_heads = num_attention_heads, num_key_value_heads
        self.max_position_embeddings, self.hidden_act = max_position_embeddings, hidden_act
        self.initializer_range, self.sliding_window = initializer_range, sliding_window
        self.ngpt_alpha_init = ngpt_alpha_init
        self.attention_output_gate_logit_scale_init = attention_output_gate_logit_scale_init
        self.mlp_bias, self.attention_bias = mlp_bias, attention_bias
        self.n_input_layers, self.n_output_layers = n_input_layers, n_output_layers
        self.n_recurrent_layers = (num_hidden_layers - n_input_layers - n_output_layers
                                   if n_recurrent_layers is None else n_recurrent_layers)
        self.n_recurrent_loops, self.recurrent_kv_cache_mode = n_recurrent_loops, recurrent_kv_cache_mode
        self.layer_types = layer_types or ["mlp" if i % 2 else "full_attention" if i % 8 == 6 else "sliding_attention"
                                           for i in range(num_hidden_layers)]
        self.rope_parameters = rope_parameters or {
            kind: {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}
            for kind in ("sliding_attention", "full_attention") if kind in self.layer_types}
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def validate(self):
        """The checks NVIDIA's config runs that matter here (vLLM's ``hf_overrides`` can change the loop count)."""
        parts = (self.n_input_layers, self.n_recurrent_layers, self.n_output_layers)
        if min(parts) < 0 or sum(parts) != self.num_hidden_layers:
            raise ValueError("Dory input, recurrent and output layers must sum to num_hidden_layers")
        if self.n_recurrent_loops < 1:
            raise ValueError("n_recurrent_loops must be a positive integer")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        if self.attention_bias or self.mlp_bias or self.hidden_act != "silu" or self.tie_word_embeddings:
            raise ValueError("Dory requires bias-free projections, SiLU and untied embeddings")


def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """nGPT normalization: x / |x| (no gain), accumulated in fp32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)).to(x.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
    """The residual update: justnorm(â + lr * (b̂ - â)), lr = |alpha * alpha_scale| per dimension."""
    lr = (alpha.float() * alpha_scale).abs()
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr * (yb - ya)).to(a.dtype)


def _rope(x, cos, sin):
    """Neox-style rotary embedding on the last axis (``[..., T, H, D]``; cos/sin ``[..., T, 1, D/2]``)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


class DoryAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        H, Hk, hd, D = config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.hidden_size
        self.num_heads, self.num_kv_heads, self.head_dim = H, Hk, hd
        self.q_proj = nn.Linear(D, H * hd, bias=False)
        self.k_proj = nn.Linear(D, Hk * hd, bias=False)
        self.v_proj = nn.Linear(D, Hk * hd, bias=False)
        self.g_proj = nn.Linear(D, H * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, D, bias=False)
        self.sqk = nn.Parameter(torch.empty(H * hd))
        self.s_gate = nn.Parameter(torch.empty(H * hd))
        std = config.initializer_range
        self._sqk_factor = hd ** 0.5 / std
        g0 = config.attention_output_gate_logit_scale_init
        g0 = config.hidden_size ** 0.5 if g0 is None else g0
        self._s_gate_factor = g0 ** 0.5 / std
        self.layer_type = config.layer_types[layer_idx]
        theta = config.rope_parameters[self.layer_type]["rope_theta"]
        self.register_buffer("inv_freq", 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd),
                             persistent=False)

    def forward(self, h, positions, masks):
        B, T, _ = h.shape
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim)
        gate = self.g_proj(h)
        freqs = positions.float()[..., None] * self.inv_freq  # [B, T, hd/2]
        cos, sin = freqs.cos()[:, :, None], freqs.sin()[:, :, None]
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        # nGPT qk-norm after rope: q scaled by (sqk * f)^2 after justnorm, k plain justnorm
        sqk = (self.sqk.float() * self._sqk_factor).view(1, 1, self.num_heads, self.head_dim)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype)
        k = justnorm(k)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=masks[self.layer_type], scale=1.0 / math.sqrt(self.head_dim),
                                             enable_gqa=True)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        sg = (self.s_gate.float() * self._s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        return self.o_proj(justnorm(out))


class DoryMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        D, F_ = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(D, F_, bias=False)
        self.up_proj = nn.Linear(D, F_, bias=False)
        self.down_proj = nn.Linear(F_, D, bias=False)
        self.suv_gate = nn.Parameter(torch.empty(F_))
        self.suv_up = nn.Parameter(torch.empty(F_))
        self._suv_factor = D ** 0.5

    def forward(self, h):
        g = (self.gate_proj(h).float() * (self.suv_gate.float() * self._suv_factor)).to(h.dtype)
        u = (self.up_proj(h).float() * (self.suv_up.float() * self._suv_factor)).to(h.dtype)
        m = (F.silu(g.float()) * u.float()).to(h.dtype)
        return self.down_proj(justnorm(m))


class DoryDecoderLayer(nn.Module):
    """One sublayer (attention OR mlp) + the normalized interpolation into the stream."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] != "mlp"
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.is_attention:
            self.mixer = DoryAttention(config, layer_idx)
            self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
        else:
            self.mixer = DoryMLP(config)
            self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))

    @property
    def alpha(self):
        return self.attn_alpha if self.is_attention else self.mlp_alpha

    def lr(self):
        """Per-dimension interpolation weight of the write (``[D]``, fp32)."""
        return (self.alpha.float() * self._alpha_scale).abs()

    def forward(self, hidden_states, positions=None, masks=None, loop_idx=0):
        out = self.mixer(hidden_states, positions, masks) if self.is_attention else self.mixer(hidden_states)
        return slerp_alpha(hidden_states, out, self.alpha, self._alpha_scale)


class DoryBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_loops = config.n_recurrent_loops
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DoryDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def _masks(self, attention_mask, B, T, device):
        """Boolean [B, 1, T, T] masks per attention type: causal (+ sliding window), padded keys excluded."""
        i = torch.arange(T, device=device)
        causal = i[None, :] <= i[:, None]
        masks = {"full_attention": causal, "sliding_attention": causal & (i[None, :] > i[:, None] - self.config.sliding_window)}
        keys = torch.ones(B, T, dtype=torch.bool, device=device) if attention_mask is None else attention_mask.bool()
        eye = torch.eye(T, dtype=torch.bool, device=device)  # a padded query always sees itself: no all-masked row
        return {k: (m[None, None] & keys[:, None, None, :]) | eye for k, m in masks.items()}

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **_):
        h = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        B, T, _ = h.shape
        if attention_mask is None:
            positions = torch.arange(T, device=h.device).expand(B, T)
        else:  # works for left and right padding
            positions = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        masks = self._masks(attention_mask, B, T, h.device)
        c = self.config
        n_in, n_rec = c.n_input_layers, c.n_recurrent_layers
        for layer in self.layers[:n_in]:
            h = layer(h, positions=positions, masks=masks, loop_idx=0)
        input_h = state = h
        for t in range(self.num_loops):
            h = justnorm(state + input_h)
            for layer in self.layers[n_in: n_in + n_rec]:
                h = layer(h, positions=positions, masks=masks, loop_idx=t)
            state = justnorm(h)
        h = state
        for layer in self.layers[n_in + n_rec:]:
            h = layer(h, positions=positions, masks=masks, loop_idx=0)
        return BaseModelOutput(last_hidden_state=h)


class DoryLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        self.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.initializer_range = config.initializer_range

    def forward(self, h):
        return F.linear(h, self.weight).float() * (self.logit_scale.float() / self.initializer_range)


class DoryForCausalLM(PreTrainedModel):
    config_class = DoryConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["DoryDecoderLayer"]
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.backbone = DoryBackbone(config)
        self.lm_head = DoryLMHead(config)
        self.post_init()

    @property
    def model(self):  # loop_steer convention: ``model.model.layers``
        return self.backbone

    def _init_weights(self, module):  # every weight comes from the checkpoint
        pass

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **_):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds).last_hidden_state
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=64, do_sample=False, temperature=1.0,
                 pad_token_id=None, eos_token_id=None, **_):
        """Minimal no-cache decoding for short validation runs (recomputes the whole sequence every step).

        Left-padded batches; greedy unless ``do_sample``. Finished rows are padded with ``pad_token_id``. Returns the
        prompt and the new tokens, ``[B, T + max_new_tokens]`` at most.
        """
        if eos_token_id is None:  # not a GenerationMixin, so from_pretrained does not attach the generation config
            from transformers import GenerationConfig

            gen = GenerationConfig.from_pretrained(self.name_or_path, revision=getattr(self.config, "_commit_hash", None))
            eos_token_id = gen.eos_token_id
        eos = eos_token_id
        eos = torch.tensor([eos] if isinstance(eos, int) else list(eos), device=input_ids.device)
        pad = pad_token_id if pad_token_id is not None else int(eos[0])
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        ids, done = input_ids, torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            logits = self(input_ids=ids, attention_mask=mask).logits[:, -1]
            nxt = (torch.multinomial(torch.softmax(logits / temperature, -1), 1)[:, 0] if do_sample
                   else logits.argmax(-1))
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            mask = torch.cat([mask, (~done).long()[:, None]], 1)
            done |= torch.isin(nxt, eos)
            if done.all():
                break
        return ids


def register():
    """Make AutoConfig / AutoModelForCausalLM resolve ``model_type: dory`` to this module (idempotent)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        AutoConfig.register("dory", DoryConfig)
    except ValueError:  # already registered
        pass
    try:
        AutoModelForCausalLM.register(DoryConfig, DoryForCausalLM)
    except ValueError:
        pass
