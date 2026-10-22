"""vLLM implementation of Nemotron-Labs Dory 1.0 (a looped normalized GPT), for vLLM 0.18.

NVIDIA's implementation from vllm-project/vllm#60149 (commit ce9dd0e, not yet in a vLLM release) with three changes:
the config class is ``loop_steer.dory.DoryConfig`` (transformers 4.57), the backbone is ``self.model`` (checkpoint
``backbone.*`` names are mapped on load) so ``model.model.layers`` holds the sublayers as for the other models, and
every sublayer is called as ``layer(positions, hidden_states, loop_idx)`` with the loop index as a positional argument
(input and output sublayers: 0), which ``loop_steer.vllm_hooks`` reads. Math, weight loading and the per-loop KV
caches are unchanged. See ``loop_steer.dory`` for the architecture.

Register with ``register()`` in the process that builds the engine, and run vLLM with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the engine core shares that process (``loop_steer.models.prepare_vllm``).
"""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.parameter import PerTensorScaleParameter

from loop_steer.dory import justnorm, slerp_alpha


class DoryAttention(nn.Module):
    """GQA attention with nGPT qk-norm, output gate, and fngpt justnorm."""

    def __init__(self, config, layer_idx, cache_config, quant_config, prefix, num_loops=1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        std = config.initializer_range
        self.qkv_proj = QKVParallelLinear(self.hidden_size, self.head_dim, self.num_heads, self.num_kv_heads,
                                          bias=False, quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.g_proj = ColumnParallelLinear(self.hidden_size, self.q_size, bias=False, quant_config=quant_config,
                                           prefix=f"{prefix}.g_proj")
        self.o_proj = RowParallelLinear(self.q_size, self.hidden_size, bias=False, quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.sqk = nn.Parameter(torch.empty(self.q_size))
        self.s_gate = nn.Parameter(torch.empty(self.q_size))
        self._sqk_factor = self.head_dim ** 0.5 / std
        g0 = config.attention_output_gate_logit_scale_init
        if g0 is None:
            g0 = config.hidden_size ** 0.5
        self._s_gate_factor = g0 ** 0.5 / std
        layer_type = config.layer_types[layer_idx]
        self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings,
                                   rope_parameters=config.rope_parameters[layer_type], is_neox_style=True)
        self.per_loop_kv = config.recurrent_kv_cache_mode == "per_loop"
        # One Attention (KV-cache slot) per loop: virtual layer index loop * num_hidden_layers + layer_idx.
        # Projection prefixes keep the physical layer index to share weights.
        self.attn = nn.ModuleList()
        for loop_idx in range(num_loops if self.per_loop_kv else 1):
            unique_layer_idx = loop_idx * config.num_hidden_layers + layer_idx
            unique_prefix = prefix.replace(f".layers.{layer_idx}.", f".layers.{unique_layer_idx}.")
            self.attn.append(Attention(
                self.num_heads, self.head_dim, scale=1.0 / math.sqrt(self.head_dim), num_kv_heads=self.num_kv_heads,
                cache_config=cache_config, quant_config=quant_config,
                per_layer_sliding_window=config.sliding_window if layer_type == "sliding_attention" else None,
                prefix=f"{unique_prefix}.attn"))

    def forward(self, positions, hidden_states, loop_idx=0):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate, _ = self.g_proj(hidden_states)
        q, k = self.rotary_emb(positions, q, k)
        # nGPT qk-norm AFTER rope: q scaled by (sqk*f)^2 post-justnorm, k plain justnorm
        t = q.shape[0]
        q = q.view(t, self.num_heads, self.head_dim)
        sqk = (self.sqk.float() * self._sqk_factor).view(1, self.num_heads, self.head_dim)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype).view(t, self.q_size)
        k = justnorm(k.view(t, self.num_kv_heads, self.head_dim)).view(t, self.kv_size)
        out = self.attn[loop_idx if self.per_loop_kv else 0](q, k, v)
        # output gate (sigmoid, pre-o_proj), then fngpt justnorm over heads x head_dim
        sg = (self.s_gate.float() * self._s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        out, _ = self.o_proj(justnorm(out))
        return out


class DoryMLP(nn.Module):
    """SwiGLU with nGPT suv scaling and fngpt justnorm."""

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.ffn = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, [self.ffn] * 2, bias=False,
                                                       quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(self.ffn, config.hidden_size, bias=False, quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.suv_gate = nn.Parameter(torch.empty(self.ffn))
        self.suv_up = nn.Parameter(torch.empty(self.ffn))
        self._suv_factor = config.hidden_size ** 0.5

    def forward(self, hidden_states):
        m, _ = self.gate_up_proj(hidden_states)
        g, u = m.chunk(2, dim=-1)
        g = (g.float() * (self.suv_gate.float() * self._suv_factor)).to(g.dtype)
        u = (u.float() * (self.suv_up.float() * self._suv_factor)).to(u.dtype)
        m = (F.silu(g.float()) * u.float()).to(m.dtype)
        out, _ = self.down_proj(justnorm(m))
        return out


class DoryDecoderLayer(nn.Module):
    """One physical layer: a single sublayer (attention OR mlp) + slerp residual."""

    def __init__(self, config, layer_idx, cache_config, quant_config, prefix, num_loops=1):
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] != "mlp"
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.is_attention:
            self.mixer = DoryAttention(config, layer_idx, cache_config, quant_config, f"{prefix}.mixer",
                                       num_loops=num_loops)
            self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
        else:
            self.mixer = DoryMLP(config, quant_config, f"{prefix}.mixer")
            self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))

    @property
    def alpha(self):
        return self.attn_alpha if self.is_attention else self.mlp_alpha

    def lr(self):
        return (self.alpha.float() * self._alpha_scale).abs()

    def forward(self, positions, hidden_states, loop_idx=0):
        out = self.mixer(positions, hidden_states, loop_idx=loop_idx) if self.is_attention else self.mixer(hidden_states)
        return slerp_alpha(hidden_states, out, self.alpha, self._alpha_scale)


@support_torch_compile
class DoryBackbone(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config, quant_config = vllm_config.cache_config, vllm_config.quant_config
        self.config = config
        self.num_loops = config.n_recurrent_loops
        n_in, n_rec = config.n_input_layers, config.n_recurrent_layers
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DoryDecoderLayer(config, i, cache_config, quant_config, f"{prefix}.layers.{i}",
                             num_loops=self.num_loops if n_in <= i < n_in + n_rec else 1)
            for i in range(config.num_hidden_layers)])

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embeddings(input_ids)
        n_in, n_rec = self.config.n_input_layers, self.config.n_recurrent_layers
        for layer in self.layers[:n_in]:
            h = layer(positions, h, 0)
        input_h = state = h
        for i in range(self.num_loops):
            h = justnorm(state + input_h)
            for layer in self.layers[n_in: n_in + n_rec]:
                h = layer(positions, h, i)
            state = justnorm(h)
        h = state
        for layer in self.layers[n_in + n_rec:]:
            h = layer(positions, h, 0)
        return h


class DoryForCausalLM(nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        config.validate()  # hf_overrides can change the loop count after the config was built
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Dory currently supports tensor_parallel_size=1 only")
        self.model = DoryBackbone(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embeddings(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            logits = logits.float() * (self.lm_head.logit_scale.float() / self.config.initializer_range)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [  # (target, checkpoint shard name, shard id)
            ("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            if name.startswith("backbone."):
                name = "model." + name[len("backbone."):]
            mapped = False
            for target, shard_name, shard_id in stacked:
                if f".{shard_name}." not in name:
                    continue
                pname = name.replace(f".{shard_name}.", f".{target}.")
                param = params[pname]
                if target == "qkv_proj":
                    heads = self.config.num_attention_heads if shard_id == "q" else self.config.num_key_value_heads
                    rows = heads * self.config.head_dim
                else:
                    rows = self.config.intermediate_size
                if isinstance(param, PerTensorScaleParameter):
                    shape = () if w.ndim == 0 else (1,)
                else:
                    shape = (rows, *param.shape[1:])  # packed input columns and per-block scale columns
                self._check_weight_shape(name, w, shape)
                param.weight_loader(param, w, shard_id)
                loaded.add(pname)
                mapped = True
                break
            if mapped:
                continue
            param = params[name]
            shape = ((self.config.vocab_size, self.config.hidden_size)
                     if name in ("model.embeddings.weight", "lm_head.weight") else tuple(param.shape))
            if isinstance(param, PerTensorScaleParameter) and w.ndim == 0:
                shape = ()
            self._check_weight_shape(name, w, shape)
            getattr(param, "weight_loader", default_weight_loader)(param, w)
            loaded.add(name)
        return loaded

    @staticmethod
    def _check_weight_shape(name, weight, expected):
        if tuple(weight.shape) != expected:  # TP loaders may narrow oversized tensors; reject incompatible exports
            raise ValueError(f"Dory weight {name}: expected shape {expected}, got {tuple(weight.shape)}")


def register() -> None:
    """Make vLLM resolve ``DoryForCausalLM`` (and transformers the ``dory`` config) to our modules."""
    from vllm import ModelRegistry

    from loop_steer.dory import register as register_hf

    register_hf()
    ModelRegistry.register_model("DoryForCausalLM", "loop_steer.vllm_dory:DoryForCausalLM")
