"""vLLM implementation of Nanbeige4.2-3B (a 2-loop, Llama-style LM) for inference.

Structure follows vLLM's own Ouro port: the same stack of decoder layers is applied
``num_loops`` times, each (loop, layer) pair gets its own ``Attention`` instance so it
has its own KV-cache slot (as in the HF code, slot = layer + loop * num_layers), and
the shared final RMSNorm runs after every loop (``skip_loop_final_norm=False``).
Layers are plain pre-norm Llama layers (GQA, SwiGLU, neox-style RoPE, no biases).
The head dimension is ``config.head_dim`` (128), not hidden_size // num_heads (64).

Register with ``register()`` in the process that builds the engine, and run vLLM with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the engine core shares that process.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, extract_layer_index, make_empty_intermediate_tensors_factory, make_layers, maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType


class NanbeigeMLP(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation {config.hidden_act!r}; only silu")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(x))
        return x


class NanbeigeAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, cache_config: CacheConfig | None,
                 quant_config: QuantizationConfig | None, prefix: str) -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp == 0
        self.num_heads = self.total_num_heads // tp
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.rotary_emb = get_rope(
            self.head_dim, max_position=config.max_position_embeddings, rope_parameters=config.rope_parameters)

        num_loops, total_layers = config.num_loops, config.num_hidden_layers
        base_idx = extract_layer_index(prefix)
        self.attn = nn.ModuleList()
        for loop in range(num_loops):
            unique_idx = loop * total_layers + base_idx
            unique_prefix = prefix.replace(f"layers.{base_idx}", f"layers.{unique_idx}")
            self.attn.append(Attention(
                self.num_heads, self.head_dim, scaling, num_kv_heads=self.num_kv_heads,
                cache_config=cache_config, quant_config=quant_config,
                attn_type=AttentionType.DECODER, prefix=f"{unique_prefix}.attn"))

    def forward(self, positions, hidden_states, loop_idx: int):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        out, _ = self.o_proj(self.attn[loop_idx](q, k, v))
        return out


class NanbeigeDecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str) -> None:
        super().__init__()
        self.self_attn = NanbeigeAttention(config, cache_config, quant_config, f"{prefix}.self_attn")
        self.mlp = NanbeigeMLP(config, quant_config, f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, loop_idx: int, residual):
        # Fused residual convention: residual stream entering the layer = hidden_states + residual.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, loop_idx)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(dynamic_arg_dims={
    "input_ids": 0, "positions": -1, "intermediate_tensors": 0, "inputs_embeds": 0})
class NanbeigeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config, quant_config = vllm_config.cache_config, vllm_config.quant_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, quant_config=quant_config, prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: NanbeigeDecoderLayer(config, cache_config, quant_config, prefix),
            prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.num_loops = config.num_loops
        if getattr(config, "skip_loop_final_norm", False):
            raise NotImplementedError("skip_loop_final_norm=True is not implemented in this port")

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        for loop_idx in range(self.num_loops):
            residual = None
            for layer in self.layers[self.start_layer: self.end_layer]:
                hidden_states, residual = layer(positions, hidden_states, loop_idx, residual)
            hidden_states, _ = self.norm(hidden_states, residual)  # shared norm after every loop
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v"),
                   ("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, shard_name, shard_id in stacked:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params[name]
                param.weight_loader(param, weight, shard_id)
                break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded


class NanbeigeForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = NanbeigeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None))
        return loader.load_weights(weights)


def register() -> None:
    """Make vLLM resolve the ``NanbeigeForCausalLM`` architecture to this module."""
    from vllm import ModelRegistry

    ModelRegistry.register_model("NanbeigeForCausalLM", "loop_steer.vllm_nanbeige:NanbeigeForCausalLM")
