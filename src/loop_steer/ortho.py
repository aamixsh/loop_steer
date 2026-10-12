"""Weight orthogonalization (Arditi et al.; reasoning-manipulation v4).

Projects a direction out of every matrix that writes to the residual stream:
token embeddings, each ``self_attn.o_proj`` and each ``mlp.down_proj`` (MoE: every expert's down projection), and
their biases. Exact
for pre-norm models whose sublayer outputs are added to the residual
unnormalized (Qwen/Llama/Nanbeige): the edited matrices cannot write along the
direction at all.

Ouro adds ``RMSNorm_2(sublayer(x))`` instead (sandwich norms). Its write along
``u`` is ``<g * x/rms(x), u> = <x, g*u>/rms(x)`` where ``g`` is the norm's gain,
so projecting ``u`` out of ``x`` does NOT zero the write (``g`` is not uniform).
``norm_aware=True`` instead projects ``g*u`` (normalised) out of each writer's
output space, which does zero the write exactly. Neither variant covers the
shared norm that runs between loops (its gain re-introduces the direction into
the next loop's input; see ``scripts/ortho_compare.py``).

Works on both HF and vLLM module trees (same ``model.layers`` naming). The
first call snapshots the original weights on CPU so candidates can be swapped
in place without reloading.
"""

import torch


def _residual_writers(model):
    """Yield (name, tensor, kind, gain, is_bias) for every residual writer.

    ``kind`` says which axis of the tensor lives in the residual stream: ``rows`` = the last axis (embeddings
    ``[vocab, d]``, biases ``[d]`` / ``[experts, d]``, HF gpt-oss experts ``[experts, inter, d]``), ``cols`` = the
    second-to-last (Linear weights ``[d, in]``, vLLM fused-MoE ``w2`` ``[experts, d, inter]``). ``gain`` is the
    per-dimension gain of a norm applied to the writer's output before it is added to the residual (sandwich-norm
    models), or None. MoE (gpt-oss): every expert's down projection and bias is a writer, since the MoE output is the
    router-weighted sum of the selected experts' outputs; editing all of them makes every possible sum orthogonal.
    """
    inner = model.model
    embed = inner.embed_tokens if hasattr(inner, "embed_tokens") else inner.embedding  # vLLM gpt-oss: embedding
    yield "embed", embed.weight, "rows", None, False
    for i, layer in enumerate(inner.layers):
        attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn
        attn_norm = getattr(layer, "input_layernorm_2", None)
        mlp_norm = getattr(layer, "post_attention_layernorm_2", None)
        g_attn = None if attn_norm is None else attn_norm.weight
        g_mlp = None if mlp_norm is None else mlp_norm.weight
        yield f"o_proj.{i}", attn.o_proj.weight, "cols", g_attn, False
        if getattr(attn.o_proj, "bias", None) is not None:
            yield f"o_proj_bias.{i}", attn.o_proj.bias, "rows", g_attn, True
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            yield f"down_proj.{i}", layer.mlp.down_proj.weight, "cols", g_mlp, False
        elif hasattr(experts, "down_proj"):  # HF gpt-oss: x @ down_proj[e] + down_proj_bias[e]
            yield f"experts_down.{i}", experts.down_proj, "rows", g_mlp, False
            yield f"experts_down_bias.{i}", experts.down_proj_bias, "rows", g_mlp, True
        else:  # vLLM FusedMoE (unquantized): w2_weight [experts, d, inter], w2_bias [experts, d]
            yield f"experts_down.{i}", experts.w2_weight, "cols", g_mlp, False
            if getattr(experts, "w2_bias", None) is not None:
                yield f"experts_down_bias.{i}", experts.w2_bias, "rows", g_mlp, True


def loop_span(model, direction: torch.Tensor, n_loops: int) -> torch.Tensor:
    """Directions ``[u, g*u, g^2*u, ...]`` (``n_loops`` of them), ``g`` = gain of the shared norm run
    between loops.  A residual stream orthogonal to all of them stays orthogonal to ``u`` after
    ``n_loops - 1`` passes through that norm: the norm maps ``<h, g^k*u>`` to ``<out, g^(k-1)*u>``.
    """
    g = model.model.norm.weight.detach().float().cpu()
    u = direction.float().cpu()
    out = []
    for _ in range(n_loops):
        out.append(u)
        u = g * u
    return torch.stack(out)  # [n_loops, d]


@torch.no_grad()
def orthogonalize_(model, direction: torch.Tensor | None, norm_aware: bool = False, biases: bool = True):
    """Restore original weights, then (if ``direction`` is not None) project it out in place.

    ``direction``: ``[d]`` (one direction) or ``[k, d]`` (a subspace; see ``loop_span``).
    ``norm_aware``: for writers followed by a sandwich norm with gain ``g``, project out
    ``g * direction`` instead, so the *normed* write is orthogonal to ``direction``.
    ``biases``: also edit the writers' biases (gpt-oss has biases on ``o_proj`` and every expert). Without it a bias
    keeps writing a constant along the direction, so the edit is not an ablation (``biases=False`` reproduces the
    reasoning-manipulation gpt-oss edit, which leaves them).
    """
    if not hasattr(model, "_ls_original"):
        model._ls_original = {name: w.detach().to("cpu", copy=True) for name, w, *_ in _residual_writers(model)}
    for name, w, kind, gain, is_bias in _residual_writers(model):
        orig = model._ls_original[name]
        if direction is None or (is_bias and not biases):
            w.copy_(orig)
            continue
        D = direction.to(device=w.device, dtype=torch.float32)
        D = D.unsqueeze(0) if D.dim() == 1 else D  # [k, d]
        if norm_aware and gain is not None:
            D = gain.detach().float() * D
        Q = torch.linalg.qr(D.T).Q  # [d, k] orthonormal basis of the span
        size = w.shape[-1] if kind == "rows" else w.shape[-2]
        if size != Q.shape[0]:  # residual axis padded by the kernel layout: the extra dims are zero
            Q = torch.cat([Q, Q.new_zeros(size - Q.shape[0], Q.shape[1])])
        W = orig.to(device=w.device, dtype=torch.float32)
        W = W - (W @ Q) @ Q.T if kind == "rows" else W - Q @ (Q.T @ W)
        w.copy_(W.to(w.dtype))
    return True
