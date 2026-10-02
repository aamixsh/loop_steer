"""Weight orthogonalization (Arditi et al.; reasoning-manipulation v4).

Projects a direction out of every matrix that writes to the residual stream:
token embeddings, each ``self_attn.o_proj`` and each ``mlp.down_proj``. Valid
only for pre-norm models whose sublayer outputs are added to the residual
unnormalized (Qwen/Llama). NOT valid for Ouro (sandwich norms re-scale writes).

Works on both HF and vLLM module trees (same ``model.layers`` naming). The
first call snapshots the original weights on CPU so candidates can be swapped
in place without reloading.
"""

import torch


def _residual_writers(model):
    inner = model.model
    yield "embed", inner.embed_tokens.weight, "rows"  # [vocab, d]: rows live in the residual
    for i, layer in enumerate(inner.layers):
        yield f"o_proj.{i}", layer.self_attn.o_proj.weight, "cols"  # [d, in]
        yield f"down_proj.{i}", layer.mlp.down_proj.weight, "cols"


@torch.no_grad()
def orthogonalize_(model, direction: torch.Tensor | None):
    """Restore original weights, then (if ``direction`` is not None) project it out in place."""
    if not hasattr(model, "_ls_original"):
        model._ls_original = {name: w.detach().to("cpu", copy=True) for name, w, _ in _residual_writers(model)}
    for name, w, kind in _residual_writers(model):
        orig = model._ls_original[name]
        if direction is None:
            w.copy_(orig)
            continue
        u = (direction / direction.norm()).to(device=w.device, dtype=torch.float32)
        W = orig.to(device=w.device, dtype=torch.float32)
        W = W - torch.outer(W @ u, u) if kind == "rows" else W - torch.outer(u, u @ W)
        w.copy_(W.to(w.dtype))
    return True
