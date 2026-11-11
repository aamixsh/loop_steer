"""Random kick directions shared by the generation (vLLM) and teacher-forced (HF) kick experiments."""

import torch


def problem_directions(seed: int, n: int, dim: int) -> torch.Tensor:
    """[n, dim] unit random directions, row i for problem i (its index in the evaluated problem list).

    Row i depends only on (seed, i, dim): the same problem gets the same direction at every site, kick size and loop
    count, and in both backends, so comparisons between conditions are paired by problem.
    """
    rows = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed * 1_000_003 + i + 1)
        v = torch.randn(dim, generator=g)
        rows.append(v / v.norm())
    return torch.stack(rows)
