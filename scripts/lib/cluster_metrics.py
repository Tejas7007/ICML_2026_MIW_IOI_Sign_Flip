"""Crossed template/name-pair bootstrap used by the locked benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Sequence
import numpy as np


def crossed_cluster_bootstrap(
    values: Sequence[float],
    template_id: Sequence[int],
    pair_id: Sequence[str],
    *,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Two-way pigeonhole bootstrap over template and unordered name-pair clusters."""
    vals = np.asarray(values, dtype=float)
    tids = np.asarray(template_id)
    pids = np.asarray(pair_id, dtype=object)
    if not (len(vals) == len(tids) == len(pids)) or not len(vals):
        raise ValueError("values and cluster labels must be non-empty and aligned")
    templates = np.unique(tids)
    pairs = np.unique(pids)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        for _ in range(100):
            tc = Counter(rng.choice(templates, len(templates), replace=True).tolist())
            pc = Counter(rng.choice(pairs, len(pairs), replace=True).tolist())
            weights = np.fromiter(
                (tc.get(t, 0) * pc.get(p, 0) for t, p in zip(tids, pids)),
                dtype=float,
            )
            if weights.sum() > 0:
                out[draw] = np.average(vals, weights=weights)
                break
        else:
            raise RuntimeError("Could not draw a non-empty crossed bootstrap sample")
    return tuple(float(x) for x in np.quantile(out, [alpha/2, 1-alpha/2]))
