"""Portfolio weight constraints.

Currently one constraint: a per-name cap on absolute weight with pro-rata
redistribution of the clipped excess. Rows (dates) are independent, and the
two sign sides of a row are treated independently, so a dollar-neutral book
stays dollar-neutral whenever both sides are feasible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.core.errors import ConfigError


def cap_weights(
    weights: pd.DataFrame,
    max_weight: float,
    n_iter: int = 50,
    tol: float = 1e-12,
) -> pd.DataFrame:
    """Cap ``|w|`` at ``max_weight`` per row, independently per sign side.

    For each row and each side (positive weights, negative weights):

    * names with ``|w| > max_weight`` are clipped to the cap and the clipped
      excess is redistributed pro-rata (proportional to current weight) among
      the uncapped names of the same side, iterating to a fixed point — so
      the side's gross is preserved whenever it is feasible;
    * if the side is infeasible (``n_names_on_side * max_weight`` is less
      than the side's gross), no redistribution can satisfy the cap: every
      active name on that side is set to the cap, i.e. the side is scaled
      down to ``n * max_weight`` and the row's TOTAL GROSS SHRINKS. This is
      deliberate documented behaviour, not an error.

    NaN or non-finite weights are treated as no position and come back 0.0.
    Returns a new DataFrame with the same index/columns as ``weights``.
    """
    if not isinstance(weights, pd.DataFrame):
        raise ConfigError("cap_weights expects a DataFrame of weights")
    if max_weight <= 0:
        raise ConfigError("max_weight must be positive")
    if n_iter < 1:
        raise ConfigError("n_iter must be >= 1")

    values = weights.to_numpy(dtype=float, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(values)
    # Rows are independent; a clear per-row loop beats a clever vectorization.
    for i in range(values.shape[0]):
        row = values[i]
        pos = np.where(row > 0.0, row, 0.0)
        neg = np.where(row < 0.0, -row, 0.0)
        out[i] = (
            _cap_side(pos, max_weight, n_iter, tol)
            - _cap_side(neg, max_weight, n_iter, tol)
        )
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)


def _cap_side(mags: np.ndarray, cap: float, n_iter: int, tol: float) -> np.ndarray:
    """Cap one sign side, given as non-negative magnitudes.

    Preserves the side sum when feasible; in the infeasible case the fixed
    point of clip-and-redistribute is every active name at the cap, which is
    exactly 'scale the side down to n * cap' — the excess has nowhere to go
    and is dropped.
    """
    active = mags > 0.0
    n = int(active.sum())
    total = float(mags.sum())
    if n == 0 or total <= 0.0:
        return mags
    if n * cap < total - tol:
        capped = np.zeros_like(mags)
        capped[active] = cap
        return capped
    w = mags.copy()
    at_cap = np.zeros(w.shape, dtype=bool)
    # Each pass either finishes or moves >= 1 new name into the capped set,
    # so this terminates in at most n_active passes.
    for _ in range(n_iter):
        over = w > cap + tol
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        at_cap |= over
        free = active & ~at_cap
        free_sum = float(w[free].sum())
        if free_sum <= tol:
            break
        w[free] += excess * w[free] / free_sum
    np.minimum(w, cap, out=w)  # shave tol-level residue at the fixed point
    return w
