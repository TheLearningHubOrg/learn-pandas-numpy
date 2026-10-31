"""Vectorized NumPy building blocks: no Python-level loops over elements."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize along an axis. Constant slices map to 0 instead of NaN."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    safe_std = np.where(std == 0, 1.0, std)
    return (x - mean) / safe_std


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over a fixed window, built from a zero-copy strided view."""
    x = np.asarray(x, dtype=np.float64)
    if window < 1 or window > x.shape[-1]:
        raise ValueError(f"window must be in [1, {x.shape[-1]}], got {window}")
    return sliding_window_view(x, window, axis=-1).mean(axis=-1)


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of a (n, d) and every row of b (m, d).

    Broadcasting (n, 1, d) against (1, m, d) yields an (n, m, d) difference cube.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"expected (n, d) and (m, d) arrays, got {a.shape} and {b.shape}")
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def normalize_rows(m: np.ndarray) -> np.ndarray:
    """Scale each row to sum to 1. All-zero rows stay zero."""
    m = np.asarray(m, dtype=np.float64)
    totals = m.sum(axis=1, keepdims=True)
    return np.divide(m, totals, out=np.zeros_like(m), where=totals != 0)
