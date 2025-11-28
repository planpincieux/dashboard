import os

import h5py
import numpy as np
from django.http import Http404
from ppcx_app.models import DIC


def load_and_filter_dic(
    dic: DIC,
    *,
    filter_outliers: bool = True,
    tails_percentile: float = 0.01,
    min_velocity: float | None = None,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read DIC HDF5 for `dic` and apply filtering + subsampling.

    Returns: x, y, u, v, mag (all numpy arrays float).
    Raises Http404 on errors or missing required datasets.
    """
    h5 = dic.result_file_path
    if not h5 or not os.path.exists(h5):
        raise Http404("DIC HDF5 file not found")

    try:
        with h5py.File(h5, "r") as f:
            points = f["points"][()] if "points" in f else None
            vectors = f["vectors"][()] if "vectors" in f else None
            magnitudes = f["magnitudes"][()] if "magnitudes" in f else None
    except Exception as e:
        raise Http404(f"Could not read DIC HDF5 file: {e}") from e

    if points is None or magnitudes is None:
        raise Http404("DIC HDF5 missing required datasets ('points' or 'magnitudes')")

    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    if vectors is not None:
        u = vectors[:, 0].astype(float)
        v = vectors[:, 1].astype(float)
    else:
        u = np.zeros_like(x)
        v = np.zeros_like(x)
    mag = magnitudes.astype(float)

    # Build mask and apply filters
    mask = np.ones_like(mag, dtype=bool)
    if filter_outliers and 0.0 < tails_percentile < 0.5:
        lo, hi = np.percentile(
            mag, [100.0 * tails_percentile, 100.0 * (1.0 - tails_percentile)]
        )
        mask &= (mag >= lo) & (mag <= hi)
    if min_velocity is not None and min_velocity > 0:
        mask &= mag >= min_velocity

    x = x[mask]
    y = y[mask]
    u = u[mask]
    v = v[mask]
    mag = mag[mask]

    if subsample is None or subsample < 1:
        subsample = 1
    if subsample > 1:
        idx = np.arange(0, len(x), subsample)
        x = x[idx]
        y = y[idx]
        u = u[idx]
        v = v[idx]
        mag = mag[idx]

    return x, y, u, v, mag
